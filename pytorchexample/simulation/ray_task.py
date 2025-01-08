import copy
import itertools
from typing import Dict, Optional, Union

import click
import ray
from flwr.common import (
    Context,
    EvaluateRes,
    FitRes,
    Parameters,
    RecordSet,
)
from flwr.common.typing import UserConfig
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr_datasets import FederatedDataset
from tqdm import tqdm
from pytorchexample.simulation.simulation import Simulation
from pytorchexample.common import ClientMethods, cleanup
from pytorchexample.task import load_fds
from pytorchexample.simulation.stubs import ClientProxyStub, ClientManagerStub

RAY_MAX_CALLS = 3


class RayTaskSimulator(Simulation):
    def __init__(self, config: dict):
        self._config = config
        self._fl_rounds = config["fl_rounds"]
        self._federation = config["federation"]
        self._num_clients = config["num_clients"]
        self._num_cpus_worker = config["num_cpus_worker"]
        self._num_gpus_worker = config["num_gpus_worker"]
        self._num_cpus_total = config["num_cpus_total"]
        self._num_gpus_total = config["num_gpus_total"]
        self._num_workers = config["num_workers"]
        self._max_memory_total = config["max_memory_total"]
        self._memory_worker = config["kwargs"]["max_memory_worker"] * 1024 * 1024
        self._client_fn = config["client_fn"]
        self._server_fn = config["server_fn"]
        self._run_id = config["kwargs"]["run_id"]
        self._server_node_id = 0
        self._ray_namespace = config["kwargs"]["ray_namespace"]
        self._ray_max_calls = RAY_MAX_CALLS

        ray.init(
            address=config["kwargs"]["ray_address"],
            num_cpus=self._num_cpus_total,
            num_gpus=self._num_gpus_total,
            include_dashboard=config["kwargs"]["ray_dashboard"],
            dashboard_host=config["kwargs"]["ray_dashboard_host"],
            dashboard_port=config["kwargs"]["ray_dashboard_port"],
            namespace=self._ray_namespace,
        )

    def run_loop(self):
        click.echo(
            click.style(
                "Starting Simulation Engine", bold=True, underline=True, fg="green"
            )
        )
        # Create server context
        server_node_config = {}
        server_state = RecordSet(
            parameters_records={}, metrics_records={}, configs_records={}
        )
        server_run_config = {
            **self._config["pyproject"]["tool"]["flwr"]["app"]["config"]
        }
        server_run_config["num-server-rounds"] = self._fl_rounds

        server_context = Context(
            run_id=self._run_id,
            node_id=self._server_node_id,
            node_config=server_node_config,
            state=server_state,
            run_config=server_run_config,
        )

        # Create client contexts
        client_states: Dict[int, RecordSet] = {
            i: RecordSet(parameters_records={}, metrics_records={}, configs_records={})
            for i in range(self._num_clients)
        }
        client_node_configs: Dict[int, UserConfig] = {
            i: {"partition-id": i, "num-partitions": self._num_clients}
            for i in range(self._num_clients)
        }
        client_configs = {
            i: copy.deepcopy(server_run_config) for i in range(self._num_clients)
        }
        client_contexts = {
            i: Context(
                self._run_id,
                i,
                client_node_configs[i],
                client_states[i],
                client_configs[i],
            )
            for i in range(self._num_clients)
        }
        # Create server
        fds = load_fds(self._num_clients)
        ray_fds = ray.put(fds)
        server = self._server_fn(server_context, ray.get(ray_fds))
        client_mng = ClientManagerStub(self._num_clients)

        init_params = server.strategy.initial_parameters  # type: ignore
        strategy: Strategy = server.strategy
        global_params = init_params
        # FL loop
        for server_round in range(1, self._fl_rounds + 1):
            click.echo(
                click.style(
                    f"Started Federated Learning Round: {server_round}",
                    italic=True,
                    fg="green",
                )
            )
            configure_fit_res = self._configure_fit(
                strategy, server_round, global_params, client_mng
            )
            fit_res, fit_failures = self._fit_clients(
                configure_fit_res, client_contexts, ray_fds
            )
            agg_params, agg_metrics = self._aggregate_fit(
                strategy, server_round, fit_res, fit_failures
            )
            click.echo(f"Aggregated Fit Metrics: {agg_metrics}")
            params, eval_res = self._evaluate(
                strategy, server_round, agg_params, global_params
            )
            global_params = params
            configure_eval_res = self._configure_evaluate_clients(
                strategy, server_round, copy.deepcopy(global_params), client_mng
            )
            eval_res, eval_failures = self._evaluate_clients(
                configure_eval_res, client_contexts, ray_fds
            )
            agg_loss, agg_metrics = self._aggregate_evaluate(
                strategy, server_round, eval_res, eval_failures
            )
            click.echo(f"Aggregated Eval Loss: {agg_loss} | Metrics: {agg_metrics}")
            click.echo(
                click.style(
                    f"Finished Federated Learning Round: {server_round}",
                    italic=True,
                    fg="green",
                )
            )
        self._finalize()

    def _configure_fit(
        self,
        strategy: Strategy,
        server_round: int,
        init_params: Parameters,
        client_mng: ClientManager,
    ):
        click.echo(click.style(" > Configure Fit", italic=True, fg="green"))
        return strategy.configure_fit(server_round, init_params, client_mng)

    def _fit_clients(self, configure_fit_res, contexts, fds: FederatedDataset):
        click.echo(click.style(" > Fit Clients", italic=True, fg="green"))
        return self._call_clients(configure_fit_res, contexts, fds, ClientMethods.FIT)

    def _evaluate_clients(self, configure_eval_res, contexts, fds: FederatedDataset):
        click.echo(click.style(" > Evaluate Clients", italic=True, fg="green"))
        return self._call_clients(
            configure_eval_res, contexts, fds, ClientMethods.EVALUATE
        )

    def _call_clients(
        self, configure_res, contexts, fds: FederatedDataset, method: ClientMethods
    ):
        with tqdm(
            total=self._num_clients,
            desc=f"Calling {method} on Clients",
            leave=True,
            unit="Client",
        ) as bar:
            tasks = {}
            obj_lists = {}

            # Check if parameters are going to be the same -> don't need to save multiple times in shared memory
            groups_ids = []  # grouped ids for same parameters e.g. [[1,2,3], [4,5,6]]
            for _k, g in itertools.groupby(
                configure_res, lambda cres: cres[1].parameters.tensors
            ):
                ids = [int(x[0].cid) for x in g]
                groups_ids.append(ids)

            # Call the clients
            dealt_with = []  # contains index of groups_id
            for c, ins in configure_res:
                id = int(c.cid)

                # get index of group
                g_id = self._get_gid(id, groups_ids)
                if g_id not in dealt_with:
                    params = ray.put(ins.parameters.tensors)
                    obj_lists[g_id] = params
                    dealt_with.append(g_id)
                ins.parameters.tensor_type = "ray.store"
                ins.parameters.tensors = obj_lists[g_id]
                tasks[
                    ray.remote(max_calls=self._ray_max_calls)(self._client_fn)
                    .options(
                        num_cpus=self._num_cpus_worker,
                        num_gpus=self._num_gpus_worker,
                        memory=self._memory_worker,
                    )
                    .remote(contexts[id], ins, method=method, fds=fds)
                ] = id
            results: list[tuple[ClientProxy, Union[FitRes, EvaluateRes]]] = []
            dealt_with = []
            remaining_refs = list(tasks.keys())
            while remaining_refs:
                ready_refs, remaining_refs = ray.wait(
                    remaining_refs, num_returns=1, timeout=None
                )
                for r in ready_refs:
                    result, new_context = ray.get(r)
                    results.append((ClientProxyStub(tasks[r]), result))
                    id = tasks[r]
                    contexts[id] = new_context
                    dealt_with.append(id)
                    g_id = self._get_gid(id, groups_ids)
                    if set(dealt_with) >= set(groups_ids[g_id]):
                        if obj_lists[g_id] is not None:
                            ray.internal.free([obj_lists[g_id]])
                            del obj_lists[g_id]
                        else:
                            pass
                    else:
                        pass
                    del tasks[r]
                    del r
                    bar.update(1)
            # assert all objects cleaned
            assert obj_lists == {}
            assert tasks == {}
        cleanup()
        # assume no failures happened
        return results, []

    def _get_gid(self, id, group_ids):
        for idx, group in enumerate(group_ids):
            if id in group:
                return idx
            else:
                continue
        return ValueError

    def _aggregate_fit(self, strategy: Strategy, server_round, results, failures):
        click.echo(click.style(" > Aggregate Fit", italic=True, fg="green"))
        return strategy.aggregate_fit(server_round, results, failures)

    def _aggregate_evaluate(self, strategy: Strategy, server_round, results, failures):
        click.echo(click.style(" > Aggregate Evaluate", italic=True, fg="green"))
        return strategy.aggregate_evaluate(server_round, results, failures)

    def _evaluate(
        self,
        strategy: Strategy,
        server_round,
        params: Optional[Parameters],
        global_params: Parameters,
    ):
        click.echo(click.style(" >Evaluate", italic=True, fg="green"))
        if params is None:
            params = global_params
        return params, strategy.evaluate(server_round, params)

    def _configure_evaluate_clients(
        self,
        strategy: Strategy,
        server_round,
        params: Parameters,
        client_mng: ClientManager,
    ):
        click.echo(click.style(" > Configure Evaluate", italic=True, fg="green"))
        return strategy.configure_evaluate(server_round, params, client_mng)

    def _finalize(self):
        click.echo(click.style("Shutdown Simulation Engine", bold=True, underline=True))
        ray.shutdown()
