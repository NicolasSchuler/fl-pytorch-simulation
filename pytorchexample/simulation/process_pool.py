import copy
import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Semaphore, connection, shared_memory
from typing import Dict, Optional, Union

import click
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
from tqdm import tqdm
from pytorchexample.simulation.simulation import Simulation
from pytorchexample.common import ClientMethods, cleanup
from pytorchexample.task import load_fds
from pytorchexample.simulation.stubs import ClientManagerStub, ClientProxyStub
from pytorchexample.simulation.client_shm import initializer, fire_and_retrieve

MAX_TASK_PER_CHILD = 5


class ProcessPoolSimulator(Simulation):
    def __init__(self, config: dict):
        self._config = config
        self._fl_rounds = config["fl_rounds"]
        self._federation = config["federation"]
        self._num_clients = config["num_clients"]
        self._num_workers = config["num_workers"]
        self._client_fn = config["client_fn"]
        self._server_fn = config["server_fn"]
        self._run_id = config["run_id"]
        self._server_node_id = 0

    def run_loop(self):
        click.echo(click.style("Starting Simulation Engine", bold=True, underline=True, fg="green"))
        # Create server context
        server_node_config = {}
        server_state = RecordSet(parameters_records={}, metrics_records={}, configs_records={})
        server_run_config = {**self._config["pyproject"]["tool"]["flwr"]["app"]["config"]}
        server_run_config["num-server-rounds"] = self._fl_rounds

        server_context = Context(
            run_id=self._run_id,
            node_id=self._server_node_id,
            node_config=server_node_config,
            state=server_state,
            run_config=server_run_config,
        )
        # Create client state and config
        client_states = {
            i: RecordSet(parameters_records={}, metrics_records={}, configs_records={})
            for i in range(self._num_clients)
        }
        client_node_configs: Dict[int, UserConfig] = {
            i: {"partition-id": i, "num-partitions": self._num_clients} for i in range(self._num_clients)
        }
        client_configs = {i: copy.deepcopy(server_run_config) for i in range(self._num_clients)}
        client_contexts = {
            i: Context(self._run_id, i, client_node_configs[i], client_states[i], client_configs[i])
            for i in range(self._num_clients)
        }

        # Create server
        fds = load_fds(self._num_clients)
        server = self._server_fn(server_context, fds)
        client_mng = ClientManagerStub(self._num_clients)

        init_params = server.strategy.initial_parameters  # type: ignore
        strategy: Strategy = server.strategy
        global_params = init_params
        # FL loop
        for server_round in range(1, self._fl_rounds + 1):
            click.echo(click.style(f"Started Federated Learning Round: {server_round}", italic=True, fg="green"))
            configure_fit_res = self._configure_fit(strategy, server_round, global_params, client_mng)
            fit_res, fit_failures = self._fit_clients(configure_fit_res, client_contexts)
            agg_params, agg_metrics = self._aggregate_fit(strategy, server_round, fit_res, fit_failures)
            click.echo(f"Aggregated Fit Metrics: {agg_metrics}")
            params, eval_res = self._evaluate(strategy, server_round, agg_params, global_params)
            global_params = params
            configure_eval_res = self._configure_evaluate_clients(
                strategy, server_round, copy.deepcopy(global_params), client_mng
            )
            eval_res, eval_failures = self._evaluate_clients(configure_eval_res, client_contexts)
            agg_loss, agg_metrics = self._aggregate_evaluate(strategy, server_round, eval_res, eval_failures)
            click.echo(f"Aggregated Eval Loss: {agg_loss} | Metrics: {agg_metrics}")
            click.echo(click.style(f"Finished Federated Learning Round: {server_round}", italic=True, fg="green"))
            cleanup()
        self._finalize()

    def _configure_fit(self, strategy: Strategy, server_round: int, init_params: Parameters, client_mng: ClientManager):
        click.echo(click.style(" > Configure Fit", italic=True, fg="green"))
        return strategy.configure_fit(server_round, init_params, client_mng)

    def _fit_clients(self, configure_fit_res, client_contexts):
        click.echo(click.style(" > Fit Clients", italic=True, fg="green"))
        return self._call_clients(configure_fit_res, client_contexts, ClientMethods.FIT)

    def _evaluate_clients(self, configure_eval_res, client_contexts):
        click.echo(click.style(" > Evaluate Clients", italic=True, fg="green"))
        return self._call_clients(configure_eval_res, client_contexts, ClientMethods.EVALUATE)

    def _call_clients(self, configure_res, contexts, method: ClientMethods):
        with tqdm(total=self._num_clients, desc=f"Calling {method} on Clients", leave=True, unit="Client") as bar:
            sem = Semaphore(5)
            with ProcessPoolExecutor(
                max_workers=self._num_workers,
                max_tasks_per_child=MAX_TASK_PER_CHILD,
                initializer=initializer,
                initargs=[sem],
            ) as p:
                tasks = {}
                cons = {i: multiprocessing.Pipe(duplex=True) for i in range(self._num_clients)}
                # Check if parameters are going to be the same -> don't need to save multiple times in shared memory
                groups_ids = []  # grouped ids for same parameters e.g. [[1,2,3], [4,5,6]]
                groups_params = {}  # grouped parameters
                for k, g in itertools.groupby(configure_res, lambda cres: cres[1].parameters.tensors):
                    ids = [int(x[0].cid) for x in g]
                    groups_ids.append(ids)
                    groups_params[len(groups_ids) - 1] = k

                dealt_with = []  # contains index of groups_id
                results = []
                for c, ins in configure_res:
                    id = int(c.cid)

                    g_id = self._get_gid(id, groups_ids)

                    if g_id not in dealt_with:
                        # create new shared memory https://github.com/python/cpython/issues/106939
                        padded_params = [tensor + b"\x01" for tensor in ins.parameters.tensors]
                        shm_name = f"ins-params-client-{g_id}"
                        _shm = shared_memory.ShareableList(padded_params, name=shm_name)
                        groups_params[g_id] = _shm
                        dealt_with.append(g_id)
                    ins.parameters = Parameters(tensors=[], tensor_type="padded.numpy.ndarray")
                    tasks[p.submit(fire_and_retrieve, self._client_fn, contexts[id], cons[id][1], id)] = id
                    cons[id][0].send(method)
                    cons[id][0].send((groups_params[g_id].shm.name, ins))
                # iterate over connections to retrieve results
                results: list[tuple[ClientProxy, Union[FitRes, EvaluateRes]]] = []
                cons_parent = [cons[id][0] for id in range(self._num_clients)]
                dealt_with = []
                while len(dealt_with) < self._num_clients:
                    for con in connection.wait(cons_parent):
                        id, msg = con.recv()
                        match method:
                            case ClientMethods.FIT | ClientMethods.GET_PARAMETERS:
                                name = msg[0]
                                padded_params_shm = shared_memory.ShareableList(name=name)
                                p = Parameters(tensors=padded_params_shm, tensor_type="padded.numpy.ndarray")
                                # close shared memory instance
                                p.tensors = list(map(lambda e: e[:-1], p.tensors))
                                p.tensor_type = "numpy.ndarray"
                                padded_params_shm.shm.close()
                                msg[1].parameters = p
                                con.send("DELETE")
                                results.append((ClientProxyStub(id), msg[1]))
                            case ClientMethods.EVALUATE:
                                results.append((ClientProxyStub(id), msg))
                        # Get Context update
                        con.send("CONTEXT")
                        _id, res = con.recv()
                        assert _id == id
                        contexts[id] = res
                        # Finishing Protocol
                        con.send("DONE")
                        _id, ack = con.recv()
                        assert _id == id and ack == "DONE"
                        con.close()
                        cons_parent.remove(con)
                        dealt_with.append(id)
                        # remove shared memory objects if necessary
                        g_id = self._get_gid(id, groups_ids)
                        if set(dealt_with) >= set(groups_ids[g_id]):
                            if groups_params[g_id] is not None:
                                groups_params[g_id].shm.close()
                                groups_params[g_id].shm.unlink()
                                del groups_params[g_id]
                            else:
                                pass
                        else:
                            pass
                        bar.update(1)
                # Check Shared memory cleanup
                assert groups_params == {}
                # Check pipes removed
                assert cons_parent == []
                # Check all tasks returned True
                assert all(t.result() for t in tasks.keys())
                # delete connections and tasks
                del cons, tasks
                # del semaphore
                del sem
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

    def _evaluate(self, strategy: Strategy, server_round, params: Optional[Parameters], global_params: Parameters):
        click.echo(click.style(" > Evaluate", italic=True, fg="green"))
        if params is None:
            params = global_params
        return params, strategy.evaluate(server_round, params)

    def _configure_evaluate_clients(
        self, strategy: Strategy, server_round, params: Parameters, client_mng: ClientManager
    ):
        click.echo(click.style(" > Configure Evaluate", italic=True, fg="green"))
        return strategy.configure_evaluate(server_round, params, client_mng)

    def _finalize(self):
        click.echo(click.style("Shutdown Simulation Engine", bold=True, underline=True))
        exit(0)
