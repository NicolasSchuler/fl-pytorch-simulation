import importlib
import os
import pprint
import signal
import sys
import uuid
from multiprocessing import active_children, cpu_count, set_start_method
from pathlib import Path

import click
import psutil
import torch
from click.core import ParameterSource
from pytorchexample.simulation.process_pool import ProcessPoolSimulator
from pytorchexample.simulation.ray_task import RayTaskSimulator
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@click.command()
@click.option(
    "-b",
    "--backend",
    required=True,
    type=click.Choice(["ray-task", "process-pool"]),
    help="Select which simulation backend to be used",
)
@click.option("-n", "--num-clients", type=int, help="Number of clients participating in the federation")
@click.option("-r", "--fl-rounds", type=int, help="Number of federated learning rounds")
@click.option(
    "--num-gpus-total",
    type=int,
    help="Number of GPUs to be used for the *whole* simulation. If not specified all available are used",
)
@click.option(
    "--num-cpus-total",
    type=int,
    help="Number of CPUs to be used for the *whole* simulation. If not specified all available are used",
)
@click.option(
    "--num-workers",
    type=int,
    help="Number of total workers used during the *whole* simulation. If not explicitly given, it will be calculated from the settings in the pyproject.toml or if not possible default to 1",
)
@click.option(
    "--num-threads",
    type=int,
    default=cpu_count(),
    help="Set number of threads for torch, OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS",
)
@click.option("--num-cpus-worker", type=int, help="Number of CPUs each worker can utilize")
@click.option("--num-gpus-worker", type=float, help="Number of GPUs each worker can utilize")
@click.option("--max-memory-total", type=int, help="Total maximum memory to be utilized")
@click.option("--max-memory-worker", type=float, default=4, help="Specify maximum memory a worker uses (in GB)")
@click.option(
    "--overwrite-app-config",
    type=str,
    help="JSON formatted string that overwrites values in pyproject.toml [tool.flwr.app.config] section",
)
@click.option(
    "--overwrite-federation-config",
    type=str,
    help="JSON formatted string that overwrites values in pyproject.toml [tool.flwr.federations.*] section",
)
@click.option("--ray-address", type=str, default="local", help="Address for ray cluster")
@click.option("--ray-dashboard", is_flag=True, type=bool, default=False, help="Activate Ray Dashboard")
@click.option("--ray-dashboard-port", type=int, default=8265, help="Ray Dashboard port")
@click.option("--ray-dashboard-host", type=str, default="127.0.0.1", help="Ray Dashboard host")
@click.option(
    "--ray-namespace", type=str, default="flwr-simulation", help="Namespace for Ray Cluster that is going to be used"
)
@click.option("--run-id", type=str, help="ID for the current run. Defaults to generated UUID4")
@click.option(
    "--print-config",
    type=bool,
    default=False,
    help="Prints the parsed configuration at the beginning of the simulation",
)
@click.argument("client_fn", nargs=1, type=str)
@click.argument("server_fn", nargs=1, type=str)
@click.argument("federation", nargs=1, type=str, required=False)
def simulation(**kwargs):
    """This script runs a federated learning simulation"""
    click.echo("Reading configuration")
    # Set multiprocessing start method
    set_start_method("spawn")
    # Set number of MP Threads
    os.environ["OMP_NUM_THREADS"] = str(kwargs["num_threads"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(kwargs["num_threads"])
    os.environ["MKL_NUM_THREADS"] = str(kwargs["num_threads"])
    torch.set_num_threads(kwargs["num_threads"])

    pyproject_path = Path(os.path.dirname(__file__)) / "../pyproject.toml"
    if not pyproject_path.exists():
        click.echo(f"pyproject.toml not found: {pyproject_path}", err=True)
        exit(-1)
    pyproject = None
    try:
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f, parse_float=float)
    except tomllib.TOMLDecodeError:
        click.echo("Could not parse the toml file. Check if the toml file is valid!", err=True)
        exit(-1)

    config = validate_config(kwargs, pyproject)
    if config["kwargs"]["print_config"]:
        click.echo(click.style("Configuration to be used: ", fg="green"))
        click.echo(pprint.pformat(config))

    backend = kwargs.pop("backend")
    if backend == "ray-task":
        click.echo("Ray-Task Simulation Engine has been chosen")
        sim = RayTaskSimulator(config)
        sim.run_loop()
    elif backend == "process-pool":
        if click.get_current_context().get_parameter_source("ray_address") == ParameterSource.COMMANDLINE:
            click.echo("--ray-address is not supported in ProcessPool Engine -> ignoring", err=True)
        if click.get_current_context().get_parameter_source("ray_dashboard") == ParameterSource.COMMANDLINE:
            click.echo("--ray-dashboard is not supported in ProcessPool Engine -> ignoring", err=True)
        if click.get_current_context().get_parameter_source("ray_dashboard_port") == ParameterSource.COMMANDLINE:
            click.echo("--ray-dashboard-port is not supported in ProcessPool Engine -> ignoring", err=True)
        if click.get_current_context().get_parameter_source("ray_dashboard_host") == ParameterSource.COMMANDLINE:
            click.echo("--ray-dashboard-host is not supported in ProcessPool Engine -> ignoring", err=True)

        if backend == "process-pool":
            click.echo("ProcessPool Simulation Engine has been chosen")
            sim = ProcessPoolSimulator(config)
        sim.run_loop()
    click.echo("Simulation Finished!")


def validate_config(kwargs, pyproject):
    config = {}

    try:
        mod_client_fn = importlib.import_module(kwargs["client_fn"])
        mod_server_fn = importlib.import_module(kwargs["server_fn"])
        config["client_fn"] = mod_client_fn.client_fn
        config["server_fn"] = mod_server_fn.server_fn

    except ModuleNotFoundError as e:
        click.echo("client_fn or server_fn could not be loaded. Check your paths", err=True)
        click.echo(e, err=True)
        exit(-1)

    if kwargs["overwrite_app_config"]:
        try:
            pyproject["tool"]["flwr"]["app"]["config"].update(kwargs.pop("overwrite_app_config"))
        except Exception:
            click.echo("Something went wrong while overwriting the app config!", err=True)
            exit(-1)

    if kwargs["fl_rounds"]:
        config["fl_rounds"] = kwargs.pop("fl_rounds")
    else:
        try:
            config["fl_rounds"] = pyproject["tool"]["flwr"]["app"]["config"]["num-server-rounds"]
        except KeyError:
            click.echo(
                "Specify number of federated learning rounds either in [tool.flwr.app.config] or via '--fl-rounds'!",
                err=True,
            )
            exit(-1)

    if kwargs["federation"]:
        config["federation"] = kwargs.pop("federation")
    else:
        try:
            config["federation"] = pyproject["tool"]["flwr"]["federations"]["default"]
        except KeyError:
            click.echo(
                "Specify the name of the federation either in [tool.flwr.federations] via 'default' or via the positional federation parameter of the cli!",
                err=True,
            )
            exit(-1)
    if kwargs["overwrite_federation_config"]:
        try:
            pyproject["tool"]["flwr"]["federations"][config["federation"]].update(
                kwargs.pop("overwrite_federation_config")
            )
        except Exception:
            click.echo("Something went wrong while overwriting the federation config", err=True)
            exit(-1)

    if kwargs["num_clients"]:
        config["num_clients"] = kwargs.pop("num_clients")
    else:
        try:
            config["num_clients"] = pyproject["tool"]["flwr"]["federations"][config["federation"]]["options"][
                "num-supernodes"
            ]
        except KeyError:
            click.echo(
                "Specify number of clients participating in the federation either in [tool.flwr.federations.<federation>] or via '--num-clients'!",
                err=True,
            )
            exit(-1)
    if kwargs["num_cpus_worker"]:
        config["num_cpus_worker"] = kwargs.pop("num_cpus_worker")

        if kwargs["backend"] == "proccess-pool":
            click.echo("--num-cpus-worker option not supported when using ProcessPool Engine -> ignoring", err=True)
    else:
        try:
            config["num_cpus_worker"] = pyproject["tool"]["flwr"]["federations"][config["federation"]]["options"][
                "backend"
            ]["client-resources"]["num-cpus"]
        except Exception:
            if kwargs["backend"] not in ["process-pool"]:
                click.echo("Could not determine number of cpus core each worker gets, will default to 1", err=True)
                config["num_cpus_worker"] = 1

    if kwargs["num_gpus_worker"]:
        config["num_gpus_worker"] = kwargs.pop("num_gpus_worker")

        if kwargs["backend"] in ["process-pool", "process-pool-shm"]:
            click.echo("--num-gpus-worker option not supported when using ProcessPool Engine -> ignoring", err=True)
    else:
        try:
            config["num_gpus_worker"] = pyproject["tool"]["flwr"]["federation"][config["federation"]]["options"][
                "backend"
            ]["client-resources"]["num-gpus"]
        except Exception:
            if kwargs["backend"] not in ["process-pool"]:
                click.echo("Could not determine number of gpus each worker gets, will default to 0", err=True)
                config["num_gpus_worker"] = 0

    if kwargs["num_cpus_total"]:
        config["num_cpus_total"] = kwargs.pop("num_cpus_total")

        if kwargs["backend"] in ["process-pool"]:
            click.echo("--num-cpus-total option not supported when using ProcessPool Engine -> ignoring", err=True)
    else:
        cpus = cpu_count()
        config["num_cpus_total"] = cpus
        if kwargs["backend"] not in ["process-pool"]:
            click.echo(f"Total number of cpus not provided assuming all core are used ({cpus})")

    if kwargs["num_gpus_total"]:
        config["num_gpus_total"] = kwargs.pop("num_gpus_total")

        if kwargs["backend"] in ["process-pool"]:
            click.echo("--num-gpus-total option not supported when using ProcessPool Engine -> ignoring", err=True)
    else:
        if kwargs["backend"] not in ["process-pool"]:
            click.echo("Total number of gpus not provided assuming none are used")
            config["num_gpus_total"] = 0

    if kwargs["max_memory_total"]:
        config["max_memory_total"] = kwargs.pop("max_memory_total")

        if kwargs["backend"] in ["process-pool"]:
            click.echo("--max-memory-total option not supported when using ProcessPool Engine -> ignoring", err=True)
    else:
        if kwargs["backend"] not in ["process-pool"]:
            click.echo("Total maximum memory not provided, assuming 98% of total available memory")
            config["max_memory_total"] = int(psutil.virtual_memory().total * 0.98)

    if kwargs["num_workers"]:
        config["num_workers"] = kwargs.pop("num_workers")
    else:
        if kwargs["backend"] in ["process-pool"]:
            click.echo("--num-workers option is required for process-pool backend", err=True)
            exit(-1)
        num_workers_cpu = config["num_cpus_total"] // config["num_cpus_worker"]
        if config["num_gpus_worker"] == 0 or config["num_gpus_total"] == 0:
            num_workers_gpu = 0
        else:
            num_workers_gpu = config["num_gpus_total"] // config["num_gpus_worker"]

        num_workers = max(1, min(num_workers_cpu, num_workers_gpu))
        config["num_workers"] = int(num_workers)
    num_workers = config["num_workers"]
    click.echo(f"According to the configuration, will use {num_workers} number of workers")
    if not kwargs["run_id"]:
        config["run_id"] = uuid.uuid4()

    config["kwargs"] = kwargs
    config["pyproject"] = pyproject
    return config


if __name__ == "__main__":

    def signal_handler(sig, frame):
        click.echo(click.style("Killing Application", bold=True, fg="red"))
        active = active_children()
        for a in active:
            a.kill()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    simulation()
