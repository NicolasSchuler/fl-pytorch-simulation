# Implementation for two (partial) Simulation Engine Backends

Clone the repository, create a virtualenv and install the packages:

```bash
# Create project directory structure
mkdir project && cd project && git clone https://github.com/NicolasSchuler/fl-pytorch-simulation.git

# create virtualenv
python -m virtualenv .venv && source .venv/bin/activate

# install dependencies
pip install "flwr" "flwr-datasets[vision]" ray psutil torch torchvision

# go into directory
cd fl-pytorch-simulation
```

Help command:

```bash
❯ python pytorchexample/run_simulation.py --help
Usage: run_simulation.py [OPTIONS] CLIENT_FN SERVER_FN [FEDERATION]

  This script runs a federated learning simulation

Options:
  -b, --backend [ray-task|process-pool]
                                  Select which simulation backend to be used
                                  [required]
  -n, --num-clients INTEGER       Number of clients participating in the
                                  federation
  -r, --fl-rounds INTEGER         Number of federated learning rounds
  --num-gpus-total INTEGER        Number of GPUs to be used for the *whole*
                                  simulation. If not specified all available
                                  are used
  --num-cpus-total INTEGER        Number of CPUs to be used for the *whole*
                                  simulation. If not specified all available
                                  are used
  --num-workers INTEGER           Number of total workers used during the
                                  *whole* simulation. If not explicitly given,
                                  it will be calculated from the settings in
                                  the pyproject.toml or if not possible
                                  default to 1
  --num-threads INTEGER           Set number of threads for torch,
                                  OMP_NUM_THREADS, MKL_NUM_THREADS,
                                  OPENBLAS_NUM_THREADS
  --num-cpus-worker INTEGER       Number of CPUs each worker can utilize
  --num-gpus-worker FLOAT         Number of GPUs each worker can utilize
  --max-memory-total INTEGER      Total maximum memory to be utilized
  --max-memory-worker FLOAT       Specify maximum memory a worker uses (in GB)
  --overwrite-app-config TEXT     JSON formatted string that overwrites values
                                  in pyproject.toml [tool.flwr.app.config]
                                  section
  --overwrite-federation-config TEXT
                                  JSON formatted string that overwrites values
                                  in pyproject.toml [tool.flwr.federations.*]
                                  section
  --ray-address TEXT              Address for ray cluster
  --ray-dashboard                 Activate Ray Dashboard
  --ray-dashboard-port INTEGER    Ray Dashboard port
  --ray-dashboard-host TEXT       Ray Dashboard host
  --ray-namespace TEXT            Namespace for Ray Cluster that is going to
                                  be used
  --run-id TEXT                   ID for the current run. Defaults to
                                  generated UUID4
  --print-config BOOLEAN          Prints the parsed configuration at the
                                  beginning of the simulation
  --help                          Show this message and exit.

```

Run with standard concurrent.futures.ProcessPool:

```bash
python pytorchexample/run_simulation.py pytorchexample.fn pytorchexample.fn -b ray-task --num-workers 5
```

Run with Ray as an Task:

```bash
python pytorchexample/run_simulation.py pytorchexample.fn pytorchexample.fn -b ray-task --num-gpus-total 1 --num-gpus-worker 0.5 --num-cpus-worker 2
```