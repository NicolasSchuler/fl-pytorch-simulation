"""pytorchexample: A Flower / PyTorch app."""

from typing import Optional, Callable
import torch
from flwr.client import Client, ClientApp
from flwr.common import (
    Context,
    FitIns,
    FitRes,
    ndarrays_to_parameters,
    Status,
    Code,
    EvaluateIns,
    EvaluateRes,
    Parameters,
    parameters_to_ndarrays,
    GetParametersIns,
    GetParametersRes,
)
import ray
from flwr_datasets import FederatedDataset
from pytorchexample.task import (
    load_fds,
    load_loaders,
    set_weights,
    train,
    test,
    get_model,
    get_weights,
)
from pytorchexample.common import ClientMethods


class FlowerClient(Client):
    def __init__(
        self,
        context: Context,
        device: torch.device,
        local_epochs,
        learning_rate,
        partition_id,
        num_partitions,
        batch_size,
        fds: Optional[FederatedDataset] = None,
        model_fn: Optional[Callable[[], torch.nn.Module]] = None,
        model: Optional[torch.nn.Module] = None,
    ):
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = device
        self.context = context
        self.batch_size = batch_size
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.client_state = self.context.state
        # Load fds
        if fds is None:
            self.fds = load_fds(self.num_partitions)
        else:
            self.fds = fds
        # Load Model
        if model_fn:
            self.net = model_fn()
        elif model:
            self.net = model
        else:
            raise ValueError("You need to specify a model to be used")
        self.trainloader, self.valloader = load_loaders(
            self.partition_id, self.batch_size, self.fds
        )
        self.net.to(self.device)

    def fit(self, ins: FitIns) -> FitRes:
        params = self._get_params(ins)
        set_weights(self.net, params)

        if "semaphore" in self.context.node_config:
            self.context.node_config["semaphore"].acquire()
            loss, acc = train(
                self.net,
                self.trainloader,
                self.valloader,
                self.local_epochs,
                self.lr,
                self.device,
            )
            self.context.node_config["semaphore"].release()
            self.context.node_config.pop("semaphore")
        else:
            loss, acc = train(
                self.net,
                self.trainloader,
                self.valloader,
                self.local_epochs,
                self.lr,
                self.device,
            )

        ndarrays_updated = get_weights(self.net)
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)

        status = Status(code=Code.OK, message="Success")

        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics={"val_loss": loss, "vall_acc": acc},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        params = self._get_params(ins)
        set_weights(self.net, params)

        loss, accuracy = test(self.net, self.valloader, self.device)
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=loss,
            num_examples=len(self.valloader),
            metrics={"accuracy": accuracy},
        )

    def _get_params(self, ins: EvaluateIns | FitIns):
        match ins.parameters.tensor_type:
            case "ray.store":
                params = Parameters(
                    tensors=ray.get(ins.parameters.tensors), tensor_type="nump.ndarray"
                )
                del ins.parameters.tensors
                return parameters_to_ndarrays(params)
            case "numpy.ndarray":
                ndarrays_original = parameters_to_ndarrays(ins.parameters)
                return ndarrays_original

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        # Get parameters as a list of NumPy ndarray's
        ndarrays = get_weights(self.net)

        # Serialize ndarray's into a Parameters object
        parameters = ndarrays_to_parameters(ndarrays)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )


def client_fn(
    context: Context,
    *args,
    method: Optional[ClientMethods] = None,
    fds: Optional[FederatedDataset] = None,
    **kwargs,
) -> tuple[FitRes | EvaluateRes | GetParametersRes, Context] | Client:
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    client = FlowerClient(
        context,
        device,
        local_epochs,
        learning_rate,
        partition_id,
        num_partitions,
        batch_size,
        fds=fds,
        model_fn=get_model,
    ).to_client()
    if method is None:
        return client
    else:
        res = None
        match method:
            case ClientMethods.FIT:
                res = client.fit(*args)
            case ClientMethods.EVALUATE:
                res = client.evaluate(*args)
            case ClientMethods.GET_PARAMETERS:
                res = client.get_parameters(*args)
            case _:
                raise NotImplementedError
        client = None
        return res, context


def client_fn_wrapper(context):
    return client_fn(context)


# Flower ClientApp
app = ClientApp(client_fn_wrapper)
