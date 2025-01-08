from typing import Optional

import ray
from click import Context
from flwr.client import Client
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
)
from flwr_datasets import FederatedDataset


@ray.remote
class ClientActor(Client):
    def __init__(
        self, client_fn, context: Context, fds: Optional[FederatedDataset] = None
    ):
        self.client = client_fn(context, fds=fds)

    def fit(self, ins: FitIns) -> FitRes:
        return self.client.fit(ins)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return self.client.evaluate(ins)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return self.client.get_parameters(ins)
