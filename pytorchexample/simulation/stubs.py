import random
from logging import INFO
from typing import Optional

from flwr.common import (
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    ReconnectIns,
    log,
)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class ClientManagerStub(ClientManager):
    """Code taken from the Flower repository (https://github.com/adap/flower)"""
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.clients = {f"{i}": ClientProxyStub(str(i)) for i in range(num_clients)}

    def num_available(self) -> int:
        return self.num_clients

    def register(self, client: ClientProxy) -> bool:
        raise NotImplementedError

    def unregister(self, client: ClientProxy) -> None:
        raise NotImplementedError

    def wait_for(self, num_clients: int, timeout: int) -> bool:
        raise NotImplementedError

    def all(self) -> dict[str, ClientProxy]:
        return self.clients  # type: ignore

    def sample(
        self, num_clients: int, min_num_clients: Optional[int] = None, criterion: Optional[Criterion] = None
    ) -> list[ClientProxy]:
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [cid for cid in available_cids if criterion.select(self.clients[cid])]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients" " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]


class ClientProxyStub(ClientProxy):
    def get_properties(
        self, ins: GetPropertiesIns, timeout: Optional[float], group_id: Optional[int]
    ) -> GetPropertiesRes:
        return None  # type: ignore

    def get_parameters(
        self, ins: GetParametersIns, timeout: Optional[float], group_id: Optional[int]
    ) -> GetParametersRes:
        return None  # type: ignore

    def fit(self, ins: FitIns, timeout: Optional[float], group_id: Optional[int]) -> FitRes:
        return None  # type: ignore

    def evaluate(self, ins: EvaluateIns, timeout: Optional[float], group_id: Optional[int]) -> EvaluateRes:
        return None  # type: ignore

    def reconnect(self, ins: ReconnectIns, timeout: Optional[float], group_id: Optional[int]) -> DisconnectRes:
        return None  # type: ignore
