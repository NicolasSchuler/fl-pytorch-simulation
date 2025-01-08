from logging import ERROR
from multiprocessing import Process, shared_memory
from multiprocessing.connection import Connection

from click import Context
from flwr.client import Client
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    log,
)
from pytorchexample.common import ClientMethods, cleanup

sem = None  # Global Semaphore object


class ClientSharedMemory(Client, Process):
    def __init__(self, client_fn, context: Context, con: Connection, id: int):
        super(ClientSharedMemory, self).__init__()
        self.con = con
        self.client_fn = client_fn
        self.context = context
        self.id = id
        self.client = None

    def fit(self, ins: FitIns) -> FitRes:
        return self.client.fit(ins)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return self.client.evaluate(ins)

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return self.client.get_parameters(ins)

    def run(self):
        self.client = self.client_fn(self.context)
        while True:
            match self.con.recv():
                case ClientMethods.FIT:
                    res = self.fit(self._read_shared_mem())
                    _shm = self._send_shared_mem(res)
                    msg = self.con.recv()
                    if msg != "DELETE":
                        raise ValueError(
                            f"DELETE MSG IS EXPECTED HERE, INSTEAD GOT {msg}"
                        )
                    _shm.shm.close()
                    _shm.shm.unlink()
                case ClientMethods.EVALUATE:
                    res = self.evaluate(self._read_shared_mem())
                    self.con.send((self.id, res))
                case ClientMethods.GET_PARAMETERS:
                    res = self.get_parameters(self._read_shared_mem())
                    _shm = self._send_shared_mem(res)
                    msg = self.con.recv()
                    if msg != "DELETE":
                        raise ValueError(
                            f"DELETE MSG IS EXPECTED HERE, INSTEAD GOT {msg}"
                        )
                    _shm.shm.close()
                    _shm.shm.unlink()
                case "CONTEXT":
                    # Remove Semaphore if available, before sharing context again
                    self.client.context.node_config.pop("semaphore", None)
                    self.con.send((self.id, (self.client.context)))
                case "DONE":
                    self.con.send((self.id, ("DONE")))
                    self.con.close()
                    break

    def _read_shared_mem(self):
        params_shm_name, ins = self.con.recv()
        from multiprocessing import shared_memory

        params_shm = shared_memory.ShareableList(name=params_shm_name)
        ins.parameters = Parameters(
            tensors=params_shm, tensor_type="padded.numpy.ndarray"
        )

        ins.parameters.tensors = list(map(lambda e: e[:-1], ins.parameters.tensors))
        ins.parameters.tensor_type = "numpy.ndarray"
        params_shm.shm.close()
        return ins

    def _send_shared_mem(self, res):
        padded_params = [tensor + b"\x01" for tensor in res.parameters.tensors]
        res.parameters.tensors = []
        res_params_shm_name = f"result-client-{self.id}"
        res_params_shm = shared_memory.ShareableList(
            padded_params, name=res_params_shm_name
        )
        self.con.send((self.id, (res_params_shm_name, res)))
        return res_params_shm


def fire_and_retrieve(client_fn, context, con, id):
    try:
        # Get global semaphore
        global sem
        # Add to context
        context.node_config["semaphore"] = sem
        client = ClientSharedMemory(client_fn, context, con, id)
        client.run()
        del client
        cleanup()
        return True
    except Exception as e:
        log(ERROR, f"Something went wrong in Child Process: {e}")
        return False


def initializer(semaphore):
    """Initializes the global Semaphore"""
    global sem

    sem = semaphore
