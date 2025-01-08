from enum import StrEnum
import torch
import gc

class ClientMethods(StrEnum):
    FIT = "fit"
    EVALUATE = "evaluate"
    GET_PARAMETERS = "get_parameters"

def cleanup():
    torch.cuda.empty_cache()
    gc.collect()