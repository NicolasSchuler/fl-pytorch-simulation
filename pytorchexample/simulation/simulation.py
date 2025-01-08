from abc import ABC, abstractmethod


class Simulation(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def run_loop(self):
        raise NotImplementedError
