from abc import abstractmethod
from torch import Tensor


class TrainerBase:
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        NotImplementedError()

    @abstractmethod
    def save_model(self, **kwargs):
        NotImplementedError()

    @abstractmethod
    def train_one_epoch(self, **kwargs):
        NotImplementedError()

    @abstractmethod
    def fit(self, **kwargs):
        NotImplementedError()
