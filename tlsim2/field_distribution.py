from __future__ import annotations
from abc import abstractmethod
from typing import Mapping, Union


class FieldDistribution:
    @abstractmethod
    def eval_terminals(self) -> Mapping[str, complex]:
        pass

    @abstractmethod
    def conjugate(self) -> FieldDistribution:
        pass

    @abstractmethod
    def __matmul__(self, other: FieldDistribution):
        pass

    @abstractmethod
    def __add__(self, other: FieldDistribution):
        pass

    @abstractmethod
    def __sub__(self, other: FieldDistribution):
        pass

    @abstractmethod
    def __mul__(self, other: Union[float, complex, FieldDistribution]):
        pass

    @abstractmethod
    def __rmul__(self, other: Union[float, complex, FieldDistribution]):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __len__(self):
        pass
