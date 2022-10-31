from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Iterable
from tabulate import tabulate
import numpy as np


class Clusterer(ABC):
    def __init__(self, **kwargs):
        self._embedding = None
        self._params = dict(**kwargs)
        self.update_params(**kwargs)

    @property
    def embedding(self) -> Iterable:
        return self._embedding

    @property
    @abstractmethod
    def labels(self) -> Iterable:
        pass

    @property
    @abstractmethod
    def cluster_docs(self) -> Dict[str, Iterable]:
        pass

    @property
    @abstractmethod
    def coverage(self) -> float:
        pass

    @property
    @abstractmethod
    def n_clusters(self) -> int:
        pass

    @property
    @abstractmethod
    def probabilities(self) -> Iterable:
        pass

    @property
    @abstractmethod
    def tree(self):
        pass

    @abstractmethod
    def fit(self, X: Iterable, **kwargs) -> Clusterer:
        pass

    @abstractmethod
    def transform(self, X: Iterable, **kwargs) -> Iterable:
        pass

    @abstractmethod
    def predict(self, X: Iterable, **kwargs) -> Iterable:
        pass

    def fit_transform(self, X: Iterable, **kwargs) -> Iterable:
        self.fit(X, **kwargs)
        return self.transform(X, **kwargs)

    def fit_predict(self, X: Iterable, **kwargs) -> Iterable:
        self.fit(X, **kwargs)
        return self.predict(X, **kwargs)

    def update_params(self, **kwargs):
        self._params.update(kwargs)
        for key, value in self._params.items():
            if hasattr(self._model, key):
                setattr(self._model, key, value)

    def __len__(self) -> int:
        return len(self._embedding) if (
            isinstance(
                self.embedding, Iterable
            ) and np.size(self.embedding) > 0
        ) else 0

    def __repr__(self) -> str:
        return tabulate([
            ["Coverage", f"{self.coverage:.2f}%"],
            ["Number of clusters", self.n_clusters]])
