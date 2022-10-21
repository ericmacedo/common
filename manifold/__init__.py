from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np


class Manifold(ABC):
    def __init__(self, n_components: int = 2, **kwargs):
        self._embedding = None
        self._params = dict(n_components=n_components, **kwargs)
        self.update_params(**kwargs)

    @property
    def embedding(self) -> Iterable:
        return self._embedding

    def __getitem__(self, indexer) -> Any:
        return self._embedding[indexer] if (
            isinstance(
                self.embedding, Iterable
            ) and np.size(self.embedding) > 0
        ) else None

    @abstractmethod
    def fit(self, X: Iterable, **kwargs) -> Manifold:
        pass

    @abstractmethod
    def transform(self, X: Iterable, **kwargs) -> Iterable:
        pass

    def fit_transform(self, X: Iterable, **kwargs) -> Iterable:
        X = [*X]
        self.fit(X, **kwargs)
        return self.transform(X, **kwargs)

    def update_params(self, **kwargs):
        self._params.update(kwargs)
        for key, value in self._params.items():
            if hasattr(self._model, key):
                setattr(self._model, key, value)

    def __len__(self) -> int:
        return len(self._embedding) if self._embedding else 0
