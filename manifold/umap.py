from typing import Iterable

from ..manifold import Manifold

import umap


class UMAP(Manifold):
    def __init__(self, n_components: int = 2, **kwargs):
        params = dict(
            n_neighbors=50,
            min_dist=0.01,
            n_components=n_components,
            n_jobs=-1,
            metric="cosine")
        params.update(kwargs)

        self._model = umap.UMAP(**params)

        super(UMAP, self).__init__(**params)

    def fit(self, X: Iterable, **kwargs) -> Manifold:
        self.update_params(**kwargs)
        self._model.fit([*X])

        return self

    def transform(self, X: Iterable, **kwargs) -> Iterable:
        self.update_params(**kwargs)
        self._embedding = self._model.transform([*X])
        return self._embedding
