from typing import Iterable

from ..manifold import Manifold

from openTSNE import TSNE as OpenTSNE


class TSNE(Manifold):
    def __init__(self, n_components: int = 2, **kwargs):
        params = dict(
            n_components=n_components,
            perplexity=30,
            metric="cosine",
            n_jobs=-1)
        params.update(kwargs)

        self._model = OpenTSNE(**params)

        super(TSNE, self).__init__(**params)

    def fit(self, X: Iterable, **kwargs) -> Manifold:
        self.update_params(**kwargs)
        self._model.fit([*X])

        return self

    def transform(self, X: Iterable, **kwargs) -> Iterable:
        self.update_params(**kwargs)
        self._embedding = self._model.transform([*X])
        return self._embedding
