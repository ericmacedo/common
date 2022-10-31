# go.Figure(go.Treemap(
#     labels=[4083, *clusters.child],
#     parents=["", *clusters.parent],
#     maxdepth=3,
#     root_color="lightgrey",
#     text=["Lorem ipsumm"] * (len(clusters) + 1),
#     hovertemplate="<b>Cluster %{label}</b><br><br>%{text}<extra></extra>"
# ), layout=dict(margin=dict(t=20, b=0, l=0, r=0)))

from typing import Dict, Iterable

import numpy as np

import hdbscan

from ..clustering import Clusterer

from ..manifold.umap import UMAP


class HDBSCAN(Clusterer):
    def __init__(self, **kwargs):
        params = dict(
            min_cluster_size=50,
            core_dist_n_jobs=-1,
            metric="l2",
            gen_min_span_tree=True,
            cluster_selection_epsilon=0.5,
            prediction_data=True)
        params.update(kwargs)

        self._model = hdbscan.HDBSCAN(**params)

        super(HDBSCAN, self).__init__(**params)

    @property
    def labels(self) -> Iterable:
        return self._model.labels_

    @property
    def cluster_docs(self) -> Dict[str, Iterable]:
        graph_df = self._model.condensed_tree_.to_pandas()
        doc_clusters = {}
        for i, cluster in enumerate(graph_df[graph_df.child_size > 1].child):
            doc_clusters[f"cluster_{i}"] = graph_df[
                (graph_df.child_size == 1) & (graph_df.parent == cluster)
            ].child.to_list()

        return doc_clusters

    @property
    def coverage(self) -> float:
        return np.sum(self.labels >= 0) / len(self)

    @property
    def n_clusters(self) -> int:
        return np.unique(self.labels).shape[0] - 1  # -1 means unclustered

    @property
    def probabilities(self) -> Iterable:
        return self._model.probabilities

    @property
    def tree(self):
        return self._model.condensed_tree_.to_networkx()

    def fit(self, X: Iterable, **kwargs) -> Clusterer:
        X = np.array(X)
        self._embedding = UMAP(
            n_components=int(X.shape[1]**0.5)
        ).fit_transform([*X])

        self._model.fit(self._embedding, **kwargs)

        return self

    def transform(self, X: Iterable, **kwargs) -> Iterable:
        return hdbscan.prediction.all_points_membership_vectors(self._model)

    def predict(self, X: Iterable, **kwargs) -> Iterable:
        return self._model.fit_predict(X, **kwargs)
