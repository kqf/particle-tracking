import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline


class PrepareInput(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, hits):
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values

        r = np.sqrt(x**2 + y**2 + z**2)
        hits["x2"] = x / r
        hits["y2"] = y / r

        r = np.sqrt(x**2 + y**2)
        hits["z2"] = z / r
        return hits[["x2", "y2", "z2"]].values


def dbscan_model(eps):
    clustering = DBSCAN(eps=eps,
                        metric="minkowski",
                        metric_params={"p": 1.},
                        min_samples=1, algorithm="kd_tree")

    model = make_pipeline(
        PrepareInput(),
        StandardScaler(),
        clustering
    )
    return model
