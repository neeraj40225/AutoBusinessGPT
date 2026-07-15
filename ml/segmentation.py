"""Customer segmentation via KMeans over RFM features.

RFM is right-skewed (a few whales dominate), so features are log-transformed
before scaling or KMeans degenerates into "whales vs everyone". k is chosen by
silhouette across a range, with a bias toward a k that yields the four named
business segments when its score is within tolerance of the peak. Clusters are
then ranked by value and mapped to VIP / Regular / Occasional / At Risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from core.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_FEATURES = ["recency_days", "frequency", "monetary", "avg_order_value", "orders_per_year"]
_LABELS = ("At Risk", "Occasional", "Regular", "VIP")


@dataclass
class SegmentationResult:
    best_k: int
    silhouette: float
    k_scores: pd.DataFrame          # k, inertia, silhouette
    profiles: pd.DataFrame          # per-segment averages
    assignments: pd.DataFrame       # customer_id, cluster, segment_label, pc1, pc2
    projection: pd.DataFrame = field(default_factory=pd.DataFrame)

    def label_counts(self) -> dict[str, int]:
        return self.assignments["segment_label"].value_counts().to_dict()


def train(customer: pd.DataFrame, forced_k: int | None = None) -> SegmentationResult:
    """Cluster customers and label segments by value."""
    df = customer.copy()
    for col in _FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    matrix = df[_FEATURES].fillna(0.0).copy()

    # log-transform skewed monetary features
    for col in ("monetary", "avg_order_value", "frequency"):
        matrix[col] = np.log1p(matrix[col].clip(lower=0))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    k_scores = _evaluate_k(scaled)
    best_k = forced_k or _select_k(k_scores, len(df))

    km = KMeans(n_clusters=best_k, random_state=settings.ml.random_state, n_init=10)
    clusters = km.fit_predict(scaled)
    sil = float(silhouette_score(scaled, clusters)) if best_k > 1 and len(df) > best_k else 0.0

    df["cluster"] = clusters
    label_map = _rank_and_label(df, best_k)
    df["segment_label"] = df["cluster"].map(label_map)

    # PCA projection for the scatter plot
    projection = _project(scaled, df)

    profiles = (df.groupby("segment_label")[_FEATURES]
                .mean().round(1).reset_index())
    profiles["count"] = df.groupby("segment_label").size().values

    assignments = df[["customer_id", "cluster", "segment_label"]].merge(
        projection[["customer_id", "pc1", "pc2"]], on="customer_id", how="left")

    logger.info("Segmentation k=%d silhouette=%.3f", best_k, sil)
    return SegmentationResult(
        best_k=best_k, silhouette=sil, k_scores=k_scores,
        profiles=profiles, assignments=assignments, projection=projection,
    )


def _evaluate_k(scaled: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(scaled)
    for k in settings.ml.segmentation_k_range:
        if k >= n:
            break
        km = KMeans(n_clusters=k, random_state=settings.ml.random_state, n_init=10)
        labels = km.fit_predict(scaled)
        sil = float(silhouette_score(scaled, labels)) if k > 1 else 0.0
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": sil})
    return pd.DataFrame(rows)


def _select_k(k_scores: pd.DataFrame, n_customers: int) -> int:
    """Prefer k=4 (four business segments) when within 20% of the peak silhouette."""
    if k_scores.empty:
        return min(4, max(2, n_customers - 1))
    peak = k_scores.loc[k_scores["silhouette"].idxmax()]
    four = k_scores[k_scores["k"] == 4]
    if not four.empty and peak["silhouette"] > 0:
        if four.iloc[0]["silhouette"] >= 0.8 * peak["silhouette"]:
            logger.info("k=4 within tolerance of peak k=%d; choosing 4", int(peak["k"]))
            return 4
    return int(peak["k"])


def _rank_and_label(df: pd.DataFrame, k: int) -> dict[int, str]:
    """Rank clusters by a value score and assign named labels."""
    agg = df.groupby("cluster").agg(
        recency=("recency_days", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
    )
    # value score: high frequency & monetary good, high recency bad
    agg["score"] = (
        agg["monetary"].rank() + agg["frequency"].rank() - agg["recency"].rank()
    )
    ordered = agg.sort_values("score").index.tolist()  # worst -> best

    labels = _LABELS if k == 4 else _spread_labels(k)
    return {cluster: labels[i] for i, cluster in enumerate(ordered)}


def _spread_labels(k: int) -> list[str]:
    """Produce k ordered labels worst->best for non-4 cluster counts."""
    if k <= len(_LABELS):
        return list(_LABELS[:k])
    extra = [f"Tier {i}" for i in range(k - len(_LABELS))]
    return [_LABELS[0], *extra, *_LABELS[1:]]


def _project(scaled: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    if scaled.shape[1] >= 2 and len(df) > 2:
        pca = PCA(n_components=2, random_state=settings.ml.random_state)
        coords = pca.fit_transform(scaled)
    else:
        coords = np.column_stack([scaled[:, 0], np.zeros(len(df))]) if scaled.size else np.zeros((len(df), 2))
    out = df[["customer_id", "segment_label"]].copy()
    out["pc1"] = coords[:, 0]
    out["pc2"] = coords[:, 1]
    return out


__all__ = ["train", "SegmentationResult"]
