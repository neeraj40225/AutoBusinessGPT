"""Customer churn on engineered customer features.

No dataset ships a churn label, so we derive one behaviourally: a customer is
churned when their recency exceeds a multiple of their own typical purchase gap.
Then three classifiers compete on ROC-AUC. The threshold is tuned for F1, not
left at 0.5, because the classes are imbalanced and a missed churner costs more
than a false alarm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False

_FEATURES = ["recency_days", "frequency", "monetary", "avg_order_value",
             "tenure_days", "orders_per_year", "avg_interpurchase_days"]


@dataclass
class ChurnScore:
    name: str
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {"Model": self.name, "ROC-AUC": round(self.roc_auc, 3),
                "Accuracy": round(self.accuracy, 3), "Precision": round(self.precision, 3),
                "Recall": round(self.recall, 3), "F1": round(self.f1, 3)}


@dataclass
class ChurnResult:
    best_name: str
    scores: list[ChurnScore]
    churn_rate: float
    roc_points: dict[str, list[float]]
    confusion: list[list[int]]
    scored: pd.DataFrame          # customer_id, churn_probability, risk_band
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def best_score(self) -> ChurnScore:
        return next(s for s in self.scores if s.name == self.best_name)

    @property
    def scores_frame(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_dict() for s in self.scores])


def _label_churn(customer: pd.DataFrame) -> pd.Series:
    """Behavioural churn label: recency > multiplier × personal purchase gap."""
    mult = settings.ml.churn_inactivity_multiplier
    # personal gap; customers with 1 order use the population median gap
    pop_gap = customer.loc[customer["frequency"] > 1, "avg_interpurchase_days"].median()
    pop_gap = pop_gap if pop_gap and pop_gap > 0 else 90.0
    personal_gap = customer["avg_interpurchase_days"].where(
        customer["frequency"] > 1, pop_gap
    ).clip(lower=1)
    churned = (customer["recency_days"] > (mult * personal_gap)).astype(int)
    return churned


def train(customer: pd.DataFrame) -> ChurnResult:
    """Label, train, select best by ROC-AUC, and score all customers."""
    df = customer.copy()
    for col in _FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df[_FEATURES] = df[_FEATURES].fillna(0.0)

    y = _label_churn(df)
    churn_rate = float(y.mean())
    if y.nunique() < 2:
        raise ValueError("Churn label is single-class; cannot train.")

    x = df[_FEATURES]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_tr, x_te, y_tr, y_te = train_test_split(
        x_scaled, y, test_size=settings.ml.test_size,
        random_state=settings.ml.random_state, stratify=y,
    )

    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=settings.ml.random_state, n_jobs=-1),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=250, learning_rate=0.05, max_depth=3,
            scale_pos_weight=pos_weight, random_state=settings.ml.random_state,
            n_jobs=-1, verbosity=0, eval_metric="logloss")

    scores: list[ChurnScore] = []
    fitted: dict[str, Any] = {}
    roc_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, model in models.items():
        model.fit(x_tr, y_tr)
        prob = model.predict_proba(x_te)[:, 1]
        thr = _best_threshold(y_te.to_numpy(), prob)
        pred = (prob >= thr).astype(int)
        scores.append(ChurnScore(
            name=name,
            roc_auc=float(roc_auc_score(y_te, prob)),
            accuracy=float(accuracy_score(y_te, pred)),
            precision=float(precision_score(y_te, pred, zero_division=0)),
            recall=float(recall_score(y_te, pred, zero_division=0)),
            f1=float(f1_score(y_te, pred, zero_division=0)),
            threshold=thr,
        ))
        fitted[name] = model
        fpr, tpr, _ = roc_curve(y_te, prob)
        roc_cache[name] = (fpr, tpr)

    scores.sort(key=lambda s: s.roc_auc, reverse=True)
    best_name = scores[0].name
    best = fitted[best_name]
    best_thr = scores[0].threshold

    # confusion on test set at tuned threshold
    prob_te = best.predict_proba(x_te)[:, 1]
    pred_te = (prob_te >= best_thr).astype(int)
    cm = confusion_matrix(y_te, pred_te).tolist()
    fpr, tpr = roc_cache[best_name]

    # score every customer
    prob_all = best.predict_proba(x_scaled)[:, 1]
    scored = pd.DataFrame({
        "customer_id": df["customer_id"],
        "churn_probability": prob_all,
    })
    scored["risk_band"] = pd.cut(
        prob_all, bins=[-0.01, 0.3, 0.6, 0.8, 1.01],
        labels=["Low", "Medium", "High", "Critical"],
    ).astype(str)

    importance = _importance(best, best_name)
    logger.info("Churn best=%s AUC=%.3f rate=%.1f%%", best_name, scores[0].roc_auc, churn_rate * 100)
    return ChurnResult(
        best_name=best_name, scores=scores, churn_rate=churn_rate,
        roc_points={"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        confusion=cm, scored=scored, importance=importance,
    )


def _best_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Pick the probability threshold maximising F1."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.1, 0.9, 33):
        pred = (prob >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def _importance(model: Any, name: str) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()
    return (pd.DataFrame({"feature": _FEATURES, "importance": vals})
            .sort_values("importance", ascending=False).reset_index(drop=True))


__all__ = ["train", "ChurnResult", "ChurnScore"]
