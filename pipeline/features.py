"""Feature engineering — gated on which roles the dataset actually has.

Each feature family declares the roles it needs. If those roles are absent, the
family is skipped and reported as skipped — never fabricated. This is the
mechanism behind the honest version of "any dataset": a bank export with no
product column simply gets no product features, and the UI can say so.

Three families are produced when possible:
  * customer features (RFM + derived) — needs customer_id, order_date, revenue
  * monthly time series — needs order_date, revenue
  * product aggregates — needs product, revenue
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.config import Role, settings
from detection.schema import SchemaMapping
from utils.helpers import safe_divide
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureSets:
    """The engineered feature tables plus a record of what was built/skipped."""

    customer: pd.DataFrame = field(default_factory=pd.DataFrame)
    monthly: pd.DataFrame = field(default_factory=pd.DataFrame)
    product: pd.DataFrame = field(default_factory=pd.DataFrame)
    built: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)  # family -> reason

    def has_customer(self) -> bool:
        return not self.customer.empty

    def has_monthly(self) -> bool:
        return not self.monthly.empty

    def has_product(self) -> bool:
        return not self.product.empty

    def summary(self) -> dict[str, Any]:
        return {
            "built": self.built,
            "skipped": self.skipped,
            "customer_rows": len(self.customer),
            "monthly_rows": len(self.monthly),
            "product_rows": len(self.product),
        }


def engineer(frame: pd.DataFrame, mapping: SchemaMapping) -> FeatureSets:
    """Build every feature family the roles support."""
    fs = FeatureSets()

    _build_customer_features(frame, mapping, fs)
    _build_monthly_series(frame, mapping, fs)
    _build_product_features(frame, mapping, fs)

    logger.info("Feature engineering: built=%s skipped=%s", fs.built, list(fs.skipped))
    return fs


def _need(mapping: SchemaMapping, *roles: str) -> str | None:
    """Return a human reason string if any role is missing, else None."""
    missing = [r for r in roles if not mapping.has_role(r)]
    if missing:
        return "missing " + ", ".join(missing)
    return None


def _build_customer_features(frame, mapping, fs: FeatureSets) -> None:
    reason = _need(mapping, Role.CUSTOMER_ID, Role.ORDER_DATE, Role.REVENUE)
    if reason:
        fs.skipped["customer"] = reason
        return

    cust = mapping.role_to_column(Role.CUSTOMER_ID)
    date = mapping.role_to_column(Role.ORDER_DATE)
    rev = mapping.role_to_column(Role.REVENUE)
    order = mapping.role_to_column(Role.ORDER_ID)

    df = frame[[cust, date, rev] + ([order] if order else [])].copy()
    df[date] = pd.to_datetime(df[date], errors="coerce", format="mixed")
    df = df.dropna(subset=[cust, date])
    if df.empty:
        fs.skipped["customer"] = "no valid rows after date parsing"
        return

    ref_date = df[date].max()
    order_key = order if order else None

    grouped = df.groupby(cust)
    features = pd.DataFrame({
        "recency_days": (ref_date - grouped[date].max()).dt.days,
        "frequency": grouped[order_key].nunique() if order_key else grouped.size(),
        "monetary": grouped[rev].sum(),
        "avg_order_value": grouped[rev].mean(),
        "first_purchase": grouped[date].min(),
        "last_purchase": grouped[date].max(),
    })
    features["tenure_days"] = (ref_date - features["first_purchase"]).dt.days
    features["active_span_days"] = (features["last_purchase"] - features["first_purchase"]).dt.days
    features["orders_per_year"] = features.apply(
        lambda r: safe_divide(r["frequency"], max(r["tenure_days"], 1) / 365.25), axis=1
    )
    features["avg_interpurchase_days"] = features.apply(
        lambda r: safe_divide(r["active_span_days"], max(r["frequency"] - 1, 1)), axis=1
    )
    features = features.reset_index().rename(columns={cust: "customer_id"})
    # drop datetime helpers not needed downstream (keep numeric matrix clean)
    features = features.drop(columns=["first_purchase", "last_purchase"])

    fs.customer = features
    fs.built.append("customer")
    logger.info("Built customer features: %d customers", len(features))


def _build_monthly_series(frame, mapping, fs: FeatureSets) -> None:
    reason = _need(mapping, Role.ORDER_DATE, Role.REVENUE)
    if reason:
        fs.skipped["monthly"] = reason
        return

    date = mapping.role_to_column(Role.ORDER_DATE)
    rev = mapping.role_to_column(Role.REVENUE)
    cust = mapping.role_to_column(Role.CUSTOMER_ID)
    order = mapping.role_to_column(Role.ORDER_ID)

    df = frame[[date, rev]].copy()
    df[date] = pd.to_datetime(df[date], errors="coerce", format="mixed")
    df = df.dropna(subset=[date])
    if df.empty:
        fs.skipped["monthly"] = "no valid dates"
        return

    df["period"] = df[date].dt.to_period("M")
    agg = df.groupby("period").agg(sales=(rev, "sum"), rows=(rev, "size"))

    # add distinct customers/orders per month if available
    if cust and cust in frame.columns:
        tmp = frame[[date, cust]].copy()
        tmp[date] = pd.to_datetime(tmp[date], errors="coerce", format="mixed")
        tmp = tmp.dropna(subset=[date])
        tmp["period"] = tmp[date].dt.to_period("M")
        agg["customers"] = tmp.groupby("period")[cust].nunique()

    agg = agg.reset_index()
    agg["month"] = agg["period"].astype(str)
    agg = agg.sort_values("period").reset_index(drop=True)

    # time-series features for forecasting
    agg["month_num"] = agg["period"].dt.month
    agg["year"] = agg["period"].dt.year
    agg["time_index"] = np.arange(len(agg))
    agg["month_sin"] = np.sin(2 * np.pi * agg["month_num"] / 12)
    agg["month_cos"] = np.cos(2 * np.pi * agg["month_num"] / 12)
    for lag in (1, 2, 3):
        agg[f"lag_{lag}"] = agg["sales"].shift(lag)
    agg["roll_mean_3"] = agg["sales"].rolling(3, min_periods=1).mean()
    agg["growth_rate"] = agg["sales"].pct_change() * 100

    if "customers" not in agg.columns:
        agg["customers"] = np.nan

    fs.monthly = agg
    fs.built.append("monthly")
    logger.info("Built monthly series: %d months", len(agg))


def _build_product_features(frame, mapping, fs: FeatureSets) -> None:
    reason = _need(mapping, Role.PRODUCT, Role.REVENUE)
    if reason:
        fs.skipped["product"] = reason
        return

    prod = mapping.role_to_column(Role.PRODUCT)
    rev = mapping.role_to_column(Role.REVENUE)
    qty = mapping.role_to_column(Role.QUANTITY)
    profit = mapping.role_to_column(Role.PROFIT)

    agg_spec: dict[str, Any] = {"revenue": (rev, "sum"), "orders": (rev, "size")}
    if qty and qty in frame.columns:
        agg_spec["units"] = (qty, "sum")
    if profit and profit in frame.columns:
        agg_spec["profit"] = (profit, "sum")

    features = frame.groupby(prod).agg(**agg_spec)
    features = features.sort_values("revenue", ascending=False).reset_index()
    features = features.rename(columns={prod: "product"})

    fs.product = features
    fs.built.append("product")
    logger.info("Built product features: %d products", len(features))


__all__ = ["engineer", "FeatureSets"]
