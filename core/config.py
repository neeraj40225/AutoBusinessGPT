"""Central configuration for AutoBusinessGPT.

Every path, tunable constant, semantic-role name, and env-backed secret lives
here. Import the singleton ``settings`` rather than reading os.environ elsewhere.

Design note: unlike a fixed-schema app, AutoBusinessGPT cannot hard-code column
names — the dataset is arbitrary. What it *can* fix is the vocabulary of
*semantic roles* every column is mapped to (customer_id, revenue, order_date,
...). That vocabulary is defined here and is the contract the whole pipeline
speaks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Semantic role vocabulary — the contract between detection and everything else
# --------------------------------------------------------------------------- #
class Role:
    """Canonical semantic roles a column can be mapped to.

    These strings are used as keys throughout the pipeline. A column that maps
    to no role is kept as a plain dimension/measure but drives no ML.
    """

    CUSTOMER_ID = "customer_id"
    CUSTOMER_NAME = "customer_name"
    ORDER_ID = "order_id"
    ORDER_DATE = "order_date"
    REVENUE = "revenue"
    QUANTITY = "quantity"
    UNIT_PRICE = "unit_price"
    COST = "cost"
    PROFIT = "profit"
    DISCOUNT = "discount"
    PRODUCT = "product"
    CATEGORY = "category"
    SUB_CATEGORY = "sub_category"
    REGION = "region"
    COUNTRY = "country"
    STATE = "state"
    CITY = "city"
    STORE = "store"
    SUPPLIER = "supplier"
    EMPLOYEE = "employee"
    EMAIL = "email"
    PHONE = "phone"
    STOCK = "stock"
    TARGET = "target"  # explicit label column for classification, if present

    ALL: Final[tuple[str, ...]] = (
        CUSTOMER_ID, CUSTOMER_NAME, ORDER_ID, ORDER_DATE, REVENUE, QUANTITY,
        UNIT_PRICE, COST, PROFIT, DISCOUNT, PRODUCT, CATEGORY, SUB_CATEGORY,
        REGION, COUNTRY, STATE, CITY, STORE, SUPPLIER, EMPLOYEE, EMAIL, PHONE,
        STOCK, TARGET,
    )

    # Human-readable descriptions, fed to Gemini and shown on the confirm screen.
    DESCRIPTIONS: Final[dict[str, str]] = {
        CUSTOMER_ID: "Unique identifier for a customer (not their name)",
        CUSTOMER_NAME: "Human-readable customer or account name",
        ORDER_ID: "Identifier grouping line-items into one order/transaction/invoice",
        ORDER_DATE: "Date (or datetime) the transaction occurred",
        REVENUE: "Money received per row — sales, amount, total, revenue",
        QUANTITY: "Count of units in the row",
        UNIT_PRICE: "Price of a single unit",
        COST: "Cost of goods for the row",
        PROFIT: "Profit or margin for the row",
        DISCOUNT: "Discount fraction or amount applied",
        PRODUCT: "Product / item / SKU name or code",
        CATEGORY: "Top-level grouping of products",
        SUB_CATEGORY: "Finer grouping of products",
        REGION: "Sales region or zone",
        COUNTRY: "Country",
        STATE: "State / province",
        CITY: "City",
        STORE: "Store / branch / outlet / location",
        SUPPLIER: "Supplier / vendor",
        EMPLOYEE: "Employee / salesperson / agent",
        EMAIL: "Email address",
        PHONE: "Phone number",
        STOCK: "Inventory level / units on hand",
        TARGET: "An explicit label to predict (churn flag, approved/denied, etc.)",
    }


class BusinessType:
    """Business types the detector can propose. Contextual only — never gates ML."""

    RETAIL = "Retail"
    ECOMMERCE = "E-commerce"
    RESTAURANT = "Restaurant"
    HOTEL = "Hotel"
    HOSPITAL = "Hospital"
    BANK = "Bank"
    FINANCE = "Finance"
    MANUFACTURING = "Manufacturing"
    INVENTORY = "Inventory"
    SALES = "Sales"
    GENERIC = "General Business"

    ALL: Final[tuple[str, ...]] = (
        RETAIL, ECOMMERCE, RESTAURANT, HOTEL, HOSPITAL, BANK, FINANCE,
        MANUFACTURING, INVENTORY, SALES, GENERIC,
    )


@dataclass(frozen=True)
class Paths:
    base: Path = BASE_DIR
    data: Path = BASE_DIR / "data"
    uploads: Path = BASE_DIR / "data" / "uploads"
    processed: Path = BASE_DIR / "data" / "processed"
    reports: Path = BASE_DIR / "data" / "reports"
    models: Path = BASE_DIR / "models"
    vector_store: Path = BASE_DIR / "models" / "vector_store"
    logs: Path = BASE_DIR / "logs"
    assets: Path = BASE_DIR / "assets"

    @property
    def sqlite_file(self) -> Path:
        return self.data / "autobusiness.db"

    @property
    def sqlite_url(self) -> str:
        return f"sqlite:///{self.sqlite_file}"

    @property
    def sqlite_url_readonly(self) -> str:
        return f"sqlite:///file:{self.sqlite_file}?mode=ro&uri=true"

    def ensure(self) -> None:
        for d in (self.uploads, self.processed, self.reports, self.models,
                  self.vector_store, self.logs, self.assets):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DetectionConfig:
    """Schema-detection engine settings."""

    strategy: str = os.getenv("DETECTION_STRATEGY", "gemini_first")
    # confidence below which a column mapping is flagged for user confirmation
    confirm_threshold: float = _env_float("DETECTION_CONFIRM_THRESHOLD", 0.75)
    sample_rows: int = _env_int("DETECTION_SAMPLE_ROWS", 20)
    max_columns: int = _env_int("DETECTION_MAX_COLUMNS", 200)


@dataclass(frozen=True)
class LLMConfig:
    """Google Gemini settings (google-genai SDK)."""

    api_key: str = os.getenv("GEMINI_API_KEY", "")
    model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    temperature: float = _env_float("GEMINI_TEMPERATURE", 0.2)
    max_output_tokens: int = _env_int("GEMINI_MAX_TOKENS", 2048)
    timeout_seconds: int = _env_int("GEMINI_TIMEOUT", 60)

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip())


@dataclass(frozen=True)
class MLConfig:
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    forecast_horizon: int = _env_int("FORECAST_HORIZON", 6)
    churn_inactivity_multiplier: float = _env_float("CHURN_INACTIVITY_MULTIPLIER", 2.0)
    churn_min_orders: int = _env_int("CHURN_MIN_ORDERS", 3)
    segmentation_k_range: tuple[int, ...] = field(default_factory=lambda: tuple(range(2, 9)))
    segment_labels: tuple[str, ...] = ("At Risk", "Occasional", "Regular", "VIP")
    min_rows_for_ml: int = _env_int("MIN_ROWS_FOR_ML", 50)


@dataclass(frozen=True)
class RAGConfig:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = _env_int("RAG_CHUNK_SIZE", 900)
    chunk_overlap: int = _env_int("RAG_CHUNK_OVERLAP", 150)
    top_k: int = _env_int("RAG_TOP_K", 4)


@dataclass(frozen=True)
class SQLConfig:
    max_rows: int = _env_int("SQL_MAX_ROWS", 500)
    statement_timeout_ms: int = _env_int("SQL_TIMEOUT_MS", 5000)


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "AutoBusinessGPT"
    tagline: str = "Upload any business dataset. Let AI do the rest."
    version: str = "1.0.0"
    debug: bool = _env_bool("DEBUG", False)
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    max_upload_mb: int = _env_int("MAX_UPLOAD_MB", 200)

    paths: Paths = field(default_factory=Paths)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    sql: SQLConfig = field(default_factory=SQLConfig)


settings: Final[AppConfig] = AppConfig()
settings.paths.ensure()


# --------------------------------------------------------------------------- #
# Visual theme
# --------------------------------------------------------------------------- #
THEME: Final[dict[str, str]] = {
    "primary": "#4F46E5",
    "primary_light": "#6366F1",
    "accent": "#0EA5E9",
    "success": "#059669",
    "danger": "#DC2626",
    "warning": "#D97706",
    "surface": "#FFFFFF",
    "surface_alt": "#F8FAFC",
    "border": "#E2E8F0",
    "text": "#0F172A",
    "text_muted": "#64748B",
}

PALETTE: Final[list[str]] = [
    "#4F46E5", "#0EA5E9", "#059669", "#D97706", "#DC2626",
    "#7C3AED", "#DB2777", "#0891B2", "#65A30D", "#EA580C",
]

__all__ = ["settings", "AppConfig", "Paths", "Role", "BusinessType", "THEME", "PALETTE"]
