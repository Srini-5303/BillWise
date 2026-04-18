"""
utils.py — Shared formatting helpers, date utilities, and constants.
"""
from __future__ import annotations
from datetime import date, timedelta
import pandas as pd

# ── Brand colours ──────────────────────────────────────────────────────────────
COLORS = {
    "primary":  "#d97706",
    "success":  "#059669",
    "warning":  "#f59e0b",
    "danger":   "#dc2626",
    "muted":    "#78716c",
    "surface":  "#ffffff",
    "bg":       "#f9f8f6",
}

CHART_COLORS = [
    "#d97706", "#2563eb", "#059669", "#dc2626",
    "#7c3aed", "#db2777", "#0891b2", "#65a30d",
    "#ea580c", "#0284c7", "#b45309", "#4f46e5",
]

CATEGORY_LIST = [
    "Vegetables", "Fruits", "Herbs", "Spices & Seasonings",
    "Oils & Fats", "Grains & Staples", "Pulses & Beans", "Dairy",
    "Poultry", "Meat", "Seafood", "Sauces & Condiments",
    "Bakery & Flour", "Snacks & Ready-to-Eat", "Beverages",
    "Frozen / Processed", "Other",
]

DATE_RANGE_OPTIONS = [
    "All Time", "Today", "This Week", "Last Week",
    "This Month", "Last Month", "Last 30 Days",
    "Last 90 Days", "This Year",
]

# ── Keyword → Category mapping (used when enriching raw GCS data) ──────────────
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Vegetables": [
        "tomato", "onion", "garlic", "spinach", "pepper", "lettuce",
        "carrot", "celery", "zucchini", "mushroom", "potato", "broccoli",
        "cucumber", "eggplant", "cabbage", "kale", "leek", "beetroot",
    ],
    "Fruits": [
        "apple", "banana", "orange", "lemon", "lime", "mango", "grape",
        "strawberry", "blueberry", "avocado", "pear", "peach", "cherry",
        "pineapple", "watermelon", "melon", "fig", "date",
    ],
    "Herbs": [
        "basil", "parsley", "cilantro", "thyme", "rosemary", "oregano",
        "mint", "dill", "chive", "bay leaf", "tarragon", "sage",
    ],
    "Spices & Seasonings": [
        "salt", "pepper", "cumin", "turmeric", "paprika", "cinnamon",
        "chili", "coriander", "cardamom", "clove", "nutmeg", "seasoning",
        "spice", "masala", "curry", "anise", "saffron",
    ],
    "Oils & Fats": [
        "olive oil", "oil", "butter", "ghee", "lard", "coconut oil",
        "sunflower oil", "vegetable oil", "canola", "margarine", "shortening",
    ],
    "Grains & Staples": [
        "pasta", "flour", "rice", "bread", "noodle", "oat", "wheat",
        "barley", "quinoa", "couscous", "semolina", "corn", "maize",
        "breadcrumb", "crumb", "tortilla", "cracker",
    ],
    "Pulses & Beans": [
        "lentil", "chickpea", "bean", "pea", "dal", "legume",
        "kidney bean", "black bean", "soybean", "tofu", "edamame",
    ],
    "Dairy": [
        "milk", "cheese", "cream", "yogurt", "mozzarella", "parmesan",
        "ricotta", "brie", "cheddar", "feta", "whey", "lactose",
        "paneer", "kefir", "custard",
    ],
    "Poultry": [
        "chicken", "turkey", "duck", "hen", "quail", "poultry",
        "breast", "thigh", "drumstick", "wing", "ground turkey",
    ],
    "Meat": [
        "beef", "pork", "lamb", "veal", "steak", "mince", "sausage",
        "bacon", "ribs", "brisket", "loin", "mutton", "prosciutto",
        "salami", "pepperoni", "chorizo", "ground beef",
    ],
    "Seafood": [
        "salmon", "shrimp", "tuna", "cod", "bass", "prawn", "lobster",
        "crab", "scallop", "fish", "squid", "octopus", "mussel", "oyster",
        "tilapia", "halibut", "anchovy", "sardine",
    ],
    "Sauces & Condiments": [
        "sauce", "ketchup", "mustard", "mayonnaise", "vinegar", "soy sauce",
        "hot sauce", "salsa", "pesto", "relish", "dressing", "marinade",
        "stock", "broth", "paste", "gravy", "jam", "honey",
    ],
    "Bakery & Flour": [
        "croissant", "roll", "focaccia", "pastry", "cake", "muffin",
        "loaf", "brioche", "sourdough", "baguette", "bun", "donut",
        "bagel", "waffle", "pancake", "baking powder", "yeast",
    ],
    "Snacks & Ready-to-Eat": [
        "chips", "crisp", "snack", "cookie", "biscuit", "candy",
        "chocolate", "granola", "bar", "popcorn", "pretzel", "nuts",
        "dried fruit", "trail mix", "ready meal", "instant",
    ],
    "Beverages": [
        "water", "juice", "coffee", "tea", "wine", "beer", "soda",
        "espresso", "syrup", "cocktail", "sparkling", "still",
        "smoothie", "kombucha", "energy drink", "lemonade",
    ],
    "Frozen / Processed": [
        "frozen", "frzn", "processed", "canned", "preserved", "pickled",
        "freeze", "ice cream", "pizza", "nugget", "finger", "ready",
        "microwave", "tinned", "jar",
    ],
    "Other": [],
}


def categorize_item(text: str) -> tuple[str, float]:
    """Return (category, confidence) for a raw item text string."""
    if not text:
        return "Other", 0.50
    lower = text.lower()
    scores: dict[str, int] = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits:
            scores[cat] = hits
    if not scores:
        return "Other", 0.45
    best = max(scores, key=scores.__getitem__)
    # Confidence scales with number of keyword hits
    conf = min(0.95, 0.60 + scores[best] * 0.12)
    return best, round(conf, 2)


def get_top3_predictions(text: str) -> list[tuple[str, float]]:
    """
    Return the top 3 (category, confidence) predictions for a raw item text.
    Used by the Human Validation page to show model predictions alongside each
    flagged item so reviewers can make fast, informed decisions.
    """
    _FALLBACKS = [
        ("Other", 0.30),
        ("Grains & Staples", 0.25),
        ("Snacks & Ready-to-Eat", 0.20),
    ]
    if not text:
        return _FALLBACKS

    lower = text.lower()
    scores: dict[str, int] = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits:
            scores[cat] = hits

    if not scores:
        return _FALLBACKS

    # Sort by hit count descending, take top 3
    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result: list[tuple[str, float]] = []
    for cat, hits in sorted_cats[:3]:
        conf = min(0.95, 0.60 + hits * 0.12)
        result.append((cat, round(conf, 2)))

    # Pad to exactly 3 predictions with generic fallbacks
    used = {c for c, _ in result}
    for fb_cat, fb_conf in _FALLBACKS:
        if len(result) >= 3:
            break
        if fb_cat not in used:
            result.append((fb_cat, fb_conf))

    return result[:3]


# ── Formatters ─────────────────────────────────────────────────────────────────
def fmt_currency(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def fmt_number(x) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return "0"


def fmt_pct(x) -> str:
    try:
        return f"{float(x):.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def confidence_label(score: float) -> str:
    if pd.isna(score):
        return "⚪ Unknown"
    if score >= 0.90:
        return "🟢 High"
    elif score >= 0.70:
        return "🟡 Medium"
    return "🔴 Low"


# ── Date utilities ─────────────────────────────────────────────────────────────
def resolve_date_range(range_str: str | None) -> tuple[date | None, date | None]:
    """Convert named range string → (start_date, end_date). Returns (None, None) for All Time."""
    if not range_str or range_str.lower() in ("all time", "all_time", ""):
        return None, None

    today = date.today()
    r = range_str.lower().replace(" ", "_")

    if r == "today":
        return today, today
    elif r == "this_week":
        return today - timedelta(days=today.weekday()), today
    elif r == "last_week":
        start = today - timedelta(days=today.weekday() + 7)
        return start, start + timedelta(days=6)
    elif r == "this_month":
        return today.replace(day=1), today
    elif r == "last_month":
        first_this = today.replace(day=1)
        last_prev  = first_this - timedelta(days=1)
        return last_prev.replace(day=1), last_prev
    elif r == "last_30_days":
        return today - timedelta(days=30), today
    elif r == "last_90_days":
        return today - timedelta(days=90), today
    elif r == "this_year":
        return today.replace(month=1, day=1), today
    return None, None


def apply_date_filter(
    df: pd.DataFrame,
    date_col: str,
    start: date | None,
    end: date | None,
) -> pd.DataFrame:
    """Mask a DataFrame on a datetime column between start and end."""
    if df.empty:
        return df
    if start:
        df = df[df[date_col].dt.date >= start]
    if end:
        df = df[df[date_col].dt.date <= end]
    return df
