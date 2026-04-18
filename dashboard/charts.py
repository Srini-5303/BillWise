"""
charts.py — All Plotly chart builders for BillWise Dashboard.

Every function returns a plotly.graph_objects.Figure.
Charts share the BillWise brand palette and a clean white theme.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import CHART_COLORS, COLORS

# ── Shared layout defaults ─────────────────────────────────────────────────────
_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font         =dict(family="Inter, system-ui, sans-serif", color="#1c1917"),
    margin       =dict(l=10, r=10, t=40, b=10),
    legend       =dict(orientation="h", yanchor="bottom", y=1.02,
                       xanchor="right", x=1),
    hoverlabel   =dict(bgcolor="white", font_size=13, font_color="#1c1917"),
)


def _apply(fig: go.Figure, title: str = "", height: int = 340) -> go.Figure:
    fig.update_layout(
        **_LAYOUT,
        title=dict(text=title, font=dict(size=15, color="#1c1917")),
        height=height,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def spend_trend_chart(df: pd.DataFrame, title: str = "Spend Over Time") -> go.Figure:
    """Area line chart — receipt_date vs spend."""
    if df.empty:
        return _empty("No spend data for this period")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["receipt_date"],
        y=df["spend"],
        mode="lines",
        fill="tozeroy",
        line=dict(color=COLORS["primary"], width=2.5),
        fillcolor="rgba(217,119,6,0.12)",
        hovertemplate="<b>%{x|%b %d}</b><br>Spend: $%{y:,.2f}<extra></extra>",
        name="Spend",
    ))
    fig.update_xaxes(showgrid=False, tickformat="%b %d")
    fig.update_yaxes(showgrid=True, gridcolor="#f0ede8",
                     tickprefix="$", tickformat=",.0f")
    return _apply(fig, title, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORIES
# ══════════════════════════════════════════════════════════════════════════════

def category_donut_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart of spend by category."""
    if df.empty:
        return _empty("No category data")

    fig = go.Figure(go.Pie(
        labels=df["category"],
        values=df["spend"],
        hole=0.55,
        marker=dict(colors=CHART_COLORS[:len(df)]),
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f} (%{percent})<extra></extra>",
    ))
    # Build layout without the shared legend key to avoid duplicate kwarg error
    _layout_no_legend = {k: v for k, v in _LAYOUT.items() if k != "legend"}
    fig.update_layout(
        **_layout_no_legend,
        height=320,
        showlegend=True,
        legend=dict(orientation="v", x=1.0, y=0.5),
        annotations=[dict(text="Spend", x=0.5, y=0.5,
                          font_size=13, showarrow=False,
                          font_color=COLORS["muted"])],
    )
    return fig


def category_bar_chart(df: pd.DataFrame, metric: str = "spend") -> go.Figure:
    """Horizontal bar chart — category vs spend or quantity."""
    if df.empty:
        return _empty("No category data")

    label  = "Spend ($)" if metric == "spend" else "Quantity"
    prefix = "$" if metric == "spend" else ""
    df_s   = df.sort_values(metric, ascending=True)

    fig = go.Figure(go.Bar(
        x=df_s[metric],
        y=df_s["category"],
        orientation="h",
        marker_color=COLORS["primary"],
        hovertemplate=f"<b>%{{y}}</b><br>{label}: {prefix}%{{x:,.2f}}<extra></extra>",
        text=[f"{prefix}{v:,.0f}" for v in df_s[metric]],
        textposition="outside",
    ))
    fig.update_xaxes(showgrid=True, gridcolor="#f0ede8")
    fig.update_yaxes(showgrid=False)
    return _apply(fig, f"Category by {label}", height=320)


def category_heatmap(df_pivot: pd.DataFrame) -> go.Figure:
    """Month × category spend heatmap."""
    if df_pivot.empty or len(df_pivot.columns) < 2:
        return _empty("Not enough data for trend")

    months = df_pivot["month"].tolist()
    cats   = [c for c in df_pivot.columns if c != "month"]
    z      = df_pivot[cats].values.T.tolist()

    fig = go.Figure(go.Heatmap(
        z=z,
        x=months,
        y=cats,
        colorscale=[[0, "#fef3c7"], [0.5, "#f59e0b"], [1, "#92400e"]],
        hovertemplate="<b>%{y}</b> — %{x}<br>Spend: $%{z:,.2f}<extra></extra>",
        showscale=True,
    ))
    fig.update_xaxes(tickangle=-30)
    return _apply(fig, "Monthly Category Spend Heatmap", height=320)


def category_line_chart(df_pivot: pd.DataFrame) -> go.Figure:
    """Multi-line chart — monthly trend per category."""
    if df_pivot.empty or len(df_pivot.columns) < 2:
        return _empty("Not enough data for trend")

    months = df_pivot["month"].tolist()
    cats   = [c for c in df_pivot.columns if c != "month"]

    fig = go.Figure()
    for i, cat in enumerate(cats):
        fig.add_trace(go.Scatter(
            x=months,
            y=df_pivot[cat],
            mode="lines+markers",
            name=cat,
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            hovertemplate=f"<b>{cat}</b><br>%{{x}}: $%{{y:,.2f}}<extra></extra>",
        ))
    fig.update_xaxes(showgrid=False, tickangle=-30)
    fig.update_yaxes(showgrid=True, gridcolor="#f0ede8",
                     tickprefix="$", tickformat=",.0f")
    return _apply(fig, "Category Spend Trend by Month", height=340)


# ══════════════════════════════════════════════════════════════════════════════
# VENDORS
# ══════════════════════════════════════════════════════════════════════════════

def vendor_bar_chart(df: pd.DataFrame, metric: str = "spend") -> go.Figure:
    """Horizontal bar — top vendors."""
    if df.empty:
        return _empty("No vendor data")

    label  = "Total Spend ($)" if metric == "spend" else "Receipt Count"
    prefix = "$"               if metric == "spend" else ""
    col    = "spend"           if metric == "spend" else "receipts"
    df_s   = df.sort_values(col, ascending=True)

    colors = [COLORS["primary"]] * len(df_s)
    colors[-1] = "#92400e"   # highlight top vendor (darker amber)

    fig = go.Figure(go.Bar(
        x=df_s[col],
        y=df_s["vendor_name"],
        orientation="h",
        marker_color=list(reversed(colors)),
        hovertemplate=f"<b>%{{y}}</b><br>{label}: {prefix}%{{x:,.2f}}<extra></extra>",
        text=[f"{prefix}{v:,.0f}" for v in df_s[col]],
        textposition="outside",
    ))
    fig.update_xaxes(showgrid=True, gridcolor="#f0ede8")
    fig.update_yaxes(showgrid=False)
    return _apply(fig, f"Top Vendors by {label}", height=max(300, len(df) * 35))


def vendor_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Multi-line monthly spend trend per vendor."""
    if df.empty:
        return _empty("No vendor trend data")

    vendors = df["vendor_name"].unique()
    fig     = go.Figure()
    for i, vendor in enumerate(vendors):
        vdf = df[df["vendor_name"] == vendor]
        fig.add_trace(go.Scatter(
            x=vdf["month"],
            y=vdf["spend"],
            mode="lines+markers",
            name=vendor,
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            hovertemplate=f"<b>{vendor}</b><br>%{{x}}: $%{{y:,.2f}}<extra></extra>",
        ))
    fig.update_xaxes(showgrid=False, tickangle=-30)
    fig.update_yaxes(showgrid=True, gridcolor="#f0ede8",
                     tickprefix="$", tickformat=",.0f")
    return _apply(fig, "Vendor Spend Trend", height=340)


# ══════════════════════════════════════════════════════════════════════════════
# ITEMS
# ══════════════════════════════════════════════════════════════════════════════

def item_bar_chart(df: pd.DataFrame, metric: str = "spend") -> go.Figure:
    """Horizontal bar — top items by spend or quantity."""
    if df.empty:
        return _empty("No item data")

    col    = "spend"    if metric == "spend" else "quantity"
    label  = "Spend ($)" if metric == "spend" else "Quantity"
    prefix = "$"         if metric == "spend" else ""
    df_s   = df.sort_values(col, ascending=True)

    # Colour bars by category
    cat_color = {cat: CHART_COLORS[i % len(CHART_COLORS)]
                 for i, cat in enumerate(df_s["category"].unique())}
    bar_colors = [cat_color.get(c, COLORS["primary"])
                  for c in df_s["category"]]

    fig = go.Figure(go.Bar(
        x=df_s[col],
        y=df_s["matched_canonical_item"],
        orientation="h",
        marker_color=bar_colors,
        hovertemplate=f"<b>%{{y}}</b><br>{label}: {prefix}%{{x:,.2f}}<extra></extra>",
    ))
    fig.update_xaxes(showgrid=True, gridcolor="#f0ede8")
    fig.update_yaxes(showgrid=False)
    return _apply(fig, f"Top Items by {label}", height=max(320, len(df) * 26))


def item_trend_chart(df: pd.DataFrame, item: str) -> go.Figure:
    """Dual-axis monthly trend — spend + quantity for one item."""
    if df.empty:
        return _empty(f"No data for '{item}'")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["month"],
        y=df["spend"],
        name="Spend ($)",
        marker_color="rgba(217,119,6,0.6)",
        yaxis="y1",
        hovertemplate="<b>%{x}</b><br>Spend: $%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["month"],
        y=df["quantity"],
        name="Quantity",
        mode="lines+markers",
        line=dict(color=COLORS["success"], width=2),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Qty: %{y:,.1f}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT,
        title=dict(text=f"{item} — Monthly Trend", font=dict(size=15)),
        height=300,
        yaxis =dict(title="Spend ($)", tickprefix="$", showgrid=True, gridcolor="#f0ede8"),
        yaxis2=dict(title="Quantity", overlaying="y", side="right", showgrid=False),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# RECEIPT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def spend_distribution_chart(df_receipts: pd.DataFrame) -> go.Figure:
    """Histogram of receipt totals to show spend distribution."""
    if df_receipts.empty:
        return _empty("No receipt data")

    fig = go.Figure(go.Histogram(
        x=df_receipts["receipt_total"],
        nbinsx=25,
        marker_color=COLORS["primary"],
        opacity=0.8,
        hovertemplate="Range: $%{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_xaxes(tickprefix="$", title="Receipt Total")
    fig.update_yaxes(title="Count")
    return _apply(fig, "Receipt Spend Distribution", height=280)


# ══════════════════════════════════════════════════════════════════════════════
# ASK BILLWISE — result chart picker
# ══════════════════════════════════════════════════════════════════════════════

def result_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Query Result",
) -> go.Figure:
    """Generic horizontal bar for NL query results."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return _empty("No chart data")

    df_s = df.sort_values(y_col, ascending=True).tail(15)
    is_money = any(w in y_col.lower() for w in ["spend", "total", "price", "cost"])
    prefix   = "$" if is_money else ""

    fig = go.Figure(go.Bar(
        x=df_s[y_col],
        y=df_s[x_col].astype(str),
        orientation="h",
        marker_color=COLORS["primary"],
        hovertemplate=f"<b>%{{y}}</b><br>{y_col}: {prefix}%{{x:,.2f}}<extra></extra>",
        text=[f"{prefix}{v:,.1f}" for v in df_s[y_col]],
        textposition="outside",
    ))
    fig.update_xaxes(showgrid=True, gridcolor="#f0ede8")
    fig.update_yaxes(showgrid=False)
    return _apply(fig, title, height=max(260, len(df_s) * 30))


def result_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Trend",
) -> go.Figure:
    """Generic line chart for NL query trend results."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return _empty("No chart data")

    is_money = any(w in y_col.lower() for w in ["spend", "total", "price", "cost"])
    fig = go.Figure(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=2.5),
        fill="tozeroy",
        fillcolor="rgba(217,119,6,0.10)",
        hovertemplate=f"<b>%{{x}}</b><br>{y_col}: {'$' if is_money else ''}%{{y:,.2f}}<extra></extra>",
    ))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0ede8",
                     tickprefix="$" if is_money else "")
    return _apply(fig, title, height=280)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _empty(message: str = "No data available") -> go.Figure:
    """Placeholder figure shown when there is no data."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color=COLORS["muted"]),
    )
    fig.update_layout(**_LAYOUT, height=260)
    return fig
