import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# 📊 Univariate Charts
# ==========================================================
def univariate_chart(df, x=None, hue=None, chart_type=None, title=None):
    """
    Flexible univariate visualization.
    - Auto-detects numeric/categorical.
    - Supports hue (color grouping).
    - Chart types: Histogram, Bar, Pie, Boxplot, Violin, Cumulative.
    """
    if x is None or x not in df.columns:
        return None

    is_numeric = pd.api.types.is_numeric_dtype(df[x])
    if title is None:
        title = f"{x} Distribution" if hue is None else f"{x} by {hue}"

    # Auto-select chart type
    if chart_type is None:
        chart_type = "Histogram" if is_numeric else "Bar"

    # --- Chart Logic ---
    if chart_type == "Histogram":
        fig = px.histogram(df, x=x, color=hue, nbins=20, title=title)

    elif chart_type == "Bar":
        if hue:
            df_grouped = df.groupby([x, hue]).size().reset_index(name="Count")
            fig = px.bar(df_grouped, x=x, y="Count", color=hue, title=title)
        else:
            df_grouped = df[x].value_counts().reset_index()
            df_grouped.columns = [x, "Count"]
            fig = px.bar(df_grouped, x=x, y="Count", title=title)

    elif chart_type == "Pie":
        fig = px.pie(df, names=x, title=title)

    elif chart_type == "Boxplot":
        fig = px.box(df, y=x if hue is None else None, x=hue if hue else None, color=hue, title=title)

    elif chart_type == "Violin":
        fig = px.violin(df, y=x, color=hue, box=True, points="all", title=title)

    elif chart_type == "Cumulative":
        if is_numeric:
            df_sorted = df.sort_values(x)
            df_sorted["Cumulative"] = df_sorted[x].cumsum()
            fig = px.line(df_sorted, y="Cumulative", title=f"Cumulative Plot of {x}")
        else:
            return None

    else:
        return None

    fig.update_layout(
        template="simple_white",
        title_font=dict(size=18),
        xaxis_title=x,
        yaxis_title="Count" if not is_numeric else "Frequency",
        legend_title=hue if hue else "",
    )
    return fig


# ==========================================================
# 📈 Bivariate Charts
# ==========================================================
def bivariate_chart(df, x, y, hue=None, chart_type=None, title=None):
    """
    Automatically selects visualization for numeric/categorical combinations.
    Chart types: Scatter, Line, Bar, Box, Bubble.
    """
    if x not in df.columns or y not in df.columns:
        return None

    x_numeric = pd.api.types.is_numeric_dtype(df[x])
    y_numeric = pd.api.types.is_numeric_dtype(df[y])

    if title is None:
        title = f"{y} vs {x}"

    # Auto-select chart type
    if chart_type is None:
        if x_numeric and y_numeric:
            chart_type = "Scatter"
        elif not x_numeric and y_numeric:
            chart_type = "Bar"
        elif x_numeric and not y_numeric:
            chart_type = "Box"
        else:
            chart_type = "Bar"

    # --- Chart Logic ---
    if chart_type == "Scatter":
        fig = px.scatter(df, x=x, y=y, color=hue, title=title)
    elif chart_type == "Line":
        fig = px.line(df, x=x, y=y, color=hue, title=title)
    elif chart_type == "Bar":
        fig = px.bar(df, x=x, y=y, color=hue, title=title)
    elif chart_type == "Box":
        fig = px.box(df, x=x, y=y, color=hue, title=title)
    elif chart_type == "Bubble":
        fig = px.scatter(df, x=x, y=y, size=y, color=hue, title=title)
    else:
        return None

    fig.update_layout(
        template="simple_white",
        title_font=dict(size=18),
        xaxis_title=x,
        yaxis_title=y,
        legend_title=hue if hue else "",
    )
    return fig


# ==========================================================
# 🧮 Multivariate Charts
# ==========================================================
def correlation_heatmap(df, numeric_cols=None):
    """
    Generates correlation heatmap for numeric columns.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if not numeric_cols:
        return None

    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap", color_continuous_scale="RdBu_r")
    fig.update_layout(template="simple_white")
    return fig


def pairplot(df, numeric_cols=None):
    """
    Scatter matrix (pairplot) for numeric columns.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if not numeric_cols:
        return None

    fig = px.scatter_matrix(df, dimensions=numeric_cols, title="Scatter Matrix / Pairplot")
    fig.update_layout(template="simple_white")
    return fig


def stacked_bar(df, x_col, y_cols, color_col=None):
    """
    Stacked bar chart for multiple numeric columns.
    """
    fig = go.Figure()
    for y in y_cols:
        fig.add_trace(go.Bar(
            x=df[x_col],
            y=df[y],
            name=y
        ))
    fig.update_layout(barmode="stack", title="Stacked Bar Chart", template="simple_white")
    return fig


# ==========================================================
# ⏱️ Time Series Charts
# ==========================================================
def timeseries_chart(df, x_col, y_col, color_col=None):
    """
    Time series chart with automatic grouping by color_col if provided.
    """
    if x_col not in df.columns or y_col not in df.columns:
        return None

    fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"Time Series: {y_col} over {x_col}")
    fig.update_layout(template="simple_white")
    return fig
