import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns

# ==========================
# Univariate Charts
# ==========================
def univariate_chart(df, col, chart_type="Histogram"):
    """
    Generates univariate charts: Histogram, Pie, Boxplot, Violin, Cumulative
    """
    if chart_type == "Histogram":
        fig = px.histogram(df, x=col, nbins=20, title=f"Histogram of {col}")
    elif chart_type == "Pie":
        fig = px.pie(df, names=col, title=f"Pie Chart of {col}")
    elif chart_type == "Boxplot":
        fig = px.box(df, y=col, title=f"Boxplot of {col}")
    elif chart_type == "Violin":
        fig = px.violin(df, y=col, box=True, points="all", title=f"Violin Plot of {col}")
    elif chart_type == "Cumulative":
        if pd.api.types.is_numeric_dtype(df[col]):
            df_sorted = df.sort_values(col)
            df_sorted['Cumulative'] = df_sorted[col].cumsum()
            fig = px.line(df_sorted, y='Cumulative', title=f"Cumulative Plot of {col}")
        else:
            return None
    else:
        return None
    return fig

# ==========================
# Bivariate Charts
# ==========================
def bivariate_chart(df, x_col, y_col, chart_type="Scatter", color_col=None):
    """
    Generates bivariate charts: Scatter, Line, Box, Bar, Bubble
    """
    if chart_type == "Scatter":
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
    elif chart_type == "Line":
        fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} over {x_col}")
    elif chart_type == "Box":
        fig = px.box(df, x=x_col, y=y_col, color=color_col, title=f"Boxplot: {y_col} vs {x_col}")
    elif chart_type == "Bar":
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"Bar Chart: {y_col} vs {x_col}")
    elif chart_type == "Bubble":
        size_col = y_col if color_col is None else color_col
        fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col, title=f"Bubble Chart: {y_col} vs {x_col}")
    else:
        return None
    return fig

# ==========================
# Multivariate Charts
# ==========================
def correlation_heatmap(df, numeric_cols=None):
    """
    Generates correlation heatmap for numeric columns
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    return fig

def pairplot(df, numeric_cols=None):
    """
    Generates scatter matrix (pairplot) for numeric columns
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
    fig = px.scatter_matrix(df, dimensions=numeric_cols, title="Scatter Matrix / Pairplot")
    return fig

def stacked_bar(df, x_col, y_cols, color_col=None):
    """
    Generates stacked bar chart
    x_col: categorical x-axis
    y_cols: list of numeric columns to stack
    """
    fig = go.Figure()
    for y in y_cols:
        fig.add_trace(go.Bar(
            x=df[x_col],
            y=df[y],
            name=y
        ))
    fig.update_layout(barmode='stack', title="Stacked Bar Chart")
    return fig

# ==========================
# Time Series Charts
# ==========================
def timeseries_chart(df, x_col, y_col, color_col=None):
    """
    Generates time series line chart
    """
    fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"Time Series: {y_col} over {x_col}")
    return fig

