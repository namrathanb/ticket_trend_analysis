import plotly.express as px
import pandas as pd

def plot_tickets_per_day(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    daily = df.groupby(pd.to_datetime(df[date_col]).dt.date).size().reset_index(name="count")
    daily.columns = [date_col, "count"]
    fig = px.line(daily, x=date_col, y="count", markers=True,
                  title="📅 Tickets per Day", hover_data={"count": True})
    fig.update_layout(height=400, margin=dict(t=50, b=50, l=50, r=50))
    return fig

def plot_tickets_by_category(df, cat_col):
    cats = df[cat_col].value_counts().head(10).reset_index()
    cats.columns = [cat_col, "count"]
    fig = px.bar(cats, x="count", y=cat_col, orientation="h",
                 title="🎟️ Tickets by Category (Top 10)",
                 hover_data={"count": True}, color="count", color_continuous_scale="Viridis")
    fig.update_layout(height=400, margin=dict(t=50, b=50, l=50, r=50))
    return fig

def plot_resolution_trend(df, date_col, cat_col, res_col):
    df[res_col] = pd.to_numeric(df[res_col], errors="coerce")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["_week"] = df[date_col].dt.to_period("W").apply(lambda r: r.start_time)
    med = df.groupby(["_week", cat_col])[res_col].median().reset_index()
    fig = px.line(med, x="_week", y=res_col, color=cat_col, markers=True,
                  title="⏱️ Median Resolution Time Trend by Category",
                  hover_data={res_col: True})
    fig.update_layout(height=400, margin=dict(t=50, b=50, l=50, r=50))
    return fig
