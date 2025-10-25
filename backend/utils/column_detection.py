import pandas as pd

def detect_date_col(df):
    for col in df.columns:
        if any(k in col.lower() for k in ["date", "created", "time", "timestamp"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                if df[col].notna().sum() > 0:
                    return col
            except Exception:
                pass
    return None

def detect_category_col(df):
    for col in df.columns:
        if any(k in col.lower() for k in ["category", "type", "issue", "priority", "queue", "status"]):
            return col
    obj_cols = [c for c in df.columns if df[c].dtype == object and df[c].nunique() < 50]
    return obj_cols[0] if obj_cols else None

def detect_resolution_col(df):
    for col in df.columns:
        if "resolution" in col.lower():
            return col
    return None

def detect_ticket_id_col(df):
    for col in df.columns:
        if any(k in col.lower() for k in ["ticket", "id", "case", "issue"]):
            return col
    return None
