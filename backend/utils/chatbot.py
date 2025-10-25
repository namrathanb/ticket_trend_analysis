import io
import os
import asyncio
from typing import Any, Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


def _extract_last_message(run_response: Any) -> str:
    """Safely extract the last message content from a RunResponse-like object."""
    try:
        last = None
        for m in getattr(run_response, "messages", []) or []:
            last = m
    except Exception:
        return ""

    if last is None:
        return ""

    if isinstance(last, dict):
        return last.get("content", str(last))
    return getattr(last, "content", str(last))


def chatbot_query(df, query: str, client: Optional[Any] = None, model: str = "gpt-4") -> str:
    """Query a dataset (pandas DataFrame) with a lightweight heuristic first, then fall back to the LLM.

    This function is intended to be called from synchronous Flask handlers. The underlying
    agent `run` method is asynchronous, so we call it with `asyncio.run(...)` to keep
    the Flask request flow synchronous and avoid "coroutine was never awaited" warnings.
    """
    try:
        q = (query or "").strip()

        # Ensure model client
        if client is None:
            client = OpenAIChatCompletionClient(model=model, api_key=os.getenv("OPENAI_API_KEY"))

        assistant_agent = AssistantAgent(name="assistant", model_client=client)

        # Quick ticket lookup heuristics
        if df is not None and hasattr(df, "columns") and len(df.columns) > 0:
            ticket_col = next((c for c in df.columns if any(k in c.lower() for k in ["ticket", "id", "case", "issue"])), None)
            res_col = next((c for c in df.columns if "resolution" in c.lower()), None)

            if ticket_col and any(word in q.lower() for word in ["ticket", "id", "case", "issue"]):
                import re

                m = re.search(r"\d+", q)
                if m:
                    ticket_id = m.group()
                    mask = df[ticket_col].astype(str).str.contains(ticket_id, case=False, na=False)
                    if mask.any():
                        row = df[mask].iloc[0]
                        if res_col and res_col in row:
                            return f"🕒 The resolution time for ticket **{row[ticket_col]}** is **{row[res_col]}**."
                        return f"✅ Ticket **{row[ticket_col]}** exists, but resolution time not available."
                    return f"⚠️ No ticket found with ID '{ticket_id}'."

        # Quick statistics heuristic
        if any(k in q.lower() for k in ["average", "mean", "trend"]):
            if df is not None and hasattr(df, "columns"):
                cat_col = next((c for c in df.columns if any(k in c.lower() for k in ["category", "type", "issue", "priority", "queue", "status"])), None)
                res_col = next((c for c in df.columns if "resolution" in c.lower()), None)
                if cat_col and res_col:
                    try:
                        avg = df.groupby(cat_col)[res_col].mean(numeric_only=True).sort_values(ascending=False)
                        if not avg.empty:
                            best_cat = avg.idxmax()
                            return f"📊 The category with the highest average resolution time is **{best_cat}** ({avg[best_cat]:.2f} units)."
                    except Exception:
                        # fall through to LLM fallback
                        pass

        # LLM fallback: hand down a sample (up to 1000 rows) and the question
        csv_buffer = io.StringIO()
        if df is not None:
            try:
                df.head(1000).to_csv(csv_buffer, index=False)
            except Exception:
                # If df can't be serialized, leave dataset_sample empty
                pass
        dataset_sample = csv_buffer.getvalue()

        prompt = f"""
You are a data analyst assistant. Answer the user's question directly and concisely.
Dataset sample (up to 1000 rows):
{dataset_sample}
Question: "{q}"
"""

        # Call the agent's async run() synchronously for Flask contexts
        run_response = asyncio.run(assistant_agent.run(task=prompt))
        return _extract_last_message(run_response)

    except Exception as e:
        return f"⚠️ Chatbot error: {e}"
