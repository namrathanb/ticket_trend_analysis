import io
import os
import asyncio
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


def _extract_last_message(run_response: Any) -> str:
    """Safely extract the last message content from a RunResponse."""
    last = None
    try:
        for m in run_response.messages:
            last = m
    except Exception:
        return ""

    if last is None:
        return ""

    # Messages may be dict-like or objects with `content` attr
    if isinstance(last, dict):
        return last.get("content", str(last))
    return getattr(last, "content", str(last))


def chatbot_query(df, query, client=None, model: str = "gpt-4"):
    # Ensure the query is clean
    q = (query or "").lower().strip()

    # Ensure we have a model client
    if client is None:
        client = OpenAIChatCompletionClient(model=model, api_key=os.getenv("OPENAI_API_KEY"))

    # Initialize the AssistantAgent with the proper signature
    assistant_agent = AssistantAgent(name="assistant", model_client=client)

    # Ticket-specific queries (Handling ticket ID related queries)
    if df is not None and not df.empty:
        ticket_col = None
        for col in df.columns:
            if any(k in col.lower() for k in ["ticket", "id", "case", "issue"]):
                ticket_col = col
                break
        res_col = None
        for col in df.columns:
            if "resolution" in col.lower():
                res_col = col
                break
        if ticket_col and any(word in q for word in ["ticket", "id", "case", "issue"]):
            import re

            num_match = re.search(r"\d+", q)
            if num_match:
                ticket_id = num_match.group()
                mask = df[ticket_col].astype(str).str.contains(ticket_id, case=False, na=False)
                if mask.any():
                    row = df[mask].iloc[0]
                    if res_col and res_col in row:
                        return f"🕒 The resolution time for ticket **{row[ticket_col]}** is **{row[res_col]}**."
                    else:
                        return f"✅ Ticket **{row[ticket_col]}** exists, but resolution time not available."
                else:
                    return f"⚠️ No ticket found with ID '{ticket_id}'."

    # Average/summary queries (Handling statistics)
    if "average" in q or "mean" in q or "trend" in q:
        if df is not None and not df.empty:
            cat_col = None
            for col in df.columns:
                if any(k in col.lower() for k in ["category", "type", "issue", "priority", "queue", "status"]):
                    cat_col = col
                    break
            for col in df.columns:
                if "resolution" in col.lower():
                    res_col = col
                    break
            if cat_col and res_col:
                avg = df.groupby(cat_col)[res_col].mean(numeric_only=True).sort_values(ascending=False).head(5)
                if not avg.empty:
                    best_cat = avg.idxmax()
                    return f"📊 The category with the highest average resolution time is **{best_cat}** ({avg[best_cat]:.2f} units)."

    # Fallback LLM (first 1000 rows)
    try:
        csv_buffer = io.StringIO()
        if df is not None:
            try:
                df.head(1000).to_csv(csv_buffer, index=False)
            except Exception:
                # If serialization fails, proceed with empty sample
                pass
        dataset_sample = csv_buffer.getvalue()

        prompt = f"""
You are a data analyst assistant. Answer the user's question directly and concisely.
Dataset sample (up to 1000 rows):
{dataset_sample}
Question: "{query}"
"""

        # Use AssistantAgent to get response (synchronous `run` wrapper)
        run_response = asyncio.run(assistant_agent.run(task=prompt))
        return _extract_last_message(run_response)
    except Exception as e:
        return f"⚠️ Chatbot error: {e}"
