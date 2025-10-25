import os
import asyncio
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


def _extract_last_message(run_response: Any) -> str:
    """Return the content of the last message in a RunResponse safely."""
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


def generate_ai_summary(df, date_col, cat_col, res_col, model: str = "gpt-4", client=None):
    """Generate a short AI summary for the provided dataframe sample.

    This function will create an OpenAIChatCompletionClient if `client` is not
    provided, instantiate an AssistantAgent using the correct constructor
    signature, run the agent synchronously (via asyncio.run), and return the
    assistant's final message content as a string.
    """
    # Prepare sample text safely
    try:
        sample_text = df.sample(min(len(df), 5)).to_csv(index=False)
    except Exception:
        try:
            sample_text = df.head(5).to_csv(index=False)
        except Exception:
            sample_text = ""

    prompt = f"""
You are an expert data analyst. Analyze the following IT ticket dataset sample.
Columns:
- Date: {date_col}
- Category: {cat_col}
- Resolution Time: {res_col}
Provide a concise summary (under 150 words):
1. Key ticket trends
2. Most frequent issue types
3. Suggestions to improve service

Dataset sample:
{sample_text}
"""

    try:
        if client is None:
            client = OpenAIChatCompletionClient(model=model, api_key=os.getenv("OPENAI_API_KEY"))

        assistant_agent = AssistantAgent(name="assistant", model_client=client)
        run_response = asyncio.run(assistant_agent.run(task=prompt))
        return _extract_last_message(run_response)
    except Exception as e:
        return f"⚠️ Error generating AI summary: {e}"
