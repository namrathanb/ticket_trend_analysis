from flask import Flask, request, jsonify
import os
import tempfile
import io
import pandas as pd
import plotly.io as pio
from dotenv import load_dotenv
from utils.column_detection import detect_date_col, detect_category_col, detect_resolution_col, detect_ticket_id_col
from utils.plotting import plot_tickets_per_day, plot_tickets_by_category, plot_resolution_trend
from utils.ai_summary import generate_ai_summary
from utils.chatbot_impl import chatbot_query

# Import Autogen-specific classes
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables
load_dotenv()

# Initialize OpenAIChatCompletionClient (Autogen client) with a model
model_client=OpenAIChatCompletionClient(
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY")
    )

# Create an assistant agent for conversation handling
# AssistantAgent expects a `name` and a `model_client` (see autogen API)
assistant_agent = AssistantAgent(name="assistant", model_client=model_client)

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    suffix = os.path.splitext(f.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name
    try:
        if suffix == ".csv":
            df = pd.read_csv(tmp_path)
        else:
            df = pd.read_excel(tmp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    date_col = detect_date_col(df)
    cat_col = detect_category_col(df)
    res_col = detect_resolution_col(df)
    ticket_col = detect_ticket_id_col(df)

    # Prepare figures
    figs = {}
    if date_col is not None:
        figs["tickets_per_day"] = pio.to_json(plot_tickets_per_day(df.copy(), date_col))
    if cat_col is not None:
        figs["tickets_by_category"] = pio.to_json(plot_tickets_by_category(df.copy(), cat_col))
    if res_col is not None and cat_col is not None and date_col is not None:
        figs["resolution_trend"] = pio.to_json(plot_resolution_trend(df.copy(), date_col, cat_col, res_col))

    model = request.args.get("model", "gpt-4")  # You can specify the model via query parameter
    summary = generate_ai_summary(df.copy(), date_col, cat_col, res_col, model=model)

    # KPIs
    kpis = {
        "total_tickets": int(df.shape[0]),
        "avg_resolution_time": None,
        "peak_category": None
    }
    if res_col is not None:
        kpis["avg_resolution_time"] = round(pd.to_numeric(df[res_col], errors='coerce').mean(), 2)
    if cat_col is not None and cat_col in df.columns and not df[cat_col].isna().all():
        try:
            kpis["peak_category"] = str(df[cat_col].value_counts().idxmax())
        except Exception:
            kpis["peak_category"] = None

    # Send back first 1000 rows for chatbot convenience
    csv_buffer = df.head(1000).to_csv(index=False)

    return jsonify({
        "date_col": date_col,
        "cat_col": cat_col,
        "res_col": res_col,
        "ticket_col": ticket_col,
        "figs": figs,
        "summary": summary,
        "kpis": kpis,
        "dataset_sample_csv": csv_buffer
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "")
    csv_sample = data.get("dataset_sample_csv", "")

    # Get model name (default to gpt-4)
    model = data.get("model", "gpt-4")

    # Reuse chatbot logic with updated client and model
    try:
        df = pd.read_csv(io.StringIO(csv_sample))
    except Exception:
        df = pd.DataFrame()

    # Delegate to chatbot helper which will instantiate/use the agent correctly
    response = chatbot_query(df, question, client=model_client, model=model)

    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(port=8000, debug=True)
