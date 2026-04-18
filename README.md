# BillWise Dashboard 🧾

BillWise Analytics is an interactive financial dashboard built with **Streamlit** that helps you track, analyze, and understand your receipt data and spending habits. It uses machine learning and Large Language Models (LLMs) to automatically categorize your expenses, detect anomalies, and even let you ask questions about your spending in natural language.

---

## 🌟 Key Features

The dashboard is divided into 7 powerful modules:

1. **📊 Overview**: Get a bird's-eye view of your finances. View crucial KPIs, track your weekly spend trend, see recently processed receipts, and catch high-priority anomaly alerts.
2. **🏷️ Categories**: Dive deep into your spending by category. Includes spend vs. quantity analysis, category heatmaps by month, and a comprehensive summary table.
3. **🏪 Vendors**: Understand where your money goes. Track your top vendors, analyze your visit frequency, monitor average spend per visit, and visualize vendor spend trends over time.
4. **📦 Items**: Gain insights at the lowest level. View your top purchased items, monthly item pricing trends, and compare raw OCR data to canonical normalized item names.
5. **🔍 Receipt Explorer**: Explore your raw data. Filter and search all receipts, drill down into line items for specific receipts, and find potential duplicate receipts or high-spend outliers.
6. **💬 Ask BillWise (Text-to-SQL)**: A natural language interface powered by Google's Gemini LLM. Ask questions about your data in plain English (e.g., *"How much did we spend on dairy last month?"*), and it dynamically translates your question into a DuckDB SQL query to fetch and chart the answer!
7. **🚨 Human Validation**: Help train the system. A two-tier review queue for flagging low or medium confidence categorizations and fixing OCR errors, ensuring data accuracy over time.

---

## 🛠 Tech Stack

- **Frontend / App Framework**: [Streamlit](https://streamlit.io/)
- **Data Processing**: Pandas, NumPy, DuckDB (for Text-to-SQL logic)
- **Visualizations**: Plotly
- **AI / LLM Integration**: Google Generative AI (Gemini)
- **Cloud Storage**: Google Cloud Storage (GCS)

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.9+ installed. It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

### 2. Install Dependencies

Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the root directory and add your API keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

If you are using Google Cloud Storage, ensure your `gcp_credentials.json` is present and the appropriate environment variable or programmatic loading path is set up for the data loaders.

### 4. Run the Dashboard

Launch the Streamlit app:

```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
├── app.py                   # Main Streamlit application and UI routing
├── analytics.py             # Data aggregation and KPI logic
├── charts.py                # Plotly visualization components
├── data_loader.py           # Logic for loading data (local or cloud)
├── text_to_sql.py           # Gemini-powered natural language to SQL engine
├── query_executor.py        # Executes structured queries against the database
├── query_parser.py          # Parses intents for the analytical queries
├── eval_text_to_sql.py      # LLM evaluation and testing functions
├── utils.py                 # Constants, formatting functions, and helpers
├── requirements.txt         # Project dependencies
└── .gitignore               # Ignored files or folders
```

---

## 📝 License

This project is proprietary and confidential. All rights reserved.
