# AI Fabric Recommender (Streamlit)

Live app (example): your Streamlit Cloud URL here.

## What it does
- Loads two Excel datasets (literature + survey) directly from GitHub
- Automatically detects feature/target columns
- Trains a Random Forest model to predict comfort score
- Lets a user set conditions and recommends the top 3 closest fabrics
- Shows metrics, feature importance, and dataset insights
- Optional analytics logging via Supabase (off by default)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
