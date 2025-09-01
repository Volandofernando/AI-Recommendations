import pandas as pd

def load_datasets(config):
    """Load and merge datasets from config.yaml (literature + survey)."""
    try:
        lit_url = config["Dataset"]["https://github.com/Volandofernando/Material-Literature-data-/blob/main/Dataset.xlsx"]
        survey_url = config["IT Innovation in Fabric Industry  (Responses)"]["https://github.com/Volandofernando/REAL-TIME-Dataset/blob/main/IT%20Innovation%20in%20Fabric%20Industry%20%20(Responses).xlsx"]

        # Read both datasets
        df_lit = pd.read_excel(lit_url)
        df_survey = pd.read_excel(survey_url)

        # Clean column names
        def clean_columns(df):
            df.columns = (
                df.columns.str.strip()
                .str.lower()
                .str.replace(r"[^\w]", "_", regex=True)
            )
            return df

        df_lit = clean_columns(df_lit)
        df_survey = clean_columns(df_survey)

        # Merge datasets
        df = pd.concat([df_lit, df_survey], ignore_index=True, sort=False)
        return df

    except Exception as e:
        raise RuntimeError(f"❌ Failed to load datasets: {e}")
