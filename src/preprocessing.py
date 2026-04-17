import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        pass

    def load_data(self, filepath: str) -> pd.DataFrame:
        logger.info(f"Loading data from {filepath}")
        return pd.read_csv(filepath)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning data...")
        df = df.copy()
        
        missing_pct = df.isnull().mean()
        cols_to_drop = missing_pct[missing_pct > 0.5].index
        if len(cols_to_drop) > 0:
            logger.info(f"Dropping columns with >50% missing: {list(cols_to_drop)}")
            df = df.drop(columns=cols_to_drop)
            
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())
            elif df[col].dtype == 'object':
                try:
                    converted = pd.to_datetime(df[col], errors='coerce')
                    if converted.notnull().mean() > 0.5:
                        df[col] = converted
                        continue
                except:
                    pass
                
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip()
                    if df[col].isnull().any() or (df[col] == '').any():
                        mode_val = df[col].mode()
                        if len(mode_val) > 0:
                            df[col] = df[col].replace('', mode_val[0]).fillna(mode_val[0])
                            
        return df

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Encoding categorical variables...")
        df = df.copy()
        cols_to_drop = []
        for col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                unique_vals = df[col].nunique()
                if unique_vals < 10:
                    df[col] = df[col].astype('category').cat.codes
                else:
                    logger.info(f"Dropping high-cardinality text column: {col}")
                    cols_to_drop.append(col)
        df = df.drop(columns=cols_to_drop)
        return df

    def profile(self, df: pd.DataFrame) -> dict:
        logger.info("Profiling data...")
        return {
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_pct": (df.isnull().mean() * 100).to_dict(),
            "numeric_cols": df.select_dtypes(include=['int64', 'float64', 'int32', 'int8']).columns.tolist(),
            "categorical_cols": df.select_dtypes(include=['object', 'category', 'int8']).columns.tolist(),
            "date_cols": df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        }

    def run(self, filepath: str) -> tuple[pd.DataFrame, dict]:
        df = self.load_data(filepath)
        df_clean = self.clean(df)
        df_encoded = self.encode(df_clean)
        prof = self.profile(df_encoded)
        # Restore date columns safely
        for col in df_clean.select_dtypes(include=['datetime64[ns]']).columns:
            if col in df_encoded.columns:
                df_encoded[col] = df_clean[col]
        return df_encoded, prof

def run_pipeline(filepath: str):
    p = DataPreprocessor()
    df, prof = p.run(filepath)
    print("Profile:", prof)
    return df, prof

if __name__ == "__main__":
    run_pipeline("C:\\Users\\Heel\\OneDrive\\Desktop\\Insight AI\\insight_ai\\data\\sample_sales_weather.csv")
