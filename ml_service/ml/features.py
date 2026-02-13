import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features to the input DataFrame.
    """
    df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
    df["total_bytes"] = df["src_bytes"] + df["dst_bytes"]
    df["log_src_bytes"] = np.log1p(df["src_bytes"])
    df["log_dst_bytes"] = np.log1p(df["dst_bytes"])
    return df

