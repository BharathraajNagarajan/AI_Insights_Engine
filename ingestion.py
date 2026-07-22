from __future__ import annotations
import cudf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler


def load_parquet(file_path: str) -> cudf.DataFrame:
    """Load a Parquet file into a cuDF GPU DataFrame."""
    return cudf.read_parquet(file_path)


def preprocess(df: cudf.DataFrame) -> cudf.DataFrame:
    """Fill nulls and one-hot encode categorical columns on GPU."""
    df = df.fillna(0)
    df = cudf.get_dummies(df, drop_first=True)
    return df


def to_spark(df: cudf.DataFrame, spark: SparkSession):
    """Transfer cuDF DataFrame to Spark via Pandas bridge."""
    return spark.createDataFrame(df.to_pandas())
