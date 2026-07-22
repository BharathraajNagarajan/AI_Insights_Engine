from __future__ import annotations
from pyspark.ml.feature import VectorAssembler, StandardScaler


def build_feature_pipeline(spark_df):
    """Assemble + StandardScale all non-label columns."""
    cols = [c for c in spark_df.columns if c != "label"]
    assembler = VectorAssembler(
        inputCols=cols, outputCol="features", handleInvalid="keep"
    )
    assembled = assembler.transform(spark_df)
    scaler = StandardScaler(
        inputCol="features", outputCol="scaledFeatures",
        withMean=True, withStd=True
    )
    scaled = scaler.fit(assembled).transform(assembled)
    return scaled.select("scaledFeatures")
