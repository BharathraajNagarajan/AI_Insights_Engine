from __future__ import annotations
import os
from pyspark.sql import SparkSession

from ingestion import load_parquet, preprocess, to_spark
from features import build_feature_pipeline
from models import build_bert_classifier, build_cnn
from graph import GraphDB
from streaming import get_producer, send, consume
from tracking import log_tf_model
from serving import deploy_pod


def main():
    spark = SparkSession.builder \
        .appName("AI_Insights_Engine") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .getOrCreate()

    # 1. GPU ingestion + preprocessing
    df_cu = load_parquet(os.getenv("DATA_PATH", "data.parquet"))
    df_cu = preprocess(df_cu)

    # 2. Spark feature pipeline
    spark_df = to_spark(df_cu, spark)
    build_feature_pipeline(spark_df)

    # 3. Model scaffolds
    _, _ = build_bert_classifier()
    cnn = build_cnn()

    # 4. Neo4j relationship
    graph = GraphDB(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )
    graph.create_relationship("AIModel1", "AIModel2", "DEPENDS_ON")
    graph.close()

    # 5. Kafka round-trip
    topic = os.getenv("KAFKA_TOPIC", "ai_topic")
    producer = get_producer()
    send(topic, {"event": "model_trained", "accuracy": 0.92}, producer)
    messages = consume(topic, max_messages=3)
    print("Kafka messages:", messages)

    # 6. MLflow tracking
    log_tf_model(cnn, "cnn_model_v1", params={"epochs": 10}, metrics={"accuracy": 0.92})

    # 7. Kubernetes pod
    try:
        deploy_pod()
        print("Kubernetes pod created: ml-model-pod")
    except Exception as e:
        print("Kubernetes pod creation failed:", e)


if __name__ == "__main__":
    main()
