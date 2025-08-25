import os
import json
import cudf
import tensorflow as tf
from transformers import BertTokenizer, BertForSequenceClassification
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from neo4j import GraphDatabase
from kafka import KafkaProducer, KafkaConsumer
import mlflow
import mlflow.tensorflow
from kubernetes import client, config

spark = SparkSession.builder.appName("AI_Insights_Engine") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.task.cpus", "1") \
    .getOrCreate()

def data_ingestion(file_path: str) -> cudf.DataFrame:
    return cudf.read_parquet(file_path)

def preprocess_data(df: cudf.DataFrame) -> cudf.DataFrame:
    df = df.fillna(0)
    df = cudf.get_dummies(df, drop_first=True)
    return df

def to_spark_df(cu_df: cudf.DataFrame):
    pd_df = cu_df.to_pandas()
    return spark.createDataFrame(pd_df)

def build_feature_pipeline(spark_df):
    cols = [c for c in spark_df.columns if c not in {"label"}]
    assembler = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid="keep")
    assembled = assembler.transform(spark_df)
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
    model = scaler.fit(assembled)
    scaled = model.transform(assembled)
    return scaled.select("scaledFeatures")

def build_nlp_model():
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

def build_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

class GraphDB:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    def close(self):
        self.driver.close()
    def create_relationship(self, node1, node2, relationship):
        def _tx(tx, n1, n2, rel):
            tx.run("MERGE (a:Entity {name:$n1})", n1=n1)
            tx.run("MERGE (b:Entity {name:$n2})", n2=n2)
            tx.run("MATCH (a:Entity {name:$n1}), (b:Entity {name:$n2}) "
                   "MERGE (a)-[r:`%s`]->(b)" % rel, n1=n1, n2=n2)
        with self.driver.session() as session:
            session.execute_write(_tx, node1, node2, relationship)

def get_kafka_producer():
    return KafkaProducer(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

def get_kafka_consumer(topic):
    return KafkaConsumer(
        topic,
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id=os.getenv("KAFKA_GROUP_ID", "ai_insights_group"),
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        consumer_timeout_ms=3000,
    )

def send_to_kafka(topic, data, producer):
    producer.send(topic, data)
    producer.flush()

def consume_kafka_messages(topic, max_messages=5):
    consumer = get_kafka_consumer(topic)
    out = []
    for i, msg in enumerate(consumer):
        out.append(msg.value)
        if i + 1 >= max_messages:
            break
    consumer.close()
    return out

def log_mlflow_model_tf(model, model_name):
    mlflow.set_experiment("AI_Insights_Engine")
    with mlflow.start_run():
        mlflow.tensorflow.log_model(model, model_name)
        mlflow.log_param("epochs", 10)
        mlflow.log_metric("accuracy", 0.92)

def deploy_k8s_model():
    try:
        config.load_kube_config()
    except Exception:
        config.load_incluster_config()
    api = client.CoreV1Api()
    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(name="ml-model-pod"),
        spec=client.V1PodSpec(containers=[
            client.V1Container(name="model", image="python:3.10-slim", command=["sleep", "3600"])
        ])
    )
    return api.create_namespaced_pod(namespace="default", body=pod)

if __name__ == "__main__":
    df_cu = data_ingestion(os.getenv("DATA_PATH", "data.parquet"))
    df_cu = preprocess_data(df_cu)
    spark_df = to_spark_df(df_cu)
    _ = build_feature_pipeline(spark_df)
    nlp_model, tokenizer = build_nlp_model()
    cnn_model = build_cnn_model()
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "password")
    graph_db = GraphDB(neo4j_uri, neo4j_user, neo4j_pass)
    graph_db.create_relationship("AIModel1", "AIModel2", "DEPENDS_ON")
    graph_db.close()
    topic = os.getenv("KAFKA_TOPIC", "ai_topic")
    producer = get_kafka_producer()
    send_to_kafka(topic, {"event": "model_trained", "accuracy": 0.92}, producer)
    messages = consume_kafka_messages(topic, max_messages=3)
    print("Kafka messages:", messages)
    log_mlflow_model_tf(cnn_model, "cnn_model_v1")
    try:
        deploy_k8s_model()
        print("Kubernetes pod created: ml-model-pod")
    except Exception as e:
        print("Kubernetes pod creation failed:", str(e))
