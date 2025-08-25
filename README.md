# AI Insights Engine (Demo Pipeline)

End-to-end demo showing a GPU-accelerated data path (RAPIDS cuDF), Spark ML feature prep, NLP (BERT), CV (Keras CNN), a Neo4j graph write, Kafka pub/sub, MLflow tracking, and a Kubernetes pod deploy.

> This repo is a *demo*. It wires systems together with minimal logic to prove connectivity and fix common pitfalls (cuDF Pandas, MLflow model type, Kafka blocking, Neo4j parameterization, and a valid K8s Pod spec).

---

## Features

- **RAPIDS cuDF**: Load & preprocess Parquet on GPU.
- **Apache Spark**: Convert to Spark DataFrame, assemble & scale features.
- **Transformers (Hugging Face)**: BERT sequence classifier scaffold.
- **TensorFlow/Keras**: Simple CNN (224×224×3 → 10 classes).
- **Neo4j**: `MERGE` nodes + relationship with parameters.
- **Kafka**: JSON producer + bounded consumer.
- **MLflow**: Logs a **TensorFlow** model artifact, params, metrics.
- **Kubernetes**: Creates a minimal Pod (`python:3.10-slim`).

---

## Prerequisites

- **CUDA + RAPIDS** (cuDF) compatible environment
- **Apache Spark** (3.x)
- **Kafka** broker reachable at `KAFKA_BOOTSTRAP` (default `localhost:9092`)
- **Neo4j** at `NEO4J_URI` (default `bolt://localhost:7687`)
- **Kubernetes** access: local kubeconfig or in-cluster
- **MLflow** (local file store by default; set `MLFLOW_TRACKING_URI` to use a server)
- Internet for Hugging Face model downloads

Python deps (example):
```bash
pip install cudf-cu12==24.02.* cupy-cuda12x \
    pyspark==3.5.1 transformers==4.* tensorflow==2.* \
    neo4j==5.* kafka-python==2.* mlflow==2.* kubernetes==29.*
