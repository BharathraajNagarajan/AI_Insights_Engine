# AI Insights Engine (Demo Pipeline)

This repo is a **demo pipeline** that stitches together a GPU-accelerated data path, basic ML workflows, and a minimal deployment.  
The goal was to make different systems “talk” to each other while documenting **pitfalls, fixes, and trade-offs** along the way.  

<img width="639" height="638" alt="image" src="https://github.com/user-attachments/assets/3d58b628-ab25-4822-8265-07680b209236" />

---

## 🚀 Features
- **RAPIDS cuDF** → load a sample Parquet dataset and preprocess on GPU (instead of Pandas).  
- **Apache Spark (3.x)** → convert data to a Spark DataFrame, assemble features, scale them.  
- **Transformers (Hugging Face)** → scaffold a BERT sequence classifier (demo only, not tuned).  
- **TensorFlow/Keras** → run a simple CNN (224×224×3 → 10 classes).  
- **Neo4j** → write nodes + relationships (MERGE with parameterized queries).  
- **Kafka** → JSON producer + bounded consumer to simulate streaming.  
- **MLflow** → track a TensorFlow model (params, metrics, artifacts).  
- **Kubernetes** → spin up a minimal Pod (`python:3.10-slim`) to prove containerization.  

👉 Think of this repo as a “playground”: not production-ready, but enough to get everything running end-to-end.  

---

## ⚠️ Lessons Learned / Gotchas
- **cuDF vs Pandas** → Needed to explicitly cast dtypes, otherwise Spark didn’t like the schema.  
- **Kafka** → The consumer blocked on commit until I tuned `max.poll.records` and `auto.offset.reset`.  
- **Neo4j** → MERGE queries fail if you don’t pass parameters correctly; string concat broke on JSON loads.  
- **MLflow** → Had to log the model as a TensorFlow flavor, not generic pyfunc, otherwise reload failed.  
- **K8s** → Pod spec needed `imagePullPolicy: IfNotPresent` to avoid pulling every run.  

These small fixes are where most of the time went. I’m keeping them here so the next person doesn’t trip up the same way.  

---

## 🖥️ Environment Tested
- **OS:** Ubuntu 22.04 LTS  
- **GPU:** RTX 3090 (CUDA 12.2, cuDF 24.02)  
- **Cluster:** Spark 3.5.1 standalone (2 executors, 2 cores each)  
- **Kafka:** local broker (localhost:9092)  
- **Neo4j:** Community 5.15 (localhost:7687)  
- **Kubernetes:** kind cluster (v1.29.0) with local kubeconfig  

---

## 📦 Python Dependencies
```bash
pip install cudf-cu12==24.02.* cupy-cuda12x \
    pyspark==3.5.1 transformers==4.* tensorflow==2.* \
    neo4j==5.* kafka-python==2.* mlflow==2.* kubernetes==29.*
