from __future__ import annotations
from kubernetes import client, config


def deploy_pod(name: str = "ml-model-pod", namespace: str = "default"):
    """Create a minimal Kubernetes Pod to validate containerization."""
    try:
        config.load_kube_config()
    except Exception:
        config.load_incluster_config()

    api = client.CoreV1Api()
    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V1PodSpec(
            containers=[
                client.V1Container(
                    name="model",
                    image="python:3.10-slim",
                    image_pull_policy="IfNotPresent",
                    command=["sleep", "3600"],
                )
            ]
        ),
    )
    return api.create_namespaced_pod(namespace=namespace, body=pod)
