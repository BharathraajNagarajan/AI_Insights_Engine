from __future__ import annotations
import json
import os
from kafka import KafkaProducer, KafkaConsumer


def get_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def get_consumer(topic: str) -> KafkaConsumer:
    return KafkaConsumer(
        topic,
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id=os.getenv("KAFKA_GROUP_ID", "ai_insights_group"),
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        consumer_timeout_ms=3000,
    )


def send(topic: str, data: dict, producer: KafkaProducer):
    producer.send(topic, data)
    producer.flush()


def consume(topic: str, max_messages: int = 5) -> list:
    consumer = get_consumer(topic)
    out = []
    for i, msg in enumerate(consumer):
        out.append(msg.value)
        if i + 1 >= max_messages:
            break
    consumer.close()
    return out
