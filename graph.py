from __future__ import annotations
from neo4j import GraphDatabase


class GraphDB:
    """Thin Neo4j wrapper for writing entity relationships."""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_relationship(self, node1: str, node2: str, relationship: str):
        def _tx(tx, n1, n2, rel):
            tx.run("MERGE (a:Entity {name:$n1})", n1=n1)
            tx.run("MERGE (b:Entity {name:$n2})", n2=n2)
            tx.run(
                "MATCH (a:Entity {name:$n1}), (b:Entity {name:$n2}) "
                "MERGE (a)-[r:`%s`]->(b)" % rel,
                n1=n1, n2=n2,
            )
        with self.driver.session() as session:
            session.execute_write(_tx, node1, node2, relationship)
