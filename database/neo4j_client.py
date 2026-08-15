from collections.abc import Callable
from typing import Any

from neo4j import GraphDatabase


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str, verify_connectivity: bool = True):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        if verify_connectivity:
            try:
                self.driver.verify_connectivity()
            except Exception as exc:
                self.close()
                raise ConnectionError(f"Could not connect to Neo4j at {uri}: {exc}") from exc

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()

    def execute_write(self, fn: Callable, *args, **kwargs) -> Any:
        with self.driver.session() as session:
            return session.execute_write(fn, *args, **kwargs)

    def execute_read(self, fn: Callable, *args, **kwargs) -> Any:
        with self.driver.session() as session:
            return session.execute_read(fn, *args, **kwargs)

    def run_query(self, query: str, params: dict | None = None) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def health_check(self) -> bool:
        result = self.run_query("RETURN 1 AS ok")
        return bool(result and result[0].get("ok") == 1)
