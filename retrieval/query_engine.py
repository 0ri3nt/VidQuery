"""Compatibility wrappers for the safe canonical retrieval implementation.

Raw LLM-generated Cypher execution was intentionally removed.  Natural
language is now parsed into an allowlisted QueryPlan and any Neo4j query is
compiled from one static, parameterized template.
"""

from vidquery.domain import SearchRequest
from vidquery.search import (
    LocalHybridSearchEngine,
    SafeCypherCompiler,
    StructuredQueryParser,
)
from vidquery.storage import SQLiteRepository


class SemanticSearchEngine:
    def __init__(self, repository: SQLiteRepository):
        self.engine = LocalHybridSearchEngine(repository)
        self.parser = StructuredQueryParser()
        self.compiler = SafeCypherCompiler()

    def parse_nl_to_plan(self, user_query: str):
        return self.parser.parse(user_query)

    def compile_safe_cypher(
        self, user_query: str, video_ids: list[str] | None = None, limit: int = 10
    ):
        return self.compiler.compile(
            self.parser.parse(user_query), video_ids or [], limit
        )

    def search(
        self, user_query: str, video_ids: list[str] | None = None, limit: int = 10
    ) -> dict:
        response = self.engine.search(
            SearchRequest(query=user_query, video_ids=video_ids or [], limit=limit)
        )
        return response.model_dump(mode="json")
