import re

from core.config import get_settings
from database.neo4j_client import Neo4jClient
from retrieval.prompt_templates import build_prompt


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def _sec_to_hhmmss(seconds: int | float) -> str:
    total = int(float(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class SemanticSearchEngine:
    def __init__(self, neo4j_client: Neo4jClient, groq_api_key: str | None = None, model: str | None = None):
        settings = get_settings()
        self.neo4j = neo4j_client
        self.groq_api_key = groq_api_key or settings.groq_api_key
        self.model = model or settings.groq_model

    def parse_nl_to_cypher(self, user_query: str) -> str:
        if not self.groq_api_key:
            return (
                "MATCH (v:Video)-[:HAS_FRAME]->(f:Frame) "
                "RETURN v.video_id AS video_id, f.timestamp AS timestamp "
                "ORDER BY f.timestamp ASC LIMIT 50"
            )

        try:
            from groq import Groq

            client = Groq(api_key=self.groq_api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=build_prompt(user_query),
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            return _strip_code_fences(content)
        except Exception:
            return (
                "MATCH (v:Video)-[:HAS_FRAME]->(f:Frame) "
                "RETURN v.video_id AS video_id, f.timestamp AS timestamp "
                "ORDER BY f.timestamp ASC LIMIT 50"
            )

    def execute_search(self, cypher_query: str) -> list[dict]:
        return self.neo4j.run_query(cypher_query)

    def format_results(self, raw_results: list[dict]) -> list[dict]:
        formatted = []
        for row in raw_results:
            ts = row.get("timestamp", 0)
            formatted.append(
                {
                    "video_id": row.get("video_id", "unknown"),
                    "timestamp": int(ts) if str(ts).isdigit() else ts,
                    "timecode": _sec_to_hhmmss(ts if isinstance(ts, (int, float)) else 0),
                }
            )
        return formatted

    def search(self, user_query: str) -> dict:
        cypher = self.parse_nl_to_cypher(user_query)
        raw = self.execute_search(cypher)
        return {
            "query": user_query,
            "cypher": cypher,
            "results": self.format_results(raw),
        }
