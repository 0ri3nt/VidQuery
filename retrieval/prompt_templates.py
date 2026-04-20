SYSTEM_PROMPT = """
You are a Cypher generator for VidQuery.
Return only one valid Cypher query and no additional prose.

Graph schema summary:
- (:Video {video_id})
- (:Frame {video_id, timestamp, frame_file})
- (:Person {person_key, class_name, confidence, node_id, video_id, timestamp})
- (:Object {object_key, class_name, confidence, node_id, video_id, timestamp})

Relationships:
- (:Video)-[:HAS_FRAME]->(:Frame)
- (:Frame)-[:HAS_PERSON]->(:Person)
- (:Frame)-[:HAS_OBJECT]->(:Object)
- Spatial/semantic relations between Person/Object nodes:
  OVERLAPS_WITH, TOUCHES, NEAR, LEFT_OF, RIGHT_OF,
  ABOVE, BELOW, CONTAINS, INSIDE, TALKS_TO, INTERACTS_WITH,
  SITTING, STANDING, LYING, KISSING, HUGGING, TALKING_TO,
  LOOKING_LEFT_AT, LOOKING_RIGHT_AT

Rules:
1) Return timestamp and video_id whenever possible.
2) Use case-insensitive matching for user terms.
3) Prefer MATCH + WHERE + RETURN + ORDER BY.
4) LIMIT 50 unless explicitly asked for all.
""".strip()


FEW_SHOT_EXAMPLES = [
    {
        "user": "Find where a person is standing",
        "cypher": (
            "MATCH (v:Video)-[:HAS_FRAME]->(f:Frame)-[:HAS_PERSON]->(p:Person) "
            "MATCH (p)-[:STANDING]->(p) "
            "RETURN v.video_id AS video_id, f.timestamp AS timestamp "
            "ORDER BY f.timestamp ASC LIMIT 50"
        ),
    },
    {
        "user": "Find scenes where person is interacting with laptop",
        "cypher": (
            "MATCH (v:Video)-[:HAS_FRAME]->(f:Frame)-[:HAS_PERSON]->(p:Person) "
            "MATCH (f)-[:HAS_OBJECT]->(o:Object) "
            "MATCH (p)-[:INTERACTS_WITH]->(o) "
            "WHERE toLower(o.class_name) CONTAINS 'laptop' "
            "RETURN v.video_id AS video_id, f.timestamp AS timestamp "
            "ORDER BY f.timestamp ASC LIMIT 50"
        ),
    },
]


def build_prompt(query: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["cypher"]})
    messages.append({"role": "user", "content": query})
    return messages
