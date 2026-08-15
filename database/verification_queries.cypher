// VidQuery live-verification queries.
// Values are always parameters. These statements are source-owned and are not
// assembled from user or model-generated Cypher.

// Parameters:
// {
//   subject: "person",
//   predicate: "near",
//   object: "chair",
//   symmetric: true,
//   video_ids: [],
//   limit: 5
// }
MATCH (v:Video)-[:HAS_SEGMENT]->(s:Segment)
MATCH (s)-[:HAS_RELATIONSHIP]->(e:RelationshipEvidence)
MATCH (e)-[:FROM_ENTITY]->(source:EntityOccurrence)
MATCH (e)-[:TO_ENTITY]->(target:EntityOccurrence)
WHERE (size($video_ids) = 0 OR v.video_id IN $video_ids)
  AND toLower(e.predicate) = $predicate
  AND (
    (toLower(source.class_label) = $subject
     AND toLower(target.class_label) = $object)
    OR ($symmetric
        AND toLower(source.class_label) = $object
        AND toLower(target.class_label) = $subject)
  )
RETURN v.video_id AS video_id,
       v.display_name AS video_title,
       s.segment_id AS segment_id,
       s.start_time AS start_time,
       s.end_time AS end_time,
       {
         relationship_id: e.relationship_id,
         source_id: e.source_id,
         source_class: source.class_label,
         predicate: e.predicate,
         target_id: e.target_id,
         target_class: target.class_label,
         directionality: e.directionality,
         confidence: e.confidence,
         source_method: e.source_method,
         timestamp: e.timestamp
       } AS matched_relationship
ORDER BY s.start_time, s.segment_id
LIMIT $limit;

// Parameters: {video_id: "canonical-video-uuid", at_time: 945.0, limit: 5}
MATCH (v:Video {video_id: $video_id})-[:HAS_SEGMENT]->(s:Segment)
WHERE s.start_time <= $at_time AND $at_time < s.end_time
RETURN v.video_id AS video_id,
       s.segment_id AS segment_id,
       s.start_time AS start_time,
       s.end_time AS end_time,
       s.transcript AS transcript
ORDER BY s.start_time
LIMIT $limit;

// Parameters: {video_ids: ["canonical-video-uuid-1", "canonical-video-uuid-2"]}
MATCH (v:Video)-[:HAS_SEGMENT]->(s:Segment)
WHERE v.video_id IN $video_ids
OPTIONAL MATCH (s)-[:HAS_RELATIONSHIP]->(e:RelationshipEvidence)
RETURN count(DISTINCT v) AS videos,
       count(DISTINCT s) AS segments,
       count(DISTINCT e) AS relationship_evidence;

RETURN 1 AS ok;
