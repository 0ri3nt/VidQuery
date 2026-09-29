CREATE CONSTRAINT video_id_unique IF NOT EXISTS
FOR (v:Video)
REQUIRE v.video_id IS UNIQUE;

CREATE CONSTRAINT frame_key_unique IF NOT EXISTS
FOR (f:Frame)
REQUIRE (f.video_id, f.timestamp) IS UNIQUE;

CREATE CONSTRAINT scene_key_unique IF NOT EXISTS
FOR (s:Scene)
REQUIRE s.scene_key IS UNIQUE;

CREATE CONSTRAINT object_key_unique IF NOT EXISTS
FOR (o:Object)
REQUIRE o.object_key IS UNIQUE;

CREATE CONSTRAINT person_key_unique IF NOT EXISTS
FOR (p:Person)
REQUIRE p.person_key IS UNIQUE;

CREATE CONSTRAINT concept_name_unique IF NOT EXISTS
FOR (c:Concept)
REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT audio_segment_key_unique IF NOT EXISTS
FOR (a:AudioSegment)
REQUIRE a.segment_key IS UNIQUE;

CREATE INDEX frame_timestamp_idx IF NOT EXISTS
FOR (f:Frame)
ON (f.timestamp);

CREATE INDEX scene_timestamp_idx IF NOT EXISTS
FOR (s:Scene)
ON (s.timestamp);

CREATE INDEX object_class_idx IF NOT EXISTS
FOR (o:Object)
ON (o.class_name);

CREATE INDEX concept_name_idx IF NOT EXISTS
FOR (c:Concept)
ON (c.name);

CREATE INDEX audio_segment_time_idx IF NOT EXISTS
FOR (a:AudioSegment)
ON (a.timestamp);

// Canonical application schema. Legacy labels above remain readable while
// the segment-centered application path uses the constraints below.
CREATE CONSTRAINT segment_id_unique IF NOT EXISTS
FOR (s:Segment)
REQUIRE s.segment_id IS UNIQUE;

CREATE CONSTRAINT detection_id_unique IF NOT EXISTS
FOR (o:EntityOccurrence)
REQUIRE o.detection_id IS UNIQUE;

CREATE CONSTRAINT transcript_segment_id_unique IF NOT EXISTS
FOR (t:TranscriptSegment)
REQUIRE t.transcript_segment_id IS UNIQUE;

CREATE CONSTRAINT speaker_key_unique IF NOT EXISTS
FOR (s:Speaker)
REQUIRE s.speaker_key IS UNIQUE;

CREATE CONSTRAINT action_name_unique IF NOT EXISTS
FOR (a:Action)
REQUIRE a.name IS UNIQUE;

CREATE CONSTRAINT relationship_evidence_id_unique IF NOT EXISTS
FOR (e:RelationshipEvidence)
REQUIRE e.relationship_id IS UNIQUE;

CREATE CONSTRAINT ocr_evidence_id_unique IF NOT EXISTS
FOR (o:OCRText)
REQUIRE o.ocr_id IS UNIQUE;

CREATE CONSTRAINT appearance_id_unique IF NOT EXISTS
FOR (a:EntityAppearance)
REQUIRE a.appearance_id IS UNIQUE;

CREATE INDEX segment_video_time_idx IF NOT EXISTS
FOR (s:Segment)
ON (s.video_id, s.start_time, s.end_time);

CREATE INDEX entity_occurrence_class_idx IF NOT EXISTS
FOR (o:EntityOccurrence)
ON (o.class_label);

CREATE INDEX appearance_class_color_idx IF NOT EXISTS
FOR (a:EntityAppearance)
ON (a.class_label, a.color, a.upper_clothing_color);

CREATE INDEX relationship_predicate_idx IF NOT EXISTS
FOR (e:RelationshipEvidence)
ON (e.predicate);

CREATE INDEX ocr_text_idx IF NOT EXISTS
FOR (o:OCRText)
ON (o.text);
