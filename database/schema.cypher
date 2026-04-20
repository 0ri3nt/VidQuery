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
