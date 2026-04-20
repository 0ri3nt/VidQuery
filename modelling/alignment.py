def fuse_modalities(scene_graphs: list[dict], audio_data: list[dict]) -> list[dict]:
    """
    Align audio segments to graph frames by timestamp overlap.

    scene_graphs: list of frame graph dicts containing at least timestamp
    audio_data: list of dicts containing start/end/speaker/text
    """
    fused = []
    for graph in scene_graphs:
        ts = float(graph.get("timestamp", 0))
        frame_start = ts
        frame_end = ts + 1.0

        matched_audio = []
        for seg in audio_data:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            overlap = max(0.0, min(frame_end, end) - max(frame_start, start))
            if overlap > 0.0:
                matched_audio.append(seg)

        fused_item = dict(graph)
        fused_item["audio_segments"] = matched_audio
        fused.append(fused_item)

    return fused
