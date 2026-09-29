from math import hypot
from types import SimpleNamespace

import torch

try:
    from torch_geometric.data import Data
except Exception:  # pragma: no cover
    Data = None


def _frame_audio_features(frame: dict) -> list[float]:
    segments = frame.get("audio_segments", [])
    if not isinstance(segments, list) or not segments:
        return [0.0, 0.0, 0.0, 0.0]

    speaker_ids = {
        str(seg.get("speaker", "unknown")).strip().lower()
        for seg in segments
        if isinstance(seg, dict)
    }

    total_duration = 0.0
    total_text_len = 0.0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        total_duration += max(0.0, end - start)
        total_text_len += float(len(str(seg.get("text", "")).strip()))

    return [
        float(len(segments)),
        float(len(speaker_ids)),
        float(total_duration),
        float(total_text_len),
    ]


def _node_features(node: dict, frame_audio_features: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in node.get("bbox", [0, 0, 0, 0])]
    cx, cy = [float(v) for v in node.get("center", [0, 0])]
    conf = float(node.get("confidence", 0.0))
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    return [cx, cy, width, height, area, conf, *frame_audio_features]


def build_sparse_graphs(visual_data: list[dict], near_threshold: float = 200.0):
    """
    Convert scene-graph style node data into sparse graph tensors.

    For each frame graph, edges are added only between nodes whose centers
    are within near_threshold pixels, avoiding dense O(N^2) fully connected graphs.
    """
    graphs = []
    for frame in visual_data:
        nodes = frame.get("nodes", [])
        if not nodes:
            continue

        audio_features = _frame_audio_features(frame)

        x = torch.tensor(
            [_node_features(node, audio_features) for node in nodes],
            dtype=torch.float,
        )

        edge_pairs = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue
                c1 = nodes[i].get("center", [0, 0])
                c2 = nodes[j].get("center", [0, 0])
                dist = hypot(float(c1[0]) - float(c2[0]), float(c1[1]) - float(c2[1]))
                if dist <= near_threshold:
                    edge_pairs.append([i, j])

        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        if Data is not None:
            graph = Data(x=x, edge_index=edge_index)
            graph.video_id = frame.get("video_id")
            graph.timestamp = frame.get("timestamp")
        else:
            graph = SimpleNamespace(
                x=x,
                edge_index=edge_index,
                video_id=frame.get("video_id"),
                timestamp=frame.get("timestamp"),
            )
        graphs.append(graph)

    return graphs
