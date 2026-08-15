"""Frozen, track-level entity appearance extraction and similarity utilities."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Protocol

import numpy as np
from PIL import Image

from .config import Settings
from .domain import (
    Detection,
    EntityAppearance,
    FrameRecord,
    RelationshipSource,
    VisualAttributeEvidence,
)

APPEARANCE_COLORS = (
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "pink",
    "purple",
    "brown",
    "gray",
)
SUPPORTED_APPEARANCE_CLASSES = frozenset(
    {
        "person",
        "car",
        "bicycle",
        "motorcycle",
        "backpack",
        "handbag",
        "bottle",
        "cup",
        "chair",
        "laptop",
        "cell phone",
        "book",
        "suitcase",
        "couch",
        "dining table",
    }
)
APPEARANCE_SOURCE = RelationshipSource.VISUAL_ATTRIBUTE_MODEL
SCORE_INTERPRETATION = "relative_prompt_score_not_calibrated_probability"


class AppearanceEncoder(Protocol):
    model_name: str
    model_version: str

    def encode_images(self, images: Sequence[Image.Image]) -> list[list[float]]: ...

    def encode_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


def _normalized(vector: Sequence[float]) -> list[float]:
    values = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError("appearance encoder produced a zero or non-finite embedding")
    return (values / norm).astype(np.float32).tolist()


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("appearance embeddings have incompatible dimensions")
    return float(np.dot(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)))


def parse_appearance_model(spec: str) -> tuple[str, str]:
    value = spec.strip()
    if ":" not in value:
        raise ValueError("APPEARANCE_MODEL must use '<architecture>:<pretrained-tag>'")
    architecture, pretrained = value.rsplit(":", 1)
    if not architecture or not pretrained:
        raise ValueError("APPEARANCE_MODEL architecture and pretrained tag are required")
    return architecture, pretrained


class OpenCLIPAppearanceEncoder:
    """One frozen OpenCLIP image/text space shared by ingestion and search."""

    def __init__(self, model_spec: str, *, device: str, cache_dir: Path) -> None:
        import open_clip  # type: ignore[import-not-found]
        import torch

        architecture, pretrained = parse_appearance_model(model_spec)
        resolved_device = device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            architecture,
            pretrained=pretrained,
            device=resolved_device,
            cache_dir=str(cache_dir),
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(architecture)
        self.device = torch.device(resolved_device)
        self.model_name = f"open_clip/{architecture}"
        self.model_version = f"{architecture}:{pretrained}:frozen:appearance-schema-v1"
        self._lock = RLock()
        self._text_cache: dict[str, list[float]] = {}

    def encode_images(self, images: Sequence[Image.Image]) -> list[list[float]]:
        if not images:
            return []
        import torch

        batch = torch.stack([self.preprocess(image.convert("RGB")) for image in images]).to(
            self.device
        )
        with self._lock, torch.inference_mode():
            values = self.model.encode_image(batch)
            values = torch.nn.functional.normalize(values.float(), dim=-1)
        return values.cpu().tolist()

    def encode_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        import torch

        missing = [text for text in dict.fromkeys(texts) if text not in self._text_cache]
        if missing:
            tokens = self.tokenizer(missing).to(self.device)
            with self._lock, torch.inference_mode():
                values = self.model.encode_text(tokens)
                values = torch.nn.functional.normalize(values.float(), dim=-1)
            for text, vector in zip(missing, values.cpu().tolist(), strict=True):
                self._text_cache[text] = vector
        return [self._text_cache[text] for text in texts]


@lru_cache(maxsize=4)
def _cached_openclip_encoder(
    model_spec: str, device: str, cache_dir: str
) -> OpenCLIPAppearanceEncoder:
    return OpenCLIPAppearanceEncoder(model_spec, device=device, cache_dir=Path(cache_dir))


def configured_appearance_encoder(settings: Settings) -> AppearanceEncoder | None:
    if not settings.enable_appearance_features:
        return None
    return _cached_openclip_encoder(
        settings.appearance_model,
        settings.appearance_device,
        str(settings.appearance_cache_dir),
    )


def appearance_component_status(settings: Settings) -> str:
    if not settings.enable_appearance_features:
        return "disabled"
    if importlib.util.find_spec("open_clip") is None:
        return "unavailable_missing_dependency"
    try:
        parse_appearance_model(settings.appearance_model)
    except ValueError:
        return "unavailable_invalid_model_config"
    return "available_configured_frozen"


def extract_detection_crop(
    image: Image.Image,
    detection: Detection,
    *,
    minimum_size: int = 24,
) -> Image.Image | None:
    """Return a clipped RGB crop or ``None`` for unusably small boxes."""

    box = detection.pixel_bbox
    x1 = max(0, min(image.width, int(math.floor(box.x1))))
    y1 = max(0, min(image.height, int(math.floor(box.y1))))
    x2 = max(0, min(image.width, int(math.ceil(box.x2))))
    y2 = max(0, min(image.height, int(math.ceil(box.y2))))
    if x2 - x1 < minimum_size or y2 - y1 < minimum_size:
        return None
    return image.crop((x1, y1, x2, y2)).convert("RGB")


def _appearance_id(video_id: str, track_id: str) -> str:
    digest = hashlib.sha1(f"{video_id}:{track_id}".encode()).hexdigest()[:20]
    return f"appearance_{digest}"


def attach_cached_appearances(
    detections: Sequence[Detection], appearances: Sequence[EntityAppearance]
) -> list[Detection]:
    by_track = {item.track_id: item.appearance_id for item in appearances}
    return [
        detection.model_copy(
            update={"appearance_id": by_track.get(detection.track_id or detection.detection_id)}
        )
        for detection in detections
    ]


def _relative_choice(
    embedding: Sequence[float],
    prompts: Sequence[str],
    encoder: AppearanceEncoder,
) -> tuple[int, float]:
    text_vectors = encoder.encode_texts(prompts)
    scores = np.asarray(
        [cosine_similarity(embedding, vector) for vector in text_vectors],
        dtype=np.float64,
    )
    scaled = (scores - float(scores.max())) * 100.0
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    index = int(scores.argmax())
    return index, float(probabilities[index])


class TrackAppearanceExtractor:
    """Aggregate high-quality crop embeddings and attributes per short track."""

    def __init__(
        self,
        encoder: AppearanceEncoder,
        *,
        attribute_min_confidence: float = 0.45,
        max_observations: int = 6,
        minimum_detection_confidence: float = 0.35,
        minimum_crop_size: int = 24,
    ) -> None:
        self.encoder = encoder
        self.attribute_min_confidence = max(0.0, min(1.0, attribute_min_confidence))
        self.max_observations = max(1, max_observations)
        self.minimum_detection_confidence = minimum_detection_confidence
        self.minimum_crop_size = minimum_crop_size

    def extract(
        self,
        *,
        video_id: str,
        frames: Sequence[FrameRecord],
        detections: Sequence[Detection],
        existing: Sequence[EntityAppearance] = (),
    ) -> tuple[list[EntityAppearance], list[Detection]]:
        existing_by_track = {
            item.track_id: item
            for item in existing
            if item.video_id == video_id
            and item.appearance_embedding_model == self.encoder.model_name
            and item.appearance_embedding_version == self.encoder.model_version
        }
        grouped: defaultdict[str, list[Detection]] = defaultdict(list)
        for detection in detections:
            class_label = detection.class_label.strip().lower()
            if class_label not in SUPPORTED_APPEARANCE_CLASSES:
                continue
            track_id = detection.track_id or detection.detection_id
            grouped[track_id].append(detection)
        existing_by_track = {
            track_id: item for track_id, item in existing_by_track.items() if track_id in grouped
        }

        frame_paths = {frame.frame_id: Path(frame.path) for frame in frames}
        crop_groups: dict[str, list[Image.Image]] = {}
        for track_id, observations in grouped.items():
            if track_id in existing_by_track:
                continue
            ranked = sorted(
                observations,
                key=lambda item: (
                    -item.confidence
                    * math.sqrt(
                        max(
                            0.0,
                            (item.pixel_bbox.x2 - item.pixel_bbox.x1)
                            * (item.pixel_bbox.y2 - item.pixel_bbox.y1),
                        )
                    ),
                    item.timestamp,
                    item.detection_id,
                ),
            )
            crops: list[Image.Image] = []
            for detection in ranked:
                if detection.confidence < self.minimum_detection_confidence:
                    continue
                frame_path = frame_paths.get(detection.frame_id)
                if frame_path is None or not frame_path.is_file():
                    continue
                with Image.open(frame_path) as frame_image:
                    crop = extract_detection_crop(
                        frame_image.convert("RGB"),
                        detection,
                        minimum_size=self.minimum_crop_size,
                    )
                if crop is not None:
                    crops.append(crop)
                if len(crops) >= self.max_observations:
                    break
            if crops:
                crop_groups[track_id] = crops

        flattened = [crop for crops in crop_groups.values() for crop in crops]
        vectors = iter(self.encoder.encode_images(flattened) if flattened else [])
        appearances = dict(existing_by_track)
        for track_id, crops in crop_groups.items():
            observation_vectors = [next(vectors) for _ in crops]
            mean_vector = np.asarray(observation_vectors, dtype=np.float32).mean(axis=0)
            embedding = _normalized(mean_vector)
            class_label = grouped[track_id][0].class_label.strip().lower()
            appearances[track_id] = EntityAppearance(
                appearance_id=_appearance_id(video_id, track_id),
                video_id=video_id,
                track_id=track_id,
                class_label=class_label,
                embedding=embedding,
                appearance_embedding_model=self.encoder.model_name,
                appearance_embedding_version=self.encoder.model_version,
                observation_count=len(crops),
                attributes=self._attributes(class_label, embedding),
                source_method=APPEARANCE_SOURCE,
            )

        attached = [
            detection.model_copy(
                update={
                    "appearance_id": (
                        appearances[track_id].appearance_id
                        if (track_id := detection.track_id or detection.detection_id) in appearances
                        else None
                    )
                }
            )
            for detection in detections
        ]
        return sorted(appearances.values(), key=lambda item: item.track_id), attached

    def _attributes(
        self, class_label: str, embedding: Sequence[float]
    ) -> dict[str, VisualAttributeEvidence]:
        if class_label == "person":
            return self._person_attributes(embedding)
        return self._object_attributes(class_label, embedding)

    @staticmethod
    def _evidence(value: str | bool, confidence: float) -> VisualAttributeEvidence:
        return VisualAttributeEvidence(
            value=value,
            confidence=confidence,
            source_method=APPEARANCE_SOURCE,
            score_interpretation=SCORE_INTERPRETATION,
        )

    def _person_attributes(self, embedding: Sequence[float]) -> dict[str, VisualAttributeEvidence]:
        attributes: dict[str, VisualAttributeEvidence] = {}
        color_prompts = [
            f"a photo of a person wearing {color} upper clothing" for color in APPEARANCE_COLORS
        ]
        color_index, color_confidence = _relative_choice(embedding, color_prompts, self.encoder)
        color = APPEARANCE_COLORS[color_index]
        if color_confidence >= self.attribute_min_confidence:
            attributes["upper_clothing_color"] = self._evidence(color, color_confidence)
            garment_values = ("shirt", "jacket", "top")
            garment_prompts = [
                f"a photo of a person wearing a {color} {garment}" for garment in garment_values
            ]
            garment_index, garment_confidence = _relative_choice(
                embedding, garment_prompts, self.encoder
            )
            description = f"{color} {garment_values[garment_index]}"
            attributes["upper_clothing_description"] = self._evidence(
                description, min(color_confidence, garment_confidence)
            )

        for name, positive, negative in (
            ("glasses", "a person wearing glasses", "a person without glasses"),
            ("hat", "a person wearing a hat", "a person without a hat"),
            (
                "bag",
                "a person carrying or wearing a bag or backpack",
                "a person without a bag or backpack",
            ),
        ):
            index, confidence = _relative_choice(embedding, (positive, negative), self.encoder)
            # Binary accessory prompts are much easier to over-interpret than
            # the controlled color vocabulary. Require stronger relative
            # evidence and continue to expose the score as non-calibrated.
            if confidence >= max(self.attribute_min_confidence, 0.65):
                attributes[name] = self._evidence(index == 0, confidence)
        return attributes

    def _object_attributes(
        self, class_label: str, embedding: Sequence[float]
    ) -> dict[str, VisualAttributeEvidence]:
        prompts = [f"a photo of a {color} {class_label}" for color in APPEARANCE_COLORS]
        index, confidence = _relative_choice(embedding, prompts, self.encoder)
        if confidence < self.attribute_min_confidence:
            return {}
        return {"color": self._evidence(APPEARANCE_COLORS[index], confidence)}
