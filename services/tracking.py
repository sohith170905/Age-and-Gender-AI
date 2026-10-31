from dataclasses import dataclass, field
from collections import deque
from typing import Any


def iou(a, b):
    ax, ay, aw, ah = map(float, a); bx, by, bw, bh = map(float, b)
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class _Track:
    box: tuple[int, int, int, int]
    age: float | None = None
    genders: deque = field(default_factory=lambda: deque(maxlen=5))
    emotions: deque = field(default_factory=lambda: deque(maxlen=5))


class TemporalFaceTracker:
    """Greedy IoU tracker with short-window smoothing; holds no frames or images."""
    def __init__(self, threshold: float = 0.25, window: int = 5, age_alpha: float = 0.35):
        self.threshold = threshold
        self.window = window
        self.age_alpha = age_alpha
        self._tracks: list[_Track] = []

    def update(self, boxes: list[tuple], predictions: list[dict[str, Any]]):
        tracks = self._tracks
        used: set[int] = set()
        next_tracks: list[_Track] = []
        output = []
        for box, prediction in zip(boxes, predictions):
            match_index, score = None, self.threshold
            for index, track in enumerate(tracks):
                if index in used:
                    continue
                candidate = iou(box, track.box)
                if candidate > score:
                    match_index, score = index, candidate
            track = tracks[match_index] if match_index is not None else _Track(tuple(map(int, box)))
            if match_index is not None:
                used.add(match_index)
            track.box = tuple(map(int, box))
            age = float(prediction.get("age", 0.0))
            track.age = age if track.age is None else self.age_alpha * age + (1 - self.age_alpha) * track.age
            gender = prediction.get("gender")
            if gender:
                track.genders.append(gender)
            emotion = prediction.get("emotion")
            if emotion:
                track.emotions.append(emotion)
            smoothed = dict(prediction)
            smoothed["age"] = int(round(track.age))
            if track.genders:
                gender_counts = {value: track.genders.count(value) for value in set(track.genders)}
                gender_max = max(gender_counts.values())
                smoothed["gender"] = next(value for value in reversed(track.genders) if gender_counts[value] == gender_max)
            if track.emotions:
                emotion_counts = {value: track.emotions.count(value) for value in set(track.emotions)}
                emotion_max = max(emotion_counts.values())
                smoothed["emotion"] = next(value for value in reversed(track.emotions) if emotion_counts[value] == emotion_max)
            next_tracks.append(track)
            output.append(smoothed)
        self._tracks = next_tracks
        return output
