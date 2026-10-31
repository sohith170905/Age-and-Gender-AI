"""In-memory export helpers for prediction metadata and annotated images."""
import csv
import io
import json
from typing import Iterable


def result_rows(boxes: Iterable, predictions: Iterable, emotions: Iterable | None = None):
    emotion_values = list(emotions or [])
    rows = []
    for index, (box, prediction) in enumerate(zip(boxes, predictions), 1):
        emotion = emotion_values[index - 1] if index - 1 < len(emotion_values) else {}
        rows.append({
            "face": index,
            "x": int(box[0]), "y": int(box[1]), "width": int(box[2]), "height": int(box[3]),
            "gender": prediction.get("gender"), "gender_confidence": prediction.get("gender_confidence"),
            "estimated_age": prediction.get("age"), "age_group": prediction.get("age_group"),
            "age_confidence": prediction.get("age_confidence"), "uncertain": bool(prediction.get("uncertain", False)),
            "quality_score": prediction.get("quality_score"), "quality_flags": list(prediction.get("quality_flags", ())),
            "expression": emotion.get("emotion", prediction.get("emotion")),
            "expression_confidence": emotion.get("confidence"),
            "expression_top3": sorted(emotion.get("probabilities", {}).items(), key=lambda item: item[1], reverse=True)[:3] if emotion else [],
            "timing_ms": prediction.get("timing_ms", {}),
        })
    return rows


def csv_bytes(rows):
    rows = list(rows)
    if not rows:
        return b""
    output = io.StringIO(newline="")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def json_bytes(rows):
    return json.dumps(list(rows), ensure_ascii=False, indent=2).encode("utf-8")


def pdf_report(rows, title="Age and gender analysis report"):
    """Create a minimal, one-page PDF using built-in PDF fonts (no disk files)."""
    rows = list(rows)
    def escape(value):
        text = str(value).encode("ascii", "replace").decode("ascii")
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    lines = [title, f"Faces analyzed: {len(rows)}", "Estimates only; not for decisions about people.", ""]
    for row in rows[:24]:
        lines.append(f"Face {row['face']}: {row.get('gender')} ({row.get('gender_confidence')}%), age {row.get('estimated_age')}, group {row.get('age_group')}, uncertain={row.get('uncertain')}")
    stream = "BT\n/F1 11 Tf\n50 790 Td\n14 TL\n" + "\n".join(f"({escape(line)}) Tj T*" for line in lines) + "\nET"
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        f"5 0 obj << /Length {len(stream.encode('ascii'))} >> stream\n{stream}\nendstream endobj\n".encode("ascii"),
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(output)); output.extend(obj)
    xref = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode("ascii"))
    return bytes(output)
