"""
Utility helpers - validation and report generation
"""

import base64
import imghdr
from io import BytesIO
from PIL import Image
from datetime import datetime


class ImageValidator:
    def __init__(self, allowed_ext=None):
        self.allowed_ext = allowed_ext or {"jpg", "jpeg", "png", "webp"}

    def validate_file(self, file_storage, max_size_bytes=15 * 1024 * 1024):
        file_storage.stream.seek(0, 2)
        size = file_storage.tell()
        file_storage.stream.seek(0)

        if size <= 0:
            return False, "Empty file."
        if size > max_size_bytes:
            return False, f"File too large (>{max_size_bytes // (1024*1024)} MB)."

        filename = file_storage.filename or ""
        if "." not in filename:
            return False, "File must have an extension."

        ext = filename.rsplit(".", 1)[-1].lower()
        if ext not in self.allowed_ext:
            return (
                False,
                f"Unsupported type '{ext}'. Allowed: {', '.join(sorted(self.allowed_ext))}",
            )

        head = file_storage.read(512)
        file_storage.stream.seek(0)
        kind = imghdr.what(None, head)
        if kind not in {"jpeg", "png", "webp"}:
            return False, "Not a valid image file."

        try:
            img = Image.open(file_storage.stream)
            img.verify()
            file_storage.stream.seek(0)
        except Exception:
            return False, "Corrupted or unsupported image."

        return True, "OK"

    def validate_base64(self, b64_str, max_size_bytes=15 * 1024 * 1024):
        if not isinstance(b64_str, str):
            return False, "Image must be base64-encoded string."

        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]

        try:
            raw = base64.b64decode(b64_str, validate=True)
        except Exception:
            return False, "Invalid base64 data."

        if len(raw) == 0:
            return False, "Empty image data."
        if len(raw) > max_size_bytes:
            return False, f"Image too large (>{max_size_bytes // (1024*1024)} MB)."

        kind = imghdr.what(None, raw[:512])
        if kind not in {"jpeg", "png", "webp"}:
            return False, "Unsupported image type."

        try:
            img = Image.open(BytesIO(raw))
            img.verify()
        except Exception:
            return False, "Corrupted or unsupported image."

        return True, "OK"


class ReportGenerator:
    def generate_text_report(self, results: dict) -> str:
        score = results.get("authenticity_score", 0)
        td = results.get("technical_details", {})
        v = results.get("verdict", {})
        mv = results.get("model_version", "unknown")
        pt = results.get("processing_time", None)

        lines = []
        lines.append("=" * 50)
        lines.append("DEEPFAKE SHIELD - FORENSIC ANALYSIS REPORT")
        lines.append("=" * 50)
        lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
        lines.append(f"Model: DeepFake Shield v{mv}")
        lines.append("")

        lines.append("OVERALL RESULT")
        lines.append("-" * 50)
        lines.append(f"Authenticity Score: {score:.1f} / 100")
        lines.append(f"Classification:     {v.get('classification','N/A').upper()}")
        lines.append(f"Title:              {v.get('title','N/A')}")
        lines.append(f"Confidence:         {v.get('confidence','N/A').upper()}")
        if pt is not None:
            lines.append(f"Processing Time:    {pt:.3f} seconds")
        lines.append("")

        lines.append("TECHNICAL BREAKDOWN")
        lines.append("-" * 50)
        lines.append(f"Compression Artifacts: {td.get('compression_artifacts', 'N/A')}/100")
        lines.append(f"Edge Consistency:      {td.get('edge_consistency', 'N/A')}/100")
        lines.append(f"Noise Patterns:        {td.get('noise_patterns', 'N/A')}/100")
        lines.append(f"Color Distribution:    {td.get('color_distribution', 'N/A')}/100")
        lines.append(f"Frequency Analysis:    {td.get('frequency_analysis', 'N/A')}/100")
        lines.append("")

        lines.append("INTERPRETATION")
        lines.append("-" * 50)
        lines.append(v.get("description", "No description available."))
        lines.append("")

        lines.append("SCORE RANGES")
        lines.append("-" * 50)
        lines.append("72-100: Likely Authentic")
        lines.append("58-71:  Uncertain / Mixed Indicators")
        lines.append("40-57:  Suspicious / Likely Manipulated")
        lines.append("0-39:   Likely AI-Generated")
        lines.append("")

        lines.append("DISCLAIMER")
        lines.append("-" * 50)
        lines.append("This tool uses heuristic forensic analysis and may")
        lines.append("produce false positives or negatives. Do NOT use as")
        lines.append("the sole basis for critical decisions. Professional")
        lines.append("forensic experts should verify important findings.")

        return "\n".join(lines)
