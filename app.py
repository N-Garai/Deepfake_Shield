"""
DeepFake Shield API Server v3.0.0
Proper Flask backend with fixed detection algorithms
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from model import DeepFakeDetector
from utils import ImageValidator, ReportGenerator
from PIL import Image
import io
import traceback
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

detector = DeepFakeDetector()
validator = ImageValidator()
report_generator = ReportGenerator()

MAX_IMAGE_SIZE = 15 * 1024 * 1024  # 15 MB


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "model_version": detector.model_version,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "features": {
                "single_analysis": True,
                "batch_analysis": True,
                "reports": True,
            },
        }
    )


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    try:
        start = datetime.utcnow()

        if request.is_json:
            data = request.get_json(silent=True) or {}
            image_b64 = data.get("image")
            if not image_b64:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "No image field",
                            "message": "Provide 'image' base64 in JSON body.",
                        }
                    ),
                    400,
                )

            ok, msg = validator.validate_base64(image_b64, max_size_bytes=MAX_IMAGE_SIZE)
            if not ok:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Invalid image data",
                            "message": msg,
                        }
                    ),
                    400,
                )

            results = detector.analyze_image(image_b64)

        elif "file" in request.files:
            file = request.files["file"]
            if not file or file.filename == "":
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "No file provided",
                            "message": "Upload an image file.",
                        }
                    ),
                    400,
                )

            ok, msg = validator.validate_file(file, max_size_bytes=MAX_IMAGE_SIZE)
            if not ok:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Invalid file",
                            "message": msg,
                        }
                    ),
                    400,
                )

            file.stream.seek(0)
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            results = detector.analyze_image(image)

        else:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Bad request",
                        "message": "Use JSON base64 or multipart 'file'.",
                    }
                ),
                400,
            )

        elapsed = (datetime.utcnow() - start).total_seconds()
        results["processing_time"] = round(elapsed, 3)

        logger.info(
            f"Analysis - score={results['authenticity_score']:.1f} classification={results['verdict']['classification']}"
        )

        return jsonify({"success": True, "results": results})

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.debug(traceback.format_exc())
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Analysis failed",
                    "message": str(e),
                }
            ),
            500,
        )


@app.route("/api/batch-analyze", methods=["POST"])
def batch_analyze():
    try:
        data = request.get_json(silent=True) or {}
        images = data.get("images", [])
        if not isinstance(images, list) or not images:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Invalid payload",
                        "message": "Provide 'images' as a non-empty list.",
                    }
                ),
                400,
            )

        if len(images) > 10:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Too many images",
                        "message": "Maximum 10 images per request.",
                    }
                ),
                400,
            )

        batch_results = []
        for idx, img_b64 in enumerate(images):
            try:
                ok, msg = validator.validate_base64(
                    img_b64, max_size_bytes=MAX_IMAGE_SIZE
                )
                if not ok:
                    batch_results.append(
                        {
                            "index": idx,
                            "success": False,
                            "error": msg,
                        }
                    )
                    continue

                r = detector.analyze_image(img_b64)
                batch_results.append(
                    {
                        "index": idx,
                        "success": True,
                        "results": r,
                    }
                )
            except Exception as ex:
                batch_results.append(
                    {
                        "index": idx,
                        "success": False,
                        "error": str(ex),
                    }
                )

        return jsonify(
            {
                "success": True,
                "total": len(images),
                "results": batch_results,
            }
        )

    except Exception as e:
        logger.error(f"Batch error: {e}")
        logger.debug(traceback.format_exc())
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Batch analysis failed",
                    "message": str(e),
                }
            ),
            500,
        )


@app.route("/api/generate-report", methods=["POST"])
def generate_report():
    data = request.get_json(silent=True) or {}
    results = data.get("results")
    if not results:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "No results",
                    "message": "Provide 'results' object.",
                }
            ),
            400,
        )

    report_text = report_generator.generate_text_report(results)
    return jsonify({"success": True, "report": report_text})


if __name__ == "__main__":
    print("=" * 60)
    print("🛡️  DeepFake Shield v3.0.0")
    print("=" * 60)
    print("Running on http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
