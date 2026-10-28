"""Flask web app + REST API for the Rock vs Mine classifier.

Local:      python app/app.py            -> http://localhost:5000
Production: gunicorn app.app:app         (see Procfile / Dockerfile)

Routes
------
GET  /                     web interface (responsive, installable PWA)
GET  /api/health           liveness check for the hosting platform
GET  /api/models           metrics + metadata of all four models
GET  /api/samples          held-out test readings for demos
POST /api/predict          {"features": [60 floats], "model": "svm"}
POST /api/predict/batch    {"readings": [[60 floats], ...], "model": "svm"}
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from flask import (Flask, jsonify, render_template, request,  # noqa: E402
                   send_from_directory)

from src import config  # noqa: E402
from src.predict import InvalidInputError, Predictor  # noqa: E402

MAX_BATCH = 500


def _load_predictor() -> Predictor:
    """Load models; train them first if the repo was cloned without artefacts."""
    if not config.MODEL_META_PATH.exists():
        from src.train import main as train_main
        train_main()
    return Predictor()


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["JSON_SORT_KEYS"] = False
    predictor = _load_predictor()

    @app.after_request
    def allow_cross_origin(resp):
        # The API is public so mobile apps / other front-ends can call it.
        if request.path.startswith("/api/"):
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

    @app.get("/")
    def index():
        return render_template("index.html", meta=predictor.meta)

    @app.get("/service-worker.js")
    def service_worker():
        resp = send_from_directory(app.static_folder, "js/service-worker.js")
        resp.headers["Service-Worker-Allowed"] = "/"
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.get("/figures/<path:name>")
    def figures(name):
        return send_from_directory(config.FIGURES_DIR, name)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", model=predictor.best_key)

    @app.get("/api/models")
    def models():
        m = predictor.meta
        return jsonify(best_model=m["best_model"], selection_rule=m["selection_rule"],
                       models=m["models"], data=m["data"], trained_at=m["trained_at"])

    @app.get("/api/samples")
    def samples():
        df = pd.read_csv(config.TEST_PATH) if config.TEST_PATH.exists() else \
            pd.read_csv(config.RAW_DATA_PATH, header=None,
                        names=config.FEATURE_NAMES + [config.TARGET])
        picks = pd.concat([df[df[config.TARGET] == "M"].head(4),
                           df[df[config.TARGET] == "R"].head(4)])
        out = [{"id": f"{row[config.TARGET]}-{i + 1}",
                "true_label": row[config.TARGET],
                "true_class": config.CLASS_NAMES[row[config.TARGET]],
                "features": [round(float(row[c]), 4) for c in config.FEATURE_NAMES]}
               for i, (_, row) in enumerate(picks.iterrows())]
        train = pd.read_csv(config.TRAIN_PATH) if config.TRAIN_PATH.exists() else df
        means = {lbl: [round(float(v), 4) for v in
                       train[train[config.TARGET] == lbl][config.FEATURE_NAMES].mean()]
                 for lbl in config.CLASSES}
        return jsonify(samples=out, class_means=means)

    @app.route("/api/predict", methods=["POST", "OPTIONS"])
    def predict():
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(predictor.predict(body.get("features"), body.get("model")))
        except InvalidInputError as exc:
            return jsonify(error=str(exc)), 400

    @app.route("/api/predict/batch", methods=["POST", "OPTIONS"])
    def predict_batch():
        if request.method == "OPTIONS":
            return "", 204
        body = request.get_json(silent=True) or {}
        rows = body.get("readings")
        if not isinstance(rows, list) or not rows:
            return jsonify(error="Send a non-empty list under 'readings'."), 400
        if len(rows) > MAX_BATCH:
            return jsonify(error=f"Send at most {MAX_BATCH} readings per request."), 400
        results = []
        for i, row in enumerate(rows, 1):
            try:
                r = predictor.predict(row, body.get("model"))
                r.pop("all_models", None)
                results.append({"row": i, **r})
            except InvalidInputError as exc:
                results.append({"row": i, "error": str(exc)})
        return jsonify(results=results)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
