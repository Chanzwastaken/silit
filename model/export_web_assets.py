"""
SILIT - Web Asset Exporter

Converts the trained model's companion artifacts (pickled label names and
scaler) into JSON files the browser build can consume.

The TensorFlow.js build (model/model_output/web_model/) only carries weights;
everything else the web app needs -- the class order and the feature scaling
parameters -- comes from these JSON files.

Usage:
    python model/export_web_assets.py
    python model/export_web_assets.py --output web/assets
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


MODEL_OUTPUT = Path(__file__).parent / "model_output"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "web" / "assets"


def load_label_names(path):
    """Load label names from the pickle, falling back to training_config.json."""
    if path.exists():
        with open(path, "rb") as f:
            return list(pickle.load(f))

    config_path = MODEL_OUTPUT / "training_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return list(json.load(f)["label_names"])

    raise FileNotFoundError(
        f"No label source found. Looked for {path} and {config_path}. "
        "Re-run extract_landmarks.py + train.py to regenerate them."
    )


def export_labels(label_names, output_dir):
    """Write the class order as JSON. Index in `labels` == model output index."""
    payload = {
        "num_classes": len(label_names),
        "labels": label_names,
    }
    target = output_dir / "labels.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {target} ({len(label_names)} classes)")
    return payload


def export_scaler(scaler_path, output_dir):
    """Write StandardScaler parameters as JSON.

    The browser must apply exactly the same transform as app/predictor.py:
        x_norm = (x - mean) / scale
    With scale_ == 0 numpy substitutes 1.0, so we replicate that here.
    """
    if not scaler_path.exists():
        print(f"Skipping scaler export: {scaler_path} not found.")
        print(
            "  The web build needs this file to reproduce the Python results.\n"
            "  Regenerate it by fitting the scaler during preprocessing and "
            "persisting it to model/model_output/scaler.pkl."
        )
        return None

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    scale = np.where(scale == 0, 1.0, scale)

    payload = {
        "type": "standard",
        "n_features": int(mean.shape[0]),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    target = output_dir / "scaler.json"
    target.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(f"Wrote {target} ({payload['n_features']} features)")
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Export SILIT model metadata to JSON for the web build"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Directory to write JSON assets into (default: web/assets)",
    )
    parser.add_argument(
        "--label-names",
        type=str,
        default=str(MODEL_OUTPUT / "label_names.pkl"),
        help="Path to label_names.pkl",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default=str(MODEL_OUTPUT / "scaler.pkl"),
        help="Path to scaler.pkl",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_names = load_label_names(Path(args.label_names))
    export_labels(label_names, output_dir)
    export_scaler(Path(args.scaler), output_dir)


if __name__ == "__main__":
    main()
