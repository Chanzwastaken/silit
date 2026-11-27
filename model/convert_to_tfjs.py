#!/usr/bin/env python3
"""
SILIT - TensorFlow.js conversion.

Converts the trained Keras model into the TF.js format that the browser
build under web/ loads, and writes the companion assets (labels, scaler)
next to it so the web app has everything it needs in one folder.

Usage:
    pip install -r requirements-build.txt
    python model/convert_to_tfjs.py
    python model/convert_to_tfjs.py --quantize uint8
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'model' / 'model_output'
WEB_MODEL_DIR = ROOT / 'web' / 'model'

# uint8 gives the smallest download but drops precision slightly.
# float16 is the safe default for a model this small.
QUANTIZATION = {
    'none': None,
    'float16': {'float16': '*'},
    'uint8': {'uint8': '*'},
}


def convert(model_path, output_dir, quantize):
    """Convert a .h5 Keras model to model.json + weight shards."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f'Model not found: {model_path}\nRun model/train.py first.'
        )

    import tensorflowjs as tfjs

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(str(model_path))
    tfjs.converters.save_keras_model(
        model,
        str(output_dir),
        quantization_dtype_map=QUANTIZATION[quantize],
    )

    shards = sorted(output_dir.glob('group*.bin'))
    total_mb = sum(s.stat().st_size for s in shards) / (1024 * 1024)
    print(f'Converted {model_path.name} -> {output_dir}')
    print(f'  quantization: {quantize}')
    print(f'  {len(shards)} weight shard(s), {total_mb:.2f} MB')


def export_labels(output_dir):
    """Write the class order the browser must index predictions into."""
    config_path = MODEL_DIR / 'training_config.json'
    if not config_path.exists():
        raise FileNotFoundError(
            f'Missing {config_path}; cannot determine label order.'
        )

    raw = json.loads(config_path.read_text(encoding='utf-8'))
    payload = {
        'num_classes': raw['num_classes'],
        'labels': raw['label_names'],
    }
    (output_dir / 'labels.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8'
    )
    print(f'  wrote labels.json ({payload["num_classes"]} classes)')


def export_scaler(output_dir, scaler_path):
    """Write the fitted scaler params so JS normalisation can match Python."""
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        print(
            f'  WARNING: no scaler at {scaler_path}; skipping scaler.json.\n'
            '  The browser will not reproduce the Python input pipeline.'
        )
        return

    with open(scaler_path, 'rb') as fh:
        scaler = pickle.load(fh)

    payload = {
        'mean': [float(v) for v in scaler.mean_],
        'scale': [float(v) for v in scaler.scale_],
    }
    (output_dir / 'scaler.json').write_text(
        json.dumps(payload), encoding='utf-8'
    )
    print(f'  wrote scaler.json ({len(payload["mean"])} features)')


def main():
    parser = argparse.ArgumentParser(
        description='Convert the SILIT model for use in the browser'
    )
    parser.add_argument(
        '--model',
        default=str(MODEL_DIR / 'best_model.h5'),
        help='Trained Keras model (default: model/model_output/best_model.h5)',
    )
    parser.add_argument(
        '--output',
        default=str(WEB_MODEL_DIR),
        help='Output directory (default: web/model)',
    )
    parser.add_argument(
        '--scaler',
        default=str(ROOT / 'model' / 'processed_data' / 'scaler.pkl'),
        help='Fitted scaler (default: model/processed_data/scaler.pkl, '
             'same file app/config.py loads)',
    )
    parser.add_argument(
        '--quantize',
        choices=sorted(QUANTIZATION),
        default='float16',
        help='Weight quantization (default: float16)',
    )
    parser.add_argument(
        '--no-assets',
        action='store_true',
        help='Skip writing labels.json and scaler.json',
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    print('=' * 60)
    print('SILIT - TensorFlow.js conversion')
    print('=' * 60)

    convert(args.model, output_dir, args.quantize)

    if not args.no_assets:
        print('Writing companion assets...')
        export_labels(output_dir)
        export_scaler(output_dir, args.scaler)

    print('=' * 60)
    print('DONE - web/model is ready')
    print('=' * 60)


if __name__ == '__main__':
    main()
