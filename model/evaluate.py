"""
SILIT - Sign Language Translator
Model Evaluation and Per-Class Accuracy Report

Loads the trained model, evaluates it on the held-out test split and
writes a per-class report to model/model_output/evaluation_report.md

Usage:
    python model/evaluate.py
    python model/evaluate.py --model model/model_output/best_model.h5
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from dataset_loader import load_and_prepare_data


MODEL_DIR = Path(__file__).resolve().parent / 'model_output'


def load_model(model_path):
    """Load a trained Keras model, with a clear error if it is missing."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f'Model not found: {model_path}\nRun model/train.py first.'
        )
    return tf.keras.models.load_model(model_path)


def top_confusions(cm, label_names, limit=10):
    """Return the most frequent off-diagonal cells, worst first."""
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), label_names[i], label_names[j]))
    pairs.sort(key=lambda p: p[0], reverse=True)
    return pairs[:limit]


def build_report(acc, loss, y_test, y_pred, label_names, cm, probs):
    """Render the evaluation results as a markdown report."""
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        digits=3,
        zero_division=0,
    )

    confidences = np.max(probs, axis=1)
    mean_conf = float(confidences.mean())
    correct = y_pred == y_test
    correct_conf = float(confidences[correct].mean()) if correct.any() else 0.0

    lines = [
        '# SILIT - Evaluation Report',
        '',
        f'- Test samples: **{len(y_test)}**',
        f'- Test accuracy: **{acc * 100:.2f}%**',
        f'- Test loss: **{loss:.4f}**',
        f'- Mean confidence, all predictions: **{mean_conf:.3f}**',
        f'- Mean confidence, correct predictions: **{correct_conf:.3f}**',
        '',
        '## Per-class metrics',
        '',
        '```',
        report.rstrip(),
        '```',
        '',
        '## Most frequent confusions',
        '',
    ]

    confusions = top_confusions(cm, label_names)
    if confusions:
        lines.append('| Count | True | Predicted as |')
        lines.append('| ----- | ---- | ------------ |')
        for count, true_label, pred_label in confusions:
            lines.append(f'| {count} | {true_label} | {pred_label} |')
    else:
        lines.append('No misclassifications on the test split.')

    lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate the trained SILIT model on the test split'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=str(MODEL_DIR / 'best_model.h5'),
        help='Path to the trained model (default: model/model_output/best_model.h5)',
    )
    parser.add_argument(
        '--data',
        type=str,
        default='processed_data',
        help='Path to processed data directory (default: processed_data)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(MODEL_DIR / 'evaluation_report.md'),
        help='Path for the markdown report',
    )
    args = parser.parse_args()

    print('Loading dataset...')
    data = load_and_prepare_data(args.data)
    label_names = data['label_names']
    X_test, y_test = data['X_test'], data['y_test']

    print(f'Loading model from {args.model}...')
    model = load_model(args.model)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(
        y_test, y_pred, labels=list(range(len(label_names)))
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_report(acc, loss, y_test, y_pred, label_names, cm, probs),
        encoding='utf-8',
    )

    print('\n' + '=' * 60)
    print('EVALUATION COMPLETE')
    print('=' * 60)
    print(f'Test accuracy: {acc * 100:.2f}%')
    print(f'Test loss: {loss:.4f}')
    print(f'Report written to: {output_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
