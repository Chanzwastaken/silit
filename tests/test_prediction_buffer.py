"""Tests for the rolling prediction buffer (get_smoothed_prediction).

Predictor.__init__ loads the TensorFlow model, so these tests build the
object without running __init__ and stub out `predict` instead. That keeps
the smoothing logic testable without any trained artefacts on disk.
"""

from collections import deque

import pytest

import config
from predictor import Predictor


@pytest.fixture
def predictor():
    instance = Predictor.__new__(Predictor)
    instance.prediction_buffer = deque(maxlen=config.PREDICTION_BUFFER_SIZE)
    instance.current_word = ''
    instance.last_stable_letter = None
    instance.stable_letter_count = 0
    instance.no_detection_count = 0
    instance.total_predictions = 0
    instance.confident_predictions = 0
    return instance


def stub_predict(sequence):
    """Return a predict() stub that yields (letter, confidence) pairs."""
    pairs = iter(sequence)

    def _predict(landmarks):
        letter, confidence = next(pairs)
        return {
            'letter': letter,
            'confidence': confidence,
            'all_probabilities': None,
            'class_index': 0,
        }

    return _predict


def test_buffer_is_cleared_when_no_hand_is_detected(predictor):
    predictor.predict = lambda landmarks: {
        'letter': 'A',
        'confidence': 0.99,
        'all_probabilities': None,
        'class_index': 0,
    }
    predictor.get_smoothed_prediction(object())
    assert len(predictor.prediction_buffer) == 1

    predictor.predict = lambda landmarks: None

    assert predictor.get_smoothed_prediction(None) is None
    assert len(predictor.prediction_buffer) == 0


def test_raw_prediction_returned_before_buffer_half_full(predictor):
    half = config.PREDICTION_BUFFER_SIZE // 2
    if half < 2:
        pytest.skip('buffer too small to test partial fill')

    predictor.predict = stub_predict([('A', 0.9)] * (half - 1))

    result = None
    for _ in range(half - 1):
        result = predictor.get_smoothed_prediction(object())

    assert result['letter'] == 'A'
    assert result['confidence'] == pytest.approx(0.9)
    assert 'buffer_agreement' not in result


def test_majority_vote_wins_over_minority(predictor):
    size = config.PREDICTION_BUFFER_SIZE
    majority = (2 * size) // 3
    minority = size - majority

    sequence = [('B', 0.8)] * majority + [('A', 0.95)] * minority
    predictor.predict = stub_predict(sequence)

    result = None
    for _ in range(size):
        result = predictor.get_smoothed_prediction(object())

    assert result['letter'] == 'B'
    assert result['buffer_agreement'] == pytest.approx(majority / size)


def test_confidence_averaged_over_winning_letter_only(predictor):
    size = config.PREDICTION_BUFFER_SIZE
    majority = (2 * size) // 3
    minority = size - majority

    sequence = [('B', 0.6)] * majority + [('A', 0.99)] * minority
    predictor.predict = stub_predict(sequence)

    result = None
    for _ in range(size):
        result = predictor.get_smoothed_prediction(object())

    # The 0.99 outliers from 'A' must not drag the reported confidence up.
    assert result['confidence'] == pytest.approx(0.6)
