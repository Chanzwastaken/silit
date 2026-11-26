"""Tests for the stability tracking and word building logic (update_word)."""

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


def pred(letter, confidence=0.99):
    return {'letter': letter, 'confidence': confidence, 'buffer_agreement': 1.0}


def feed(predictor, letter, frames, confidence=0.99):
    """Feed `frames` identical predictions and return the last word state."""
    state = None
    for _ in range(frames):
        state = predictor.update_word(pred(letter, confidence))
    return state


def test_low_confidence_does_not_change_word(predictor):
    state = feed(
        predictor,
        'A',
        config.STABLE_FRAMES_REQUIRED,
        confidence=config.CONFIDENCE_THRESHOLD - 0.01,
    )

    assert predictor.get_word() == ''
    assert state['action'] == 'low_confidence'


def test_letter_added_only_after_required_stable_frames(predictor):
    frames = config.STABLE_FRAMES_REQUIRED
    if frames < 2:
        pytest.skip('stability window too small to test partial tracking')

    feed(predictor, 'A', frames - 1)
    assert predictor.get_word() == ''

    state = feed(predictor, 'A', 1)
    assert predictor.get_word() == 'A'
    assert state['action'] == 'added'


def test_duplicate_letter_is_ignored(predictor):
    predictor.current_word = 'A'

    state = feed(predictor, 'A', config.STABLE_FRAMES_REQUIRED)

    assert predictor.get_word() == 'A'
    assert state['action'] == 'duplicate_ignored'


def test_space_gesture_appends_one_space(predictor):
    predictor.current_word = 'AB'

    state = feed(predictor, 'space', config.STABLE_FRAMES_REQUIRED)

    assert predictor.get_word() == 'AB '
    assert state['action'] == 'space_added'


def test_space_gesture_ignored_when_word_already_ends_with_space(predictor):
    predictor.current_word = 'AB '

    state = feed(predictor, 'space', config.STABLE_FRAMES_REQUIRED)

    assert predictor.get_word() == 'AB '
    assert state['action'] == 'space_ignored'


def test_delete_gesture_removes_last_character(predictor):
    predictor.current_word = 'ABC'

    state = feed(predictor, 'del', config.STABLE_FRAMES_REQUIRED)

    assert predictor.get_word() == 'AB'
    assert state['action'] == 'deleted'


def test_delete_gesture_ignored_on_empty_word(predictor):
    state = feed(predictor, 'del', config.STABLE_FRAMES_REQUIRED)

    assert predictor.get_word() == ''
    assert state['action'] == 'delete_ignored'


def test_nothing_gesture_leaves_word_untouched(predictor):
    predictor.current_word = 'AB'

    state = feed(predictor, 'nothing', config.STABLE_FRAMES_REQUIRED)

    assert predictor.get_word() == 'AB'
    assert state['action'] == 'nothing'


def test_no_hand_resets_stability_tracking(predictor):
    predictor.stable_letter_count = 3
    predictor.last_stable_letter = 'A'

    state = predictor.update_word(None)

    assert state['action'] == 'no_detection'
    assert predictor.stable_letter_count == 0
    assert predictor.last_stable_letter is None


def test_counter_resets_after_action_so_letter_is_not_repeated(predictor):
    """Holding one sign must add the letter once, not every frame."""
    frames = config.STABLE_FRAMES_REQUIRED

    feed(predictor, 'A', frames)
    assert predictor.get_word() == 'A'

    feed(predictor, 'A', frames)
    assert predictor.get_word() == 'A'
    assert predictor.stable_letter_count == 0
