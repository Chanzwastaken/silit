"""
SILIT - Sign Language Translator
Configuration Constants

This module contains all configuration parameters for the application.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Model paths
MODEL_DIR = Path(__file__).parent.parent / 'model'
MODEL_PATH = MODEL_DIR / 'model_output' / 'silit_model.h5'
LABEL_NAMES_PATH = MODEL_DIR / 'model_output' / 'label_names.pkl'
SCALER_PATH = MODEL_DIR / 'processed_data' / 'scaler.pkl'

# ============================================================================
# MEDIAPIPE SETTINGS
# ============================================================================

# Hand detection parameters
MEDIAPIPE_STATIC_IMAGE_MODE = False
MEDIAPIPE_MAX_NUM_HANDS = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.7
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# ============================================================================
# PREDICTION SETTINGS
# ============================================================================

# Prediction smoothing
PREDICTION_BUFFER_SIZE = 10  # Number of frames to average
CONFIDENCE_THRESHOLD = 0.7   # Minimum confidence to display prediction

# Word building
STABLE_FRAMES_REQUIRED = 15  # Frames a letter must be stable to add to word
SPACE_FRAMES_REQUIRED = 30   # Frames of no detection to add space

# ============================================================================
# CAMERA SETTINGS
# ============================================================================

CAMERA_INDEX = 0  # Default camera (0 = primary webcam)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# ============================================================================
# UI SETTINGS
# ============================================================================

# Window settings
WINDOW_NAME = "SILIT - Sign Language Translator"
FLIP_CAMERA = True  # Mirror the camera feed

# Colors (BGR format for OpenCV)
COLOR_PRIMARY = (255, 100, 50)      # Blue
COLOR_SUCCESS = (100, 255, 100)     # Green
COLOR_WARNING = (50, 200, 255)      # Yellow
COLOR_ERROR = (50, 50, 255)         # Red
COLOR_TEXT = (255, 255, 255)        # White
COLOR_BACKGROUND = (40, 40, 40)     # Dark gray
COLOR_SKELETON = (0, 255, 0)        # Green for hand skeleton

# Font settings
FONT_FACE = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LARGE = 2.0
FONT_SCALE_MEDIUM = 1.0
FONT_SCALE_SMALL = 0.6
FONT_THICKNESS_LARGE = 3
FONT_THICKNESS_MEDIUM = 2
FONT_THICKNESS_SMALL = 1

# UI Layout
MARGIN = 20
INFO_PANEL_HEIGHT = 200

# ============================================================================
# TEXT-TO-SPEECH SETTINGS
# ============================================================================

TTS_ENABLED = True  # Enable/disable text-to-speech
TTS_RATE = 150      # Speech rate (words per minute)
TTS_VOLUME = 0.8    # Volume (0.0 to 1.0)

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

SHOW_FPS = True
TARGET_FPS = 30

# ============================================================================
# KEYBOARD CONTROLS
# ============================================================================

KEY_QUIT = ord('q')
KEY_SPACE = ord(' ')
KEY_BACKSPACE = 8
KEY_CLEAR = ord('c')
KEY_SPEAK = ord('s')
KEY_TOGGLE_SKELETON = ord('t')
KEY_ESC = 27
