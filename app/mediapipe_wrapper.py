"""
SILIT - Sign Language Translator
MediaPipe Hand Detection Wrapper

This module wraps MediaPipe Hands for real-time hand detection
and landmark extraction.
"""

import cv2
import numpy as np
import mediapipe as mp

import config


class MediaPipeWrapper:
    """Wrapper for MediaPipe Hands detection."""
    
    def __init__(self):
        """Initialize MediaPipe Hands detector."""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hands detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=config.MEDIAPIPE_STATIC_IMAGE_MODE,
            max_num_hands=config.MEDIAPIPE_MAX_NUM_HANDS,
            min_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )
        
        self.results = None
    
    def process_frame(self, frame):
        """
        Process a frame and detect hands.
        
        Args:
            frame: BGR image from camera
        
        Returns:
            Processing results from MediaPipe
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        self.results = self.hands.process(frame_rgb)
        
        return self.results
    
    def extract_landmarks(self, frame_shape):
        """
        Extract normalized hand landmarks from last processed frame.
        
        Args:
            frame_shape: Shape of the frame (height, width, channels)
        
        Returns:
            numpy array of shape (63,) containing normalized landmarks,
            or None if no hand detected
        """
        if self.results and self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[0]
            
            # Extract x, y, z coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks, dtype=np.float32)
        
        return None
    
    def draw_landmarks(self, frame, show_skeleton=True):
        """
        Draw hand landmarks on frame.
        
        Args:
            frame: BGR image to draw on
            show_skeleton: Whether to draw hand skeleton
        
        Returns:
            Frame with landmarks drawn
        """
        if not show_skeleton or not self.results:
            return frame
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Draw landmarks and connections
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame
    
    def get_hand_bbox(self, frame_shape):
        """
        Get bounding box of detected hand.
        
        Args:
            frame_shape: Shape of the frame (height, width, channels)
        
        Returns:
            Tuple of (x_min, y_min, x_max, y_max) or None
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return None
        
        hand_landmarks = self.results.multi_hand_landmarks[0]
        h, w = frame_shape[:2]
        
        # Get all x and y coordinates
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        # Calculate bounding box with padding
        padding = 20
        x_min = max(0, int(min(x_coords)) - padding)
        y_min = max(0, int(min(y_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_max = min(h, int(max(y_coords)) + padding)
        
        return (x_min, y_min, x_max, y_max)
    
    def is_hand_detected(self):
        """
        Check if a hand is currently detected.
        
        Returns:
            Boolean indicating if hand is detected
        """
        return (self.results is not None and 
                self.results.multi_hand_landmarks is not None and
                len(self.results.multi_hand_landmarks) > 0)
    
    def get_handedness(self):
        """
        Get handedness (left or right) of detected hand.
        
        Returns:
            'Left', 'Right', or None
        """
        if self.results and self.results.multi_handedness:
            handedness = self.results.multi_handedness[0]
            return handedness.classification[0].label
        
        return None
    
    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()


if __name__ == "__main__":
    # Test MediaPipe wrapper
    print("Testing MediaPipe wrapper...")
    print("Press 'q' to quit")
    
    wrapper = MediaPipeWrapper()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Process frame
        wrapper.process_frame(frame)
        
        # Draw landmarks
        frame = wrapper.draw_landmarks(frame)
        
        # Extract landmarks
        landmarks = wrapper.extract_landmarks(frame.shape)
        
        # Display info
        if landmarks is not None:
            cv2.putText(
                frame,
                f"Hand detected - {wrapper.get_handedness()}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"Landmarks: {landmarks.shape}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "No hand detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        
        # Show frame
        cv2.imshow('MediaPipe Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    wrapper.close()
    
    print("Test complete!")
