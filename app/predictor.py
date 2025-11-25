"""
SILIT - Sign Language Translator
Prediction Engine

This module handles model inference, prediction smoothing,
and word building logic.
"""

import numpy as np
import pickle
from collections import deque
from pathlib import Path
import tensorflow as tf

import config


class Predictor:
    """Real-time sign language prediction engine."""
    
    def __init__(self, model_path=None, label_names_path=None, scaler_path=None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to trained model (.h5 file)
            label_names_path: Path to label names pickle file
            scaler_path: Path to fitted scaler pickle file
        """
        # Use default paths if not provided
        if model_path is None:
            model_path = config.MODEL_PATH
        if label_names_path is None:
            label_names_path = config.LABEL_NAMES_PATH
        if scaler_path is None:
            scaler_path = config.SCALER_PATH
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Load label names
        print(f"Loading label names from: {label_names_path}")
        with open(label_names_path, 'rb') as f:
            self.label_names = pickle.load(f)
        print(f"Loaded {len(self.label_names)} classes: {self.label_names}")
        
        # Load scaler
        print(f"Loading scaler from: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("Scaler loaded successfully!")
        
        # Prediction smoothing buffer
        self.prediction_buffer = deque(maxlen=config.PREDICTION_BUFFER_SIZE)
        
        # Word building state
        self.current_word = ""
        self.last_stable_letter = None
        self.stable_letter_count = 0
        self.no_detection_count = 0
        
        # Statistics
        self.total_predictions = 0
        self.confident_predictions = 0
    
    def predict(self, landmarks):
        """
        Predict sign language letter from landmarks.
        
        Args:
            landmarks: numpy array of shape (63,) containing hand landmarks
        
        Returns:
            Dictionary containing:
                - letter: predicted letter
                - confidence: prediction confidence
                - all_probabilities: probabilities for all classes
        """
        if landmarks is None:
            return None
        
        # Normalize landmarks using fitted scaler
        landmarks_normalized = self.scaler.transform(landmarks.reshape(1, -1))
        
        # Get prediction
        probabilities = self.model.predict(landmarks_normalized, verbose=0)[0]
        
        # Get predicted class and confidence
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        letter = self.label_names[predicted_class]
        
        # Update statistics
        self.total_predictions += 1
        if confidence >= config.CONFIDENCE_THRESHOLD:
            self.confident_predictions += 1
        
        return {
            'letter': letter,
            'confidence': float(confidence),
            'all_probabilities': probabilities,
            'class_index': predicted_class
        }
    
    def get_smoothed_prediction(self, landmarks):
        """
        Get smoothed prediction using rolling buffer.
        
        Args:
            landmarks: numpy array of hand landmarks
        
        Returns:
            Dictionary with smoothed prediction or None
        """
        # Get raw prediction
        prediction = self.predict(landmarks)
        
        if prediction is None:
            self.prediction_buffer.clear()
            return None
        
        # Add to buffer
        self.prediction_buffer.append(prediction)
        
        # Need enough predictions for smoothing
        if len(self.prediction_buffer) < config.PREDICTION_BUFFER_SIZE // 2:
            return prediction
        
        # Get most common prediction from buffer
        letter_counts = {}
        total_confidence = {}
        
        for pred in self.prediction_buffer:
            letter = pred['letter']
            letter_counts[letter] = letter_counts.get(letter, 0) + 1
            total_confidence[letter] = total_confidence.get(letter, 0) + pred['confidence']
        
        # Find most common letter
        most_common_letter = max(letter_counts, key=letter_counts.get)
        avg_confidence = total_confidence[most_common_letter] / letter_counts[most_common_letter]
        
        return {
            'letter': most_common_letter,
            'confidence': avg_confidence,
            'buffer_agreement': letter_counts[most_common_letter] / len(self.prediction_buffer)
        }
    
    def update_word(self, prediction):
        """
        Update current word based on stable predictions.
        Handles special gestures: 'space', 'del', 'nothing'
        
        Args:
            prediction: Prediction dictionary from get_smoothed_prediction
        
        Returns:
            Dictionary with word building state
        """
        if prediction is None:
            # No hand detected
            self.no_detection_count += 1
            self.stable_letter_count = 0
            self.last_stable_letter = None
            
            # Don't auto-add space - let user use 'space' gesture
            
            return {
                'word': self.current_word,
                'action': 'no_detection',
                'stable_letter': None,
                'stability': 0
            }
        
        # Reset no detection counter
        self.no_detection_count = 0
        
        # Check if prediction is confident enough
        if prediction['confidence'] < config.CONFIDENCE_THRESHOLD:
            return {
                'word': self.current_word,
                'action': 'low_confidence',
                'stable_letter': prediction['letter'],
                'stability': 0
            }
        
        current_letter = prediction['letter']
        
        # Check if letter is stable
        if current_letter == self.last_stable_letter:
            self.stable_letter_count += 1
        else:
            self.last_stable_letter = current_letter
            self.stable_letter_count = 1
        
        # Process stable predictions
        action = 'tracking'
        if self.stable_letter_count >= config.STABLE_FRAMES_REQUIRED:
            # Handle special gestures
            if current_letter.lower() == 'space':
                # Add space
                if self.current_word and not self.current_word.endswith(' '):
                    self.current_word += ' '
                    action = 'space_added'
                else:
                    action = 'space_ignored'
            
            elif current_letter.lower() == 'del':
                # Delete last character
                if self.current_word:
                    self.current_word = self.current_word[:-1]
                    action = 'deleted'
                else:
                    action = 'delete_ignored'
            
            elif current_letter.lower() == 'nothing':
                # Neutral position - do nothing
                action = 'nothing'
            
            else:
                # Regular letter (A-Z)
                # Only add if it's different from last character in word
                if not self.current_word or self.current_word[-1] != current_letter:
                    self.current_word += current_letter
                    action = 'added'
                else:
                    action = 'duplicate_ignored'
            
            # Reset counter to avoid repeated actions
            self.stable_letter_count = 0
        
        return {
            'word': self.current_word,
            'action': action,
            'stable_letter': current_letter,
            'stability': self.stable_letter_count / config.STABLE_FRAMES_REQUIRED
        }

    
    def add_space(self):
        """Manually add a space to the current word."""
        if self.current_word and not self.current_word.endswith(' '):
            self.current_word += ' '
    
    def backspace(self):
        """Remove last character from current word."""
        if self.current_word:
            self.current_word = self.current_word[:-1]
    
    def clear_word(self):
        """Clear the current word."""
        self.current_word = ""
        self.last_stable_letter = None
        self.stable_letter_count = 0
        self.no_detection_count = 0
    
    def get_word(self):
        """Get the current word."""
        return self.current_word
    
    def get_statistics(self):
        """
        Get prediction statistics.
        
        Returns:
            Dictionary with statistics
        """
        confidence_rate = 0
        if self.total_predictions > 0:
            confidence_rate = self.confident_predictions / self.total_predictions
        
        return {
            'total_predictions': self.total_predictions,
            'confident_predictions': self.confident_predictions,
            'confidence_rate': confidence_rate
        }


if __name__ == "__main__":
    # Test predictor
    print("Testing predictor...")
    
    try:
        predictor = Predictor()
        print("\nPredictor initialized successfully!")
        
        # Test with random landmarks
        print("\nTesting with random landmarks...")
        random_landmarks = np.random.rand(63).astype(np.float32)
        
        prediction = predictor.predict(random_landmarks)
        print(f"\nPrediction: {prediction['letter']}")
        print(f"Confidence: {prediction['confidence']:.4f}")
        
        # Test smoothed prediction
        for i in range(15):
            smoothed = predictor.get_smoothed_prediction(random_landmarks)
            word_state = predictor.update_word(smoothed)
            print(f"Frame {i+1}: {smoothed['letter']} (conf: {smoothed['confidence']:.2f}, "
                  f"stability: {word_state['stability']:.2f})")
        
        print(f"\nFinal word: '{predictor.get_word()}'")
        print(f"Statistics: {predictor.get_statistics()}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you have trained the model first!")
        print("Run: python model/train.py")
