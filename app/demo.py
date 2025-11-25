"""
SILIT - Demo Script
This script runs a demonstration of SILIT with enhanced visual feedback
and automatic demo mode for showcasing.
"""

import cv2
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import config
from mediapipe_wrapper import MediaPipeWrapper
from predictor import Predictor
from utils import FPSCounter, TextToSpeech


class DemoMode:
    """Enhanced demo mode with visual effects and instructions."""
    
    def __init__(self):
        """Initialize demo components."""
        print("="*60)
        print("SILIT - DEMO MODE")
        print("="*60)
        
        # Initialize components
        self.mediapipe = MediaPipeWrapper()
        self.predictor = Predictor()
        self.tts = TextToSpeech()
        self.fps_counter = FPSCounter()
        
        # Demo state
        self.demo_phrases = [
            "HELLO",
            "WORLD",
            "SIGN LANGUAGE",
            "AI DEMO"
        ]
        self.current_phrase_idx = 0
        self.show_instructions = True
        self.show_skeleton = True
        
        # Camera
        self.cap = None
        
        print("✓ Demo mode initialized!")
    
    def initialize_camera(self):
        """Initialize camera."""
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        print("✓ Camera initialized")
    
    def draw_demo_overlay(self, frame, prediction, word_state):
        """Draw enhanced demo overlay."""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Title bar
        cv2.rectangle(overlay, (0, 0), (w, 80), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Title
        title = "SILIT - Sign Language Translator Demo"
        cv2.putText(frame, title, (20, 50), cv2.FONT_HERSHEY_BOLD, 
                   1.2, (255, 100, 50), 3, cv2.LINE_AA)
        
        # Current prediction (large, center)
        if prediction:
            letter = prediction['letter']
            confidence = prediction['confidence']
            
            # Color based on confidence
            if confidence >= 0.8:
                color = (100, 255, 100)  # Green
            elif confidence >= 0.6:
                color = (50, 200, 255)   # Yellow
            else:
                color = (50, 50, 255)    # Red
            
            # Draw letter
            font_scale = 3.0
            thickness = 5
            (text_width, text_height), _ = cv2.getTextSize(
                letter, cv2.FONT_HERSHEY_BOLD, font_scale, thickness
            )
            
            x = (w - text_width) // 2
            y = h // 2 - 100
            
            # Background box
            padding = 30
            cv2.rectangle(frame, 
                         (x - padding, y - text_height - padding),
                         (x + text_width + padding, y + padding),
                         (0, 0, 0), -1)
            
            # Letter text
            cv2.putText(frame, letter, (x, y), cv2.FONT_HERSHEY_BOLD,
                       font_scale, color, thickness, cv2.LINE_AA)
            
            # Confidence bar
            bar_width = 300
            bar_height = 30
            bar_x = (w - bar_width) // 2
            bar_y = y + 50
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            
            # Fill
            fill_width = int(bar_width * confidence)
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + fill_width, bar_y + bar_height),
                         color, -1)
            
            # Confidence text
            conf_text = f"{confidence*100:.1f}%"
            cv2.putText(frame, conf_text, (bar_x + bar_width + 15, bar_y + 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Current word (bottom center)
        word = word_state['word'] if word_state else ""
        if word:
            font_scale = 1.5
            thickness = 3
            (text_width, text_height), _ = cv2.getTextSize(
                word, cv2.FONT_HERSHEY_BOLD, font_scale, thickness
            )
            
            x = (w - text_width) // 2
            y = h - 150
            
            # Background
            padding = 20
            cv2.rectangle(frame,
                         (x - padding, y - text_height - padding),
                         (x + text_width + padding, y + padding),
                         (0, 0, 0), -1)
            
            # Word
            cv2.putText(frame, word, (x, y), cv2.FONT_HERSHEY_BOLD,
                       font_scale, (255, 100, 50), thickness, cv2.LINE_AA)
        
        # Info panel (bottom)
        panel_height = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Stats
        stats = self.predictor.get_statistics()
        fps = self.fps_counter.get_fps()
        
        y_offset = h - panel_height + 30
        
        info_items = [
            f"FPS: {fps:.1f}",
            f"Predictions: {stats['total_predictions']}",
            f"Confidence Rate: {stats['confidence_rate']*100:.1f}%",
            f"Classes: 29 (A-Z + space, del, nothing)"
        ]
        
        x_offset = 20
        for item in info_items:
            cv2.putText(frame, item, (x_offset, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 25
        
        # Controls (right side)
        controls = [
            "Controls:",
            "SPACE: Add space",
            "C: Clear word",
            "Q: Quit"
        ]
        
        x_offset = w - 250
        y_offset = h - panel_height + 30
        for control in controls:
            cv2.putText(frame, control, (x_offset, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            y_offset += 20
        
        return frame
    
    def run(self):
        """Run demo mode."""
        try:
            self.initialize_camera()
            
            cv2.namedWindow("SILIT Demo", cv2.WINDOW_NORMAL)
            
            print("\n" + "="*60)
            print("DEMO MODE ACTIVE")
            print("="*60)
            print("\nShowcasing SILIT capabilities...")
            print("Press 'Q' to quit\n")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip frame
                if config.FLIP_CAMERA:
                    frame = cv2.flip(frame, 1)
                
                # Process frame
                self.mediapipe.process_frame(frame)
                landmarks = self.mediapipe.extract_landmarks(frame.shape)
                
                # Get prediction
                if landmarks is not None:
                    prediction = self.predictor.get_smoothed_prediction(landmarks)
                    word_state = self.predictor.update_word(prediction)
                else:
                    prediction = None
                    word_state = self.predictor.update_word(None)
                
                # Draw hand skeleton
                if self.show_skeleton:
                    frame = self.mediapipe.draw_landmarks(frame)
                
                # Draw demo overlay
                frame = self.draw_demo_overlay(frame, prediction, word_state)
                
                # Update FPS
                self.fps_counter.update()
                
                # Display
                cv2.imshow("SILIT Demo", frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord(' '):
                    self.predictor.add_space()
                elif key == ord('c'):
                    self.predictor.clear_word()
                elif key == ord('t'):
                    self.show_skeleton = not self.show_skeleton
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.mediapipe.close()
        
        print("\n" + "="*60)
        print("DEMO COMPLETE")
        print("="*60)


def main():
    """Main entry point."""
    try:
        demo = DemoMode()
        demo.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
