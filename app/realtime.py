"""
SILIT - Sign Language Translator
Real-Time Application

This is the main application that integrates all components
for real-time sign language translation.
"""

import cv2
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import config
from mediapipe_wrapper import MediaPipeWrapper
from predictor import Predictor
from utils import (
    FPSCounter, TextToSpeech, draw_info_panel,
    draw_prediction, draw_title
)


class SILITApp:
    """Main SILIT application."""
    
    def __init__(self):
        """Initialize the application."""
        print("="*60)
        print("SILIT - SIGN LANGUAGE TRANSLATOR")
        print("="*60)
        
        # Initialize components
        print("\nInitializing components...")
        
        try:
            # MediaPipe wrapper
            print("  - MediaPipe hand detector...")
            self.mediapipe = MediaPipeWrapper()
            
            # Predictor
            print("  - Loading trained model...")
            self.predictor = Predictor()
            
            # TTS
            print("  - Text-to-speech engine...")
            self.tts = TextToSpeech()
            
            # FPS counter
            self.fps_counter = FPSCounter()
            
            print("\n✓ All components initialized successfully!")
            
        except Exception as e:
            print(f"\n✗ Error initializing components: {e}")
            print("\nPlease make sure you have:")
            print("  1. Trained the model (run: python model/train.py)")
            print("  2. Installed all dependencies (run: pip install -r requirements.txt)")
            raise
        
        # Camera
        self.cap = None
        
        # UI state
        self.show_skeleton = True
        self.running = False
        
        # Current prediction state
        self.current_prediction = None
        self.word_state = {'word': '', 'action': 'none', 'stable_letter': None, 'stability': 0}
    
    def initialize_camera(self):
        """Initialize camera capture."""
        print(f"\nInitializing camera {config.CAMERA_INDEX}...")
        
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {config.CAMERA_INDEX}. "
                "Please check your camera connection."
            )
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        
        # Get actual camera properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"✓ Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame: Camera frame
        
        Returns:
            Processed frame with overlays
        """
        # Flip frame if configured
        if config.FLIP_CAMERA:
            frame = cv2.flip(frame, 1)
        
        # Process with MediaPipe
        self.mediapipe.process_frame(frame)
        
        # Extract landmarks
        landmarks = self.mediapipe.extract_landmarks(frame.shape)
        
        # Get prediction
        if landmarks is not None:
            self.current_prediction = self.predictor.get_smoothed_prediction(landmarks)
            self.word_state = self.predictor.update_word(self.current_prediction)
        else:
            self.current_prediction = None
            self.word_state = self.predictor.update_word(None)
        
        # Draw hand skeleton
        if self.show_skeleton:
            frame = self.mediapipe.draw_landmarks(frame)
        
        # Draw UI elements
        frame = draw_title(frame)
        frame = draw_prediction(frame, self.current_prediction, self.word_state)
        frame = draw_info_panel(
            frame,
            self.predictor,
            self.fps_counter.get_fps(),
            self.show_skeleton
        )
        
        return frame
    
    def handle_keyboard(self, key):
        """
        Handle keyboard input.
        
        Args:
            key: Key code from cv2.waitKey()
        
        Returns:
            Boolean indicating whether to continue running
        """
        # Quit
        if key == config.KEY_QUIT or key == config.KEY_ESC:
            return False
        
        # Add space
        elif key == config.KEY_SPACE:
            self.predictor.add_space()
            print("Added space")
        
        # Backspace
        elif key == config.KEY_BACKSPACE:
            self.predictor.backspace()
            print("Deleted last character")
        
        # Clear word
        elif key == config.KEY_CLEAR:
            self.predictor.clear_word()
            print("Cleared word")
        
        # Speak word
        elif key == config.KEY_SPEAK:
            word = self.predictor.get_word()
            if word:
                print(f"Speaking: '{word}'")
                self.tts.speak(word)
            else:
                print("No word to speak")
        
        # Toggle skeleton
        elif key == config.KEY_TOGGLE_SKELETON:
            self.show_skeleton = not self.show_skeleton
            status = "ON" if self.show_skeleton else "OFF"
            print(f"Hand skeleton: {status}")
        
        return True
    
    def run(self):
        """Run the main application loop."""
        try:
            # Initialize camera
            self.initialize_camera()
            
            # Create window
            cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
            
            print("\n" + "="*60)
            print("APPLICATION STARTED")
            print("="*60)
            print("\nControls:")
            print("  SPACE      - Add space to word")
            print("  BACKSPACE  - Delete last character")
            print("  C          - Clear word")
            print("  S          - Speak current word")
            print("  T          - Toggle hand skeleton")
            print("  Q or ESC   - Quit")
            print("\nStart signing to translate!")
            print("="*60 + "\n")
            
            self.running = True
            
            # Main loop
            while self.running:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Update FPS
                self.fps_counter.update()
                
                # Display frame
                cv2.imshow(config.WINDOW_NAME, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    if not self.handle_keyboard(key):
                        break
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        except Exception as e:
            print(f"\n\nError: {e}")
            raise
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\n" + "="*60)
        print("SHUTTING DOWN")
        print("="*60)
        
        # Get final statistics
        stats = self.predictor.get_statistics()
        final_word = self.predictor.get_word()
        
        print(f"\nFinal word: '{final_word}'")
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Confident predictions: {stats['confident_predictions']}")
        print(f"Confidence rate: {stats['confidence_rate']*100:.1f}%")
        
        # Release resources
        if self.cap is not None:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.mediapipe.close()
        
        print("\n✓ Cleanup complete")
        print("="*60)


def main():
    """Main entry point."""
    try:
        app = SILITApp()
        app.run()
    
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
