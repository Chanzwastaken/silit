"""
SILIT - Sign Language Translator
Utility Functions

This module contains helper functions for UI rendering,
text-to-speech, and other utilities.
"""

import cv2
import numpy as np
import time

import config


class FPSCounter:
    """Calculate and display FPS."""
    
    def __init__(self, buffer_size=30):
        """
        Initialize FPS counter.
        
        Args:
            buffer_size: Number of frames to average
        """
        self.buffer_size = buffer_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self):
        """Update FPS calculation."""
        current_time = time.time()
        delta = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(delta)
        if len(self.frame_times) > self.buffer_size:
            self.frame_times.pop(0)
    
    def get_fps(self):
        """Get current FPS."""
        if not self.frame_times:
            return 0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0


class TextToSpeech:
    """Text-to-speech wrapper."""
    
    def __init__(self, enabled=True):
        """
        Initialize TTS engine.
        
        Args:
            enabled: Whether TTS is enabled
        """
        self.enabled = enabled and config.TTS_ENABLED
        self.engine = None
        
        if self.enabled:
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', config.TTS_RATE)
                self.engine.setProperty('volume', config.TTS_VOLUME)
                print("Text-to-speech initialized")
            except Exception as e:
                print(f"Warning: Could not initialize TTS: {e}")
                self.enabled = False
    
    def speak(self, text):
        """
        Speak the given text.
        
        Args:
            text: Text to speak
        """
        if self.enabled and self.engine and text:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
    
    def speak_async(self, text):
        """
        Speak text asynchronously (non-blocking).
        
        Args:
            text: Text to speak
        """
        if self.enabled and self.engine and text:
            try:
                self.engine.say(text)
                self.engine.startLoop(False)
                self.engine.iterate()
                self.engine.endLoop()
            except Exception as e:
                print(f"TTS error: {e}")


def draw_text_with_background(frame, text, position, font_scale=1.0, 
                              font_thickness=2, text_color=(255, 255, 255),
                              bg_color=(0, 0, 0), padding=10):
    """
    Draw text with a background rectangle.
    
    Args:
        frame: Image to draw on
        text: Text to draw
        position: (x, y) position for text
        font_scale: Font scale
        font_thickness: Font thickness
        text_color: Text color (BGR)
        bg_color: Background color (BGR)
        padding: Padding around text
    
    Returns:
        Modified frame
    """
    x, y = position
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        config.FONT_FACE,
        font_scale,
        font_thickness
    )
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color,
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        text,
        (x, y),
        config.FONT_FACE,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA
    )
    
    return frame


def draw_info_panel(frame, predictor, fps, show_skeleton):
    """
    Draw information panel at the bottom of the frame.
    
    Args:
        frame: Image to draw on
        predictor: Predictor instance
        fps: Current FPS
        show_skeleton: Whether skeleton is shown
    
    Returns:
        Modified frame
    """
    h, w = frame.shape[:2]
    panel_height = config.INFO_PANEL_HEIGHT
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, h - panel_height),
        (w, h),
        config.COLOR_BACKGROUND,
        -1
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw border
    cv2.line(
        frame,
        (0, h - panel_height),
        (w, h - panel_height),
        config.COLOR_PRIMARY,
        2
    )
    
    # Get statistics
    stats = predictor.get_statistics()
    
    # Draw information
    y_offset = h - panel_height + 40
    line_spacing = 35
    
    # FPS
    if config.SHOW_FPS:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (config.MARGIN, y_offset),
            config.FONT_FACE,
            config.FONT_SCALE_SMALL,
            config.COLOR_TEXT,
            config.FONT_THICKNESS_SMALL,
            cv2.LINE_AA
        )
    
    # Predictions count
    cv2.putText(
        frame,
        f"Predictions: {stats['total_predictions']}",
        (config.MARGIN, y_offset + line_spacing),
        config.FONT_FACE,
        config.FONT_SCALE_SMALL,
        config.COLOR_TEXT,
        config.FONT_THICKNESS_SMALL,
        cv2.LINE_AA
    )
    
    # Confidence rate
    cv2.putText(
        frame,
        f"Confidence Rate: {stats['confidence_rate']*100:.1f}%",
        (config.MARGIN, y_offset + line_spacing * 2),
        config.FONT_FACE,
        config.FONT_SCALE_SMALL,
        config.COLOR_TEXT,
        config.FONT_THICKNESS_SMALL,
        cv2.LINE_AA
    )
    
    # Skeleton status
    skeleton_status = "ON" if show_skeleton else "OFF"
    cv2.putText(
        frame,
        f"Skeleton: {skeleton_status}",
        (config.MARGIN, y_offset + line_spacing * 3),
        config.FONT_FACE,
        config.FONT_SCALE_SMALL,
        config.COLOR_TEXT,
        config.FONT_THICKNESS_SMALL,
        cv2.LINE_AA
    )
    
    # Controls (right side)
    controls = [
        "SPACE: Add space",
        "BACKSPACE: Delete",
        "C: Clear word",
        "S: Speak word",
        "T: Toggle skeleton",
        "Q/ESC: Quit"
    ]
    
    x_offset = w - 300
    for i, control in enumerate(controls):
        cv2.putText(
            frame,
            control,
            (x_offset, y_offset + i * 25),
            config.FONT_FACE,
            config.FONT_SCALE_SMALL - 0.1,
            config.COLOR_TEXT,
            config.FONT_THICKNESS_SMALL,
            cv2.LINE_AA
        )
    
    return frame


def draw_prediction(frame, prediction, word_state):
    """
    Draw current prediction and word.
    
    Args:
        frame: Image to draw on
        prediction: Current prediction dictionary
        word_state: Word building state dictionary
    
    Returns:
        Modified frame
    """
    h, w = frame.shape[:2]
    
    # Draw current letter (top center)
    if prediction is not None:
        letter = prediction['letter']
        confidence = prediction['confidence']
        
        # Choose color based on confidence
        if confidence >= config.CONFIDENCE_THRESHOLD:
            color = config.COLOR_SUCCESS
        else:
            color = config.COLOR_WARNING
        
        # Draw letter
        text = f"{letter}"
        (text_width, text_height), _ = cv2.getTextSize(
            text,
            config.FONT_FACE,
            config.FONT_SCALE_LARGE,
            config.FONT_THICKNESS_LARGE
        )
        
        x = (w - text_width) // 2
        y = 80
        
        draw_text_with_background(
            frame,
            text,
            (x, y),
            font_scale=config.FONT_SCALE_LARGE,
            font_thickness=config.FONT_THICKNESS_LARGE,
            text_color=color,
            bg_color=(0, 0, 0),
            padding=20
        )
        
        # Draw confidence bar
        bar_width = 200
        bar_height = 20
        bar_x = (w - bar_width) // 2
        bar_y = y + 20
        
        # Background
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50),
            -1
        )
        
        # Confidence fill
        fill_width = int(bar_width * confidence)
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + fill_width, bar_y + bar_height),
            color,
            -1
        )
        
        # Confidence text
        cv2.putText(
            frame,
            f"{confidence*100:.1f}%",
            (bar_x + bar_width + 10, bar_y + 15),
            config.FONT_FACE,
            config.FONT_SCALE_SMALL,
            config.COLOR_TEXT,
            config.FONT_THICKNESS_SMALL,
            cv2.LINE_AA
        )
        
        # Draw stability indicator
        if word_state and 'stability' in word_state:
            stability = word_state['stability']
            if stability > 0:
                stability_text = f"Stability: {stability*100:.0f}%"
                cv2.putText(
                    frame,
                    stability_text,
                    (bar_x, bar_y + bar_height + 25),
                    config.FONT_FACE,
                    config.FONT_SCALE_SMALL,
                    config.COLOR_PRIMARY,
                    config.FONT_THICKNESS_SMALL,
                    cv2.LINE_AA
                )
    
    # Draw current word (center)
    word = word_state['word'] if word_state else ""
    if word:
        # Split into lines if too long
        max_chars_per_line = 30
        lines = []
        current_line = ""
        
        for char in word:
            if char == ' ' and len(current_line) >= max_chars_per_line:
                lines.append(current_line)
                current_line = ""
            else:
                current_line += char
                if len(current_line) >= max_chars_per_line:
                    lines.append(current_line)
                    current_line = ""
        
        if current_line:
            lines.append(current_line)
        
        # Draw lines
        y_start = h // 2 - (len(lines) * 40) // 2
        
        for i, line in enumerate(lines):
            (text_width, text_height), _ = cv2.getTextSize(
                line,
                config.FONT_FACE,
                config.FONT_SCALE_MEDIUM,
                config.FONT_THICKNESS_MEDIUM
            )
            
            x = (w - text_width) // 2
            y = y_start + i * 50
            
            draw_text_with_background(
                frame,
                line,
                (x, y),
                font_scale=config.FONT_SCALE_MEDIUM,
                font_thickness=config.FONT_THICKNESS_MEDIUM,
                text_color=config.COLOR_PRIMARY,
                bg_color=(0, 0, 0),
                padding=15
            )
    else:
        # Show instruction
        instruction = "Start signing to build a word"
        (text_width, _), _ = cv2.getTextSize(
            instruction,
            config.FONT_FACE,
            config.FONT_SCALE_SMALL,
            config.FONT_THICKNESS_SMALL
        )
        
        x = (w - text_width) // 2
        y = h // 2
        
        cv2.putText(
            frame,
            instruction,
            (x, y),
            config.FONT_FACE,
            config.FONT_SCALE_SMALL,
            (150, 150, 150),
            config.FONT_THICKNESS_SMALL,
            cv2.LINE_AA
        )
    
    return frame


def draw_title(frame):
    """
    Draw title bar.
    
    Args:
        frame: Image to draw on
    
    Returns:
        Modified frame
    """
    h, w = frame.shape[:2]
    
    # Draw background
    cv2.rectangle(
        frame,
        (0, 0),
        (w, 50),
        config.COLOR_BACKGROUND,
        -1
    )
    
    # Draw title
    title = "SILIT - Sign Language Translator"
    (text_width, _), _ = cv2.getTextSize(
        title,
        config.FONT_FACE,
        config.FONT_SCALE_MEDIUM,
        config.FONT_THICKNESS_MEDIUM
    )
    
    x = (w - text_width) // 2
    
    cv2.putText(
        frame,
        title,
        (x, 35),
        config.FONT_FACE,
        config.FONT_SCALE_MEDIUM,
        config.COLOR_PRIMARY,
        config.FONT_THICKNESS_MEDIUM,
        cv2.LINE_AA
    )
    
    return frame
