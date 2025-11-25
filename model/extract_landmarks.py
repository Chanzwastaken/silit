"""
SILIT - Sign Language Translator
MediaPipe Landmark Extraction from Image Dataset

This script processes an image dataset of hand signs and extracts
MediaPipe hand landmarks for model training.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import pickle


class LandmarkExtractor:
    """Extract hand landmarks from images using MediaPipe."""
    
    def __init__(self, static_image_mode=True, max_num_hands=1, 
                 min_detection_confidence=0.5):
        """
        Initialize MediaPipe Hands detector.
        
        Args:
            static_image_mode: Whether to treat images as static
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
    
    def extract_landmarks(self, image_path):
        """
        Extract normalized hand landmarks from an image.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            numpy array of shape (63,) containing x, y, z coordinates
            of 21 landmarks, or None if no hand detected
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(image_rgb)
        
        # Extract landmarks
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract x, y, z coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks, dtype=np.float32)
        
        return None
    
    def close(self):
        """Close MediaPipe Hands detector."""
        self.hands.close()


def process_dataset(dataset_path, output_path='processed_data', filter_alphabet_only=False):
    """
    Process entire dataset and extract landmarks.
    
    Expected dataset structure:
    dataset_path/
        A/
            img1.jpg
            img2.jpg
            ...
        B/
            img1.jpg
            ...
        ...
        Z/
        del/      # Delete gesture (optional)
        space/    # Space gesture (optional)
        nothing/  # Neutral/rest position (optional)
    
    Args:
        dataset_path: Path to dataset root directory
        output_path: Path to save processed features and labels
        filter_alphabet_only: If True, only process A-Z folders (default: False)
                             Set to False to include special gestures (del, space, nothing)
    
    Returns:
        Dictionary with statistics
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    extractor = LandmarkExtractor()
    
    features = []
    labels = []
    label_names = []
    
    # Get all class directories
    all_class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    # Filter to only A-Z if requested
    if filter_alphabet_only:
        # Valid alphabet letters
        valid_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        class_dirs = [d for d in all_class_dirs if d.name in valid_letters]
        print(f"Filtering to alphabet only: {len(class_dirs)} classes (A-Z)")
        if len(all_class_dirs) > len(class_dirs):
            excluded = [d.name for d in all_class_dirs if d.name not in valid_letters]
            print(f"Excluded classes: {', '.join(excluded)}")
    else:
        class_dirs = all_class_dirs
    
    if len(class_dirs) == 0:
        raise ValueError(f"No valid class directories found in {dataset_path}")
    
    print(f"Found {len(class_dirs)} classes to process")
    print(f"Processing images from: {dataset_path}")
    print(f"Classes: {', '.join([d.name for d in class_dirs])}")
    
    total_images = 0
    successful_extractions = 0
    failed_extractions = 0
    
    # Process each class
    for class_idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        label_names.append(class_name)
        
        # Get all images in class directory
        image_files = list(class_dir.glob('*.jpg')) + \
                     list(class_dir.glob('*.png')) + \
                     list(class_dir.glob('*.jpeg'))
        
        if len(image_files) == 0:
            print(f"\nWarning: No images found in class '{class_name}'")
            continue
        
        print(f"\nProcessing class '{class_name}' ({len(image_files)} images)...")
        
        # Process each image
        for image_path in tqdm(image_files, desc=f"Class {class_name}"):
            total_images += 1
            
            # Extract landmarks
            landmarks = extractor.extract_landmarks(image_path)
            
            if landmarks is not None:
                features.append(landmarks)
                labels.append(class_idx)
                successful_extractions += 1
            else:
                failed_extractions += 1
    
    extractor.close()
    
    if len(features) == 0:
        raise ValueError("No features extracted! Check your dataset and ensure images contain visible hands.")
    
    # Convert to numpy arrays
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    # Save processed data
    np.save(output_path / 'features.npy', features)
    np.save(output_path / 'labels.npy', labels)
    
    # Save label names
    with open(output_path / 'label_names.pkl', 'wb') as f:
        pickle.dump(label_names, f)
    
    # Print statistics
    stats = {
        'total_images': total_images,
        'successful_extractions': successful_extractions,
        'failed_extractions': failed_extractions,
        'num_classes': len(label_names),
        'feature_shape': features.shape,
        'label_shape': labels.shape
    }
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Total images processed: {total_images}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")
    if total_images > 0:
        print(f"Success rate: {successful_extractions/total_images*100:.2f}%")
    print(f"Number of classes: {len(label_names)}")
    print(f"Classes: {', '.join(label_names)}")
    print(f"Feature shape: {features.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"\nData saved to: {output_path}")
    print("="*60)
    
    return stats



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract MediaPipe landmarks from hand sign images'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset',
        help='Path to dataset directory (default: dataset)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='processed_data',
        help='Path to save processed data (default: processed_data)'
    )
    
    args = parser.parse_args()
    
    # Process dataset
    stats = process_dataset(args.dataset, args.output)
