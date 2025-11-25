"""
SILIT - Sign Language Translator
Dataset Loading and Preprocessing

This module handles loading, splitting, and preprocessing of
extracted landmark features for model training.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


class DatasetLoader:
    """Load and preprocess hand landmark dataset."""
    
    def __init__(self, data_path='processed_data'):
        """
        Initialize dataset loader.
        
        Args:
            data_path: Path to processed data directory
        """
        self.data_path = Path(data_path)
        self.scaler = StandardScaler()
        self.label_names = None
        
    def load_data(self):
        """
        Load features and labels from disk.
        
        Returns:
            Tuple of (features, labels, label_names)
        """
        features_path = self.data_path / 'features.npy'
        labels_path = self.data_path / 'labels.npy'
        label_names_path = self.data_path / 'label_names.pkl'
        
        # Check if files exist
        if not features_path.exists():
            raise FileNotFoundError(
                f"Features file not found: {features_path}\n"
                "Please run extract_landmarks.py first."
            )
        
        # Load data
        features = np.load(features_path)
        labels = np.load(labels_path)
        
        # Load label names
        if label_names_path.exists():
            with open(label_names_path, 'rb') as f:
                self.label_names = pickle.load(f)
        
        print(f"Loaded {len(features)} samples")
        print(f"Feature shape: {features.shape}")
        print(f"Number of classes: {len(np.unique(labels))}")
        
        return features, labels, self.label_names
    
    def split_data(self, features, labels, test_size=0.15, val_size=0.15, 
                   random_state=42):
        """
        Split data into train, validation, and test sets.
        
        Args:
            features: Feature array
            labels: Label array
            test_size: Proportion for test set (default: 0.15)
            val_size: Proportion for validation set (default: 0.15)
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        print("\nDataset split:")
        print(f"  Training:   {len(X_train)} samples ({len(X_train)/len(features)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(features)*100:.1f}%)")
        print(f"  Test:       {len(X_test)} samples ({len(X_test)/len(features)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize_features(self, X_train, X_val, X_test):
        """
        Normalize features using StandardScaler.
        
        Fits scaler on training data and transforms all sets.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
        
        Returns:
            Tuple of (X_train_norm, X_val_norm, X_test_norm)
        """
        # Fit on training data
        X_train_norm = self.scaler.fit_transform(X_train)
        
        # Transform validation and test data
        X_val_norm = self.scaler.transform(X_val)
        X_test_norm = self.scaler.transform(X_test)
        
        print("\nFeatures normalized using StandardScaler")
        
        return X_train_norm, X_val_norm, X_test_norm
    
    def save_scaler(self, output_path='processed_data/scaler.pkl'):
        """
        Save fitted scaler for inference.
        
        Args:
            output_path: Path to save scaler
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Scaler saved to: {output_path}")
    
    def create_tf_dataset(self, X, y, batch_size=32, shuffle=True, 
                         augment=False):
        """
        Create TensorFlow dataset for training.
        
        Args:
            X: Features
            y: Labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augment: Whether to apply data augmentation
        
        Returns:
            tf.data.Dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        if augment:
            # Add small random noise for augmentation
            def augment_fn(features, label):
                noise = tf.random.normal(
                    shape=tf.shape(features),
                    mean=0.0,
                    stddev=0.02
                )
                features = features + noise
                return features, label
            
            dataset = dataset.map(augment_fn)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_class_weights(self, y_train):
        """
        Calculate class weights for imbalanced datasets.
        
        Args:
            y_train: Training labels
        
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        
        class_weights = dict(zip(classes, weights))
        
        print("\nClass weights computed for imbalanced dataset")
        
        return class_weights


def load_and_prepare_data(data_path='processed_data', batch_size=32, 
                          normalize=True, augment=True):
    """
    Convenience function to load and prepare data in one call.
    
    Args:
        data_path: Path to processed data
        batch_size: Batch size for training
        normalize: Whether to normalize features
        augment: Whether to augment training data
    
    Returns:
        Dictionary containing all prepared data and metadata
    """
    loader = DatasetLoader(data_path)
    
    # Load data
    features, labels, label_names = loader.load_data()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
        features, labels
    )
    
    # Normalize if requested
    if normalize:
        X_train, X_val, X_test = loader.normalize_features(
            X_train, X_val, X_test
        )
        loader.save_scaler()
    
    # Create TensorFlow datasets
    train_dataset = loader.create_tf_dataset(
        X_train, y_train, batch_size, shuffle=True, augment=augment
    )
    val_dataset = loader.create_tf_dataset(
        X_val, y_val, batch_size, shuffle=False, augment=False
    )
    test_dataset = loader.create_tf_dataset(
        X_test, y_test, batch_size, shuffle=False, augment=False
    )
    
    # Calculate class weights
    class_weights = loader.get_class_weights(y_train)
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'X_test': X_test,
        'y_test': y_test,
        'label_names': label_names,
        'class_weights': class_weights,
        'num_classes': len(label_names)
    }


if __name__ == "__main__":
    # Test data loading
    print("Testing dataset loader...")
    data = load_and_prepare_data()
    
    print("\n" + "="*60)
    print("Dataset prepared successfully!")
    print(f"Number of classes: {data['num_classes']}")
    print(f"Label names: {data['label_names']}")
    print("="*60)
