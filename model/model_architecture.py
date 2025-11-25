"""
SILIT - Sign Language Translator
Model Architecture Definition

This module defines the neural network architecture for classifying
hand sign landmarks into alphabetical characters (A-Z).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2


def create_model(input_shape=(63,), num_classes=26, dropout_rate=0.3):
    """
    Create a dense neural network for hand sign classification.
    
    Architecture:
    - Input: 63 features (21 landmarks × 3 coordinates)
    - Hidden layers with batch normalization and dropout
    - Output: 26 classes (A-Z) with softmax
    
    Args:
        input_shape: Shape of input features (default: 63)
        num_classes: Number of output classes (default: 26 for A-Z)
        dropout_rate: Dropout rate for regularization (default: 0.3)
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First dense block
        layers.Dense(256, kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        
        # Second dense block
        layers.Dense(128, kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        
        # Third dense block
        layers.Dense(64, kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate / 2),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ], name='SILIT_Classifier')
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_advanced_model(input_shape=(63,), num_classes=26):
    """
    Create an advanced model with residual connections.
    
    This is an alternative architecture with skip connections
    for potentially better performance.
    
    Args:
        input_shape: Shape of input features
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # First block
    x = layers.Dense(256, kernel_regularizer=l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Second block with residual
    residual = layers.Dense(128)(x)
    x = layers.Dense(128, kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Add()([x, residual])
    x = layers.Dropout(0.3)(x)
    
    # Third block
    x = layers.Dense(64, kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='SILIT_Advanced')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Creating standard model...")
    model = create_model()
    model.summary()
    
    print("\n" + "="*60 + "\n")
    print("Creating advanced model...")
    advanced_model = create_advanced_model()
    advanced_model.summary()
