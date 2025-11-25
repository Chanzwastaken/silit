"""
SILIT - Sign Language Translator
Model Training Script

This script trains the hand sign classification model using
extracted MediaPipe landmarks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

from model_architecture import create_model, create_advanced_model
from dataset_loader import load_and_prepare_data


class TrainingCallback(keras.callbacks.Callback):
    """Custom callback to display training progress."""
    
    def on_epoch_end(self, epoch, logs=None):
        """Print metrics at end of each epoch."""
        print(f"\nEpoch {epoch + 1}:")
        print(f"  Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}")
        print(f"  Val Loss: {logs['val_loss']:.4f} - Val Accuracy: {logs['val_accuracy']:.4f}")


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation metrics.
    
    Args:
        history: Keras training history object
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining history plot saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, label_names, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: List of label names
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def evaluate_model(model, X_test, y_test, label_names):
    """
    Evaluate model on test set and print metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_names: List of label names
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATING MODEL ON TEST SET")
    print("="*60)
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate test accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_names,
        digits=4
    ))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, label_names)
    
    # Calculate per-class accuracy
    cm = confusion_matrix(y_test, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    print("\nPer-Class Accuracy:")
    for i, (name, acc) in enumerate(zip(label_names, per_class_accuracy)):
        print(f"  {name}: {acc:.4f}")
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_probs
    }


def train_model(data_path='processed_data', 
                model_type='standard',
                epochs=50,
                batch_size=32,
                early_stopping_patience=10,
                save_dir='model_output'):
    """
    Train the hand sign classification model.
    
    Args:
        data_path: Path to processed data
        model_type: 'standard' or 'advanced'
        epochs: Number of training epochs
        batch_size: Batch size
        early_stopping_patience: Patience for early stopping
        save_dir: Directory to save model and outputs
    
    Returns:
        Trained model and training history
    """
    # Create output directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SILIT - SIGN LANGUAGE TRANSLATOR")
    print("MODEL TRAINING")
    print("="*60)
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    data = load_and_prepare_data(
        data_path=data_path,
        batch_size=batch_size,
        normalize=True,
        augment=True
    )
    
    # Create model
    print(f"\nCreating {model_type} model...")
    if model_type == 'advanced':
        model = create_advanced_model(num_classes=data['num_classes'])
    else:
        model = create_model(num_classes=data['num_classes'])
    
    model.summary()
    
    # Define callbacks
    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Custom callback
        TrainingCallback()
    ]
    
    # Train model
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    history = model.fit(
        data['train_dataset'],
        validation_data=data['val_dataset'],
        epochs=epochs,
        callbacks=callbacks,
        class_weight=data['class_weights'],
        verbose=0
    )
    
    # Plot training history
    plot_training_history(history, save_path=str(save_dir / 'training_history.png'))
    
    # Evaluate on test set
    metrics = evaluate_model(
        model,
        data['X_test'],
        data['y_test'],
        data['label_names']
    )
    
    # Save final model
    final_model_path = save_dir / 'silit_model.h5'
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save label names
    label_names_path = save_dir / 'label_names.pkl'
    with open(label_names_path, 'wb') as f:
        pickle.dump(data['label_names'], f)
    print(f"Label names saved to: {label_names_path}")
    
    # Save training configuration
    config = {
        'model_type': model_type,
        'epochs': epochs,
        'batch_size': batch_size,
        'num_classes': data['num_classes'],
        'label_names': data['label_names'],
        'final_test_accuracy': float(metrics['test_accuracy']),
        'final_test_loss': float(metrics['test_loss'])
    }
    
    config_path = save_dir / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training configuration saved to: {config_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Model saved to: {final_model_path}")
    print("="*60)
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train SILIT hand sign classification model'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='processed_data',
        help='Path to processed data directory'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='standard',
        choices=['standard', 'advanced'],
        help='Model architecture type'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='model_output',
        help='Output directory for model and plots'
    )
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_model(
        data_path=args.data,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.output
    )
