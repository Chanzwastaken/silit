"""
SILIT - Quick Start Helper
This script helps you get started with SILIT by checking prerequisites
and guiding you through the setup process.
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def check_python_version():
    """Check if Python version is 3.10 or higher."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.10+)")
        return False


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    
    required = [
        'tensorflow',
        'cv2',
        'mediapipe',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'tqdm'
    ]
    
    missing = []
    
    for package in required:
        try:
            if package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    return missing


def check_project_structure():
    """Check if project structure is correct."""
    print("\nChecking project structure...")
    
    required_dirs = [
        'model',
        'app'
    ]
    
    required_files = [
        'requirements.txt',
        'README.md',
        'model/model_architecture.py',
        'model/extract_landmarks.py',
        'model/dataset_loader.py',
        'model/train.py',
        'app/config.py',
        'app/mediapipe_wrapper.py',
        'app/predictor.py',
        'app/utils.py',
        'app/realtime.py'
    ]
    
    all_ok = True
    
    for dir_name in required_dirs:
        if Path(dir_name).is_dir():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ (missing)")
            all_ok = False
    
    for file_name in required_files:
        if Path(file_name).is_file():
            print(f"✓ {file_name}")
        else:
            print(f"✗ {file_name} (missing)")
            all_ok = False
    
    return all_ok


def check_dataset():
    """Check if dataset exists."""
    print("\nChecking for dataset...")
    
    dataset_path = Path('model/dataset')
    
    if not dataset_path.exists():
        print("✗ Dataset not found")
        print("\nYou need to download a dataset:")
        print("  1. Download ASL Alphabet from Kaggle:")
        print("     https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print("  2. Extract to model/dataset/")
        print("  3. Structure should be: dataset/A/, dataset/B/, ..., dataset/Z/")
        return False
    
    # Check for class folders
    class_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if len(class_folders) == 0:
        print("✗ No class folders found in dataset/")
        return False
    
    print(f"✓ Dataset found with {len(class_folders)} classes")
    
    # Count images
    total_images = 0
    for class_folder in class_folders:
        images = list(class_folder.glob('*.jpg')) + \
                list(class_folder.glob('*.png')) + \
                list(class_folder.glob('*.jpeg'))
        total_images += len(images)
    
    print(f"  Total images: {total_images}")
    
    if total_images < 100:
        print("  ⚠ Warning: Very few images. Consider getting more data.")
    
    return True


def check_trained_model():
    """Check if model is trained."""
    print("\nChecking for trained model...")
    
    model_path = Path('model/model_output/silit_model.h5')
    
    if model_path.exists():
        print("✓ Trained model found")
        return True
    else:
        print("✗ No trained model found")
        print("\nYou need to train the model:")
        print("  1. cd model")
        print("  2. python extract_landmarks.py")
        print("  3. python train.py")
        return False


def main():
    """Main function."""
    print_header("SILIT - Quick Start Helper")
    
    print("This script will check if you're ready to use SILIT.\n")
    
    # Check Python version
    if not check_python_version():
        print("\n⚠ Please upgrade to Python 3.10 or higher")
        return
    
    # Check project structure
    if not check_project_structure():
        print("\n⚠ Project structure is incomplete")
        print("Make sure all files are in place")
        return
    
    # Check dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠ Missing {len(missing)} package(s)")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
        return
    
    # Check dataset
    has_dataset = check_dataset()
    
    # Check trained model
    has_model = check_trained_model()
    
    # Summary
    print_header("Summary")
    
    if has_dataset and has_model:
        print("✓ Everything is ready!")
        print("\nTo run SILIT:")
        print("  cd app")
        print("  python realtime.py")
        print("\nControls:")
        print("  SPACE      - Add space")
        print("  BACKSPACE  - Delete character")
        print("  C          - Clear word")
        print("  S          - Speak word")
        print("  T          - Toggle skeleton")
        print("  Q or ESC   - Quit")
    
    elif has_dataset and not has_model:
        print("⚠ Dataset found, but model not trained")
        print("\nNext steps:")
        print("  1. cd model")
        print("  2. python extract_landmarks.py")
        print("  3. python train.py")
        print("  4. cd ../app")
        print("  5. python realtime.py")
    
    elif not has_dataset and has_model:
        print("⚠ Model found, but no dataset")
        print("(This is OK if you just want to use the trained model)")
        print("\nTo run SILIT:")
        print("  cd app")
        print("  python realtime.py")
    
    else:
        print("⚠ Dataset and model not found")
        print("\nNext steps:")
        print("  1. Download ASL Alphabet dataset from Kaggle")
        print("  2. Extract to model/dataset/")
        print("  3. cd model")
        print("  4. python extract_landmarks.py")
        print("  5. python train.py")
        print("  6. cd ../app")
        print("  7. python realtime.py")
    
    print("\n" + "="*60)
    print("\nFor detailed instructions, see README.md")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
