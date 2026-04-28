"""Setup script for the multimodal NER project."""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_spacy_model():
    """Download spaCy model."""
    print("Downloading spaCy model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    directories = [
        "data/images",
        "data/text", 
        "data/annotations",
        "logs",
        "checkpoints",
        "assets/outputs",
        "assets/visualizations"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")

def setup_pre_commit():
    """Setup pre-commit hooks."""
    try:
        print("Setting up pre-commit hooks...")
        subprocess.check_call(["pre-commit", "install"])
        print("Pre-commit hooks installed successfully")
    except subprocess.CalledProcessError:
        print("Pre-commit not available, skipping...")

def main():
    """Main setup function."""
    print("Setting up Multimodal NER Project...")
    print("=" * 50)
    
    try:
        create_directories()
        install_requirements()
        download_spacy_model()
        setup_pre_commit()
        
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python scripts/quick_test.py' to test the installation")
        print("2. Run 'streamlit run demo/app.py' to launch the demo")
        print("3. Run 'python scripts/train.py --config configs/multimodal_ner.yaml' to train a model")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
