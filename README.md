# Multimodal Named Entity Recognition Project

A production-ready implementation of Multi-modal Named Entity Recognition (NER) that combines text-based and vision-based entity extraction using state-of-the-art transformer models.

## Overview

This project implements a comprehensive multimodal NER system that can extract named entities from:
- **Text**: Using transformer-based language models (BERT, RoBERTa)
- **Images**: Using vision-language models (CLIP, BLIP) for visual entity recognition
- **Combined**: Fusing text and visual information for enhanced entity recognition

## Features

- **Modern Architecture**: PyTorch 2.x with transformer-based models
- **Multimodal Fusion**: Late fusion, early fusion, and cross-attention mechanisms
- **Comprehensive Evaluation**: Token-level F1, entity-level F1, and visual grounding metrics
- **Interactive Demo**: Streamlit/Gradio interface for real-time testing
- **Production Ready**: Type hints, comprehensive testing, and proper documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Multimodal-Named-Entity-Recognition-Project.git
cd Multimodal-Named-Entity-Recognition-Project

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Basic Usage

```python
from src.models.multimodal_ner import MultimodalNERModel
from src.data.loaders import MultimodalNERDataLoader

# Initialize model
model = MultimodalNERModel.from_pretrained("bert-base-uncased")

# Load data
loader = MultimodalNERDataLoader("data/annotations.json")

# Run inference
results = model.predict(text="Elon Musk founded SpaceX", image_path="elon.jpg")
print(results.entities)
```

### Demo

Launch the interactive demo:

```bash
streamlit run demo/app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model implementations
│   ├── losses/            # Loss functions
│   ├── eval/              # Evaluation metrics
│   ├── viz/               # Visualization utilities
│   └── utils/              # Utility functions
├── configs/               # Configuration files
├── data/                  # Dataset storage
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── assets/                # Generated outputs and visualizations
└── demo/                  # Interactive demo application
```

## Models

### Text-based NER
- **BERT-NER**: Fine-tuned BERT for named entity recognition
- **RoBERTa-NER**: Enhanced RoBERTa model for better performance
- **SpanBERT**: Span-based entity recognition

### Vision-based NER
- **CLIP-NER**: Using CLIP embeddings for visual entity recognition
- **BLIP-NER**: BLIP model for image-text entity alignment
- **DETR-NER**: Object detection with entity classification

### Multimodal Fusion
- **Late Fusion**: Concatenating text and visual features
- **Early Fusion**: Joint encoding of text and images
- **Cross-Attention**: Attention-based fusion mechanisms

## Evaluation Metrics

- **Token-level F1**: Precision, recall, and F1 at token level
- **Entity-level F1**: Complete entity matching evaluation
- **Visual Grounding**: Accuracy of visual entity localization
- **Cross-modal Alignment**: Text-image entity correspondence

## Dataset

The project includes a synthetic multimodal NER dataset with:
- Text annotations with entity labels
- Corresponding images with visual entities
- Entity alignment between text and visual modalities

## Training

```bash
# Train text-only NER model
python scripts/train.py --config configs/text_ner.yaml

# Train multimodal NER model
python scripts/train.py --config configs/multimodal_ner.yaml

# Evaluate model
python scripts/evaluate.py --model_path checkpoints/best_model.pt
```

## Safety and Limitations

**IMPORTANT DISCLAIMERS:**

- This is a research/educational project and should not be used for production applications without proper validation
- The model may have biases present in the training data
- Visual entity recognition is limited by the quality and diversity of training images
- Results should be validated by domain experts for critical applications

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- spaCy for NLP tools
- Hugging Face Transformers for model implementations
- OpenCLIP for vision-language models
- The multimodal NLP research community
# Multimodal-Named-Entity-Recognition-Project
