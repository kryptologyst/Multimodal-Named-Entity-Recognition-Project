"""Simple evaluation script for quick testing."""

import torch
from src.models.multimodal_ner import MultimodalNERModel, TextNERModel
from src.utils.device import setup_device, set_seed
from transformers import AutoTokenizer

def quick_test():
    """Quick test of the multimodal NER system."""
    print("Testing Multimodal NER System...")
    
    # Setup
    device = setup_device()
    set_seed(42, True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Test text
    text = "Elon Musk founded SpaceX in California."
    
    # Initialize models
    text_model = TextNERModel(num_labels=9)
    multimodal_model = MultimodalNERModel(num_labels=9)
    
    # Tokenize text
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    # Create dummy image
    image = torch.randn(1, 3, 224, 224)
    
    # Test text model
    with torch.no_grad():
        text_outputs = text_model(input_ids, attention_mask)
        text_predictions = torch.argmax(text_outputs.logits, dim=-1)
        print(f"Text model predictions shape: {text_predictions.shape}")
    
    # Test multimodal model
    with torch.no_grad():
        multimodal_outputs = multimodal_model(input_ids, attention_mask, image)
        multimodal_predictions = torch.argmax(multimodal_outputs.logits, dim=-1)
        print(f"Multimodal model predictions shape: {multimodal_predictions.shape}")
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    quick_test()
