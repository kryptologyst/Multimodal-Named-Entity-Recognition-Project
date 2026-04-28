"""Data loading and preprocessing for multimodal NER."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
import cv2

logger = logging.getLogger(__name__)


class MultimodalNERDataset(Dataset):
    """Dataset for multimodal Named Entity Recognition."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer_name: str = "bert-base-uncased",
        max_length: int = 512,
        image_size: int = 224,
        split: str = "train"
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to annotation file
            tokenizer_name: Name of tokenizer to use
            max_length: Maximum sequence length
            image_size: Size to resize images to
            split: Dataset split (train/val/test)
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.image_size = image_size
        self.split = split
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load data
        self.data = self._load_data()
        
        # Entity label mapping
        self.label2id = {
            "O": 0,  # Outside
            "B-PER": 1,  # Begin Person
            "I-PER": 2,  # Inside Person
            "B-ORG": 3,  # Begin Organization
            "I-ORG": 4,  # Inside Organization
            "B-LOC": 5,  # Begin Location
            "I-LOC": 6,  # Inside Location
            "B-MISC": 7,  # Begin Miscellaneous
            "I-MISC": 8,  # Inside Miscellaneous
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from annotation file."""
        if not self.data_path.exists():
            logger.warning(f"Data file {self.data_path} not found, creating synthetic data")
            return self._create_synthetic_data()
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def _create_synthetic_data(self) -> List[Dict[str, Any]]:
        """Create synthetic multimodal NER data for demonstration."""
        synthetic_data = [
            {
                "id": "sample_1",
                "text": "Elon Musk founded SpaceX in California.",
                "image_path": "data/images/sample_1.jpg",
                "entities": [
                    {"text": "Elon Musk", "label": "PER", "start": 0, "end": 10},
                    {"text": "SpaceX", "label": "ORG", "start": 18, "end": 24},
                    {"text": "California", "label": "LOC", "start": 28, "end": 37}
                ],
                "visual_entities": [
                    {"label": "person", "bbox": [100, 50, 200, 300], "confidence": 0.95},
                    {"label": "building", "bbox": [300, 200, 500, 400], "confidence": 0.87}
                ]
            },
            {
                "id": "sample_2",
                "text": "Apple Inc. is located in Cupertino, California.",
                "image_path": "data/images/sample_2.jpg",
                "entities": [
                    {"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 9},
                    {"text": "Cupertino", "label": "LOC", "start": 25, "end": 34},
                    {"text": "California", "label": "LOC", "start": 36, "end": 45}
                ],
                "visual_entities": [
                    {"label": "building", "bbox": [50, 100, 400, 350], "confidence": 0.92},
                    {"label": "car", "bbox": [200, 300, 300, 400], "confidence": 0.78}
                ]
            },
            {
                "id": "sample_3",
                "text": "Tim Cook is the CEO of Apple.",
                "image_path": "data/images/sample_3.jpg",
                "entities": [
                    {"text": "Tim Cook", "label": "PER", "start": 0, "end": 8},
                    {"text": "Apple", "label": "ORG", "start": 22, "end": 27}
                ],
                "visual_entities": [
                    {"label": "person", "bbox": [150, 80, 250, 280], "confidence": 0.89},
                    {"label": "computer", "bbox": [300, 200, 450, 300], "confidence": 0.85}
                ]
            }
        ]
        
        # Create synthetic images
        self._create_synthetic_images(synthetic_data)
        
        return synthetic_data
    
    def _create_synthetic_images(self, data: List[Dict[str, Any]]) -> None:
        """Create synthetic images for demonstration."""
        images_dir = Path("data/images")
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for sample in data:
            image_path = images_dir / f"{sample['id']}.jpg"
            if not image_path.exists():
                # Create a simple synthetic image
                img = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
                
                # Add some basic shapes to represent entities
                for entity in sample.get("visual_entities", []):
                    bbox = entity["bbox"]
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(img, entity["label"], (bbox[0], bbox[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imwrite(str(image_path), img)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item from dataset.
        
        Args:
            idx: Index of item
            
        Returns:
            Dictionary containing text, image, labels, and metadata
        """
        sample = self.data[idx]
        
        # Process text
        text_encoding = self._encode_text(sample["text"], sample["entities"])
        
        # Process image
        image_tensor = self._load_image(sample["image_path"])
        
        # Process visual entities
        visual_entities = sample.get("visual_entities", [])
        
        return {
            "id": sample["id"],
            "text": sample["text"],
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"],
            "labels": text_encoding["labels"],
            "image": image_tensor,
            "visual_entities": visual_entities,
            "entities": sample["entities"]
        }
    
    def _encode_text(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Encode text and create labels."""
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels
        labels = [self.label2id["O"]] * self.max_length
        
        # Map entities to token positions
        for entity in entities:
            start_pos = entity["start"]
            end_pos = entity["end"]
            label = entity["label"]
            
            # Find token positions for entity
            token_start = encoding.char_to_token(start_pos)
            token_end = encoding.char_to_token(end_pos - 1)
            
            if token_start is not None and token_end is not None:
                # Set B- label for first token
                labels[token_start] = self.label2id[f"B-{label}"]
                
                # Set I- labels for remaining tokens
                for i in range(token_start + 1, token_end + 1):
                    if i < self.max_length:
                        labels[i] = self.label2id[f"I-{label}"]
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    
    def _load_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Load and preprocess image."""
        image_path = Path(image_path)
        
        if not image_path.exists():
            # Create a placeholder image if file doesn't exist
            img = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size))
        
        # Convert to tensor and normalize
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img_tensor = (img_tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
        
        return img_tensor


def create_dataloader(
    dataset: MultimodalNERDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create DataLoader for dataset.
    
    Args:
        dataset: Dataset to create loader for
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched data
    """
    return {
        "ids": [item["id"] for item in batch],
        "texts": [item["text"] for item in batch],
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "images": torch.stack([item["image"] for item in batch]),
        "visual_entities": [item["visual_entities"] for item in batch],
        "entities": [item["entities"] for item in batch]
    }
