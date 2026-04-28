"""Test suite for multimodal NER project."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json

from src.models.multimodal_ner import MultimodalNERModel, TextNERModel
from src.data.loaders import MultimodalNERDataset, create_dataloader
from src.eval.metrics import MultimodalNEREvaluator, VisualGroundingMetrics, CrossModalAlignmentMetrics
from src.losses.multimodal_losses import MultimodalNERLoss, FocalLoss, ContrastiveLoss
from src.utils.device import setup_device, set_seed
from src.utils.config import Config, create_default_config


class TestMultimodalNERModel:
    """Test cases for multimodal NER model."""
    
    def test_text_ner_model_initialization(self):
        """Test text NER model initialization."""
        model = TextNERModel(
            model_name="bert-base-uncased",
            num_labels=9,
            dropout=0.1
        )
        
        assert model.num_labels == 9
        assert hasattr(model, 'bert')
        assert hasattr(model, 'classifier')
    
    def test_multimodal_ner_model_initialization(self):
        """Test multimodal NER model initialization."""
        model = MultimodalNERModel(
            text_model_name="bert-base-uncased",
            vision_model_name="ViT-B/32",
            num_labels=9,
            fusion_method="cross_attention",
            dropout=0.1
        )
        
        assert model.num_labels == 9
        assert hasattr(model, 'text_encoder')
        assert hasattr(model, 'vision_encoder')
        assert hasattr(model, 'fusion')
    
    def test_text_ner_forward_pass(self):
        """Test text NER forward pass."""
        model = TextNERModel(num_labels=9)
        
        # Create dummy inputs
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 9, (batch_size, seq_len))
        
        outputs = model(input_ids, attention_mask, labels)
        
        assert outputs.logits.shape == (batch_size, seq_len, 9)
        assert outputs.loss is not None
        assert outputs.loss.item() > 0
    
    def test_multimodal_ner_forward_pass(self):
        """Test multimodal NER forward pass."""
        model = MultimodalNERModel(num_labels=9)
        
        # Create dummy inputs
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        images = torch.randn(batch_size, 3, 224, 224)
        labels = torch.randint(0, 9, (batch_size, seq_len))
        
        outputs = model(input_ids, attention_mask, images, labels)
        
        assert outputs.logits.shape == (batch_size, seq_len, 9)
        assert outputs.loss is not None
        assert outputs.loss.item() > 0


class TestDataLoaders:
    """Test cases for data loaders."""
    
    def test_multimodal_dataset_creation(self):
        """Test multimodal dataset creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy data
            data_path = Path(temp_dir) / "test_data.json"
            dummy_data = [
                {
                    "id": "test_1",
                    "text": "Elon Musk founded SpaceX.",
                    "image_path": "test_image.jpg",
                    "entities": [
                        {"text": "Elon Musk", "label": "PER", "start": 0, "end": 9},
                        {"text": "SpaceX", "label": "ORG", "start": 18, "end": 24}
                    ],
                    "visual_entities": [
                        {"label": "person", "bbox": [100, 50, 200, 300], "confidence": 0.9}
                    ]
                }
            ]
            
            with open(data_path, 'w') as f:
                json.dump(dummy_data, f)
            
            # Create dataset
            dataset = MultimodalNERDataset(
                data_path=str(data_path),
                tokenizer_name="bert-base-uncased",
                max_length=128,
                image_size=224,
                split="test"
            )
            
            assert len(dataset) == 1
            assert dataset.label2id["O"] == 0
            assert dataset.label2id["B-PER"] == 1
    
    def test_dataloader_creation(self):
        """Test dataloader creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy data
            data_path = Path(temp_dir) / "test_data.json"
            dummy_data = [
                {
                    "id": "test_1",
                    "text": "Test text",
                    "image_path": "test_image.jpg",
                    "entities": [],
                    "visual_entities": []
                }
            ]
            
            with open(data_path, 'w') as f:
                json.dump(dummy_data, f)
            
            # Create dataset and dataloader
            dataset = MultimodalNERDataset(
                data_path=str(data_path),
                tokenizer_name="bert-base-uncased",
                max_length=128,
                image_size=224,
                split="test"
            )
            
            dataloader = create_dataloader(dataset, batch_size=1, shuffle=False)
            
            # Test dataloader
            batch = next(iter(dataloader))
            
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" in batch
            assert "images" in batch
            assert batch["input_ids"].shape[0] == 1


class TestMetrics:
    """Test cases for evaluation metrics."""
    
    def test_ner_metrics(self):
        """Test NER metrics computation."""
        label2id = {
            "O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4,
            "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8
        }
        
        evaluator = MultimodalNEREvaluator(label2id)
        
        # Create dummy predictions and labels
        batch_size, seq_len = 2, 10
        predictions = torch.randint(0, 9, (batch_size, seq_len))
        labels = torch.randint(0, 9, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        metrics = evaluator.ner_metrics.compute_token_metrics(
            predictions, labels, attention_mask
        )
        
        assert "token_precision" in metrics
        assert "token_recall" in metrics
        assert "token_f1" in metrics
        assert 0 <= metrics["token_f1"] <= 1
    
    def test_visual_grounding_metrics(self):
        """Test visual grounding metrics."""
        metrics_calc = VisualGroundingMetrics()
        
        predicted_entities = [
            {"text": "person", "label": "PER", "start": 0, "end": 6}
        ]
        visual_entities = [
            {"label": "person", "bbox": [100, 50, 200, 300], "confidence": 0.9}
        ]
        text_entities = []
        
        metrics = metrics_calc.compute_grounding_metrics(
            predicted_entities, visual_entities, text_entities
        )
        
        assert "visual_grounding_precision" in metrics
        assert "visual_grounding_recall" in metrics
        assert "visual_grounding_f1" in metrics
    
    def test_cross_modal_alignment_metrics(self):
        """Test cross-modal alignment metrics."""
        metrics_calc = CrossModalAlignmentMetrics()
        
        # Create dummy features
        batch_size, seq_len, dim = 2, 10, 768
        text_features = torch.randn(batch_size, seq_len, dim)
        vision_features = torch.randn(batch_size, dim)
        attention_mask = torch.ones(batch_size, seq_len)
        
        metrics = metrics_calc.compute_alignment_metrics(
            text_features, vision_features, attention_mask
        )
        
        assert "cross_modal_similarity" in metrics
        assert "cross_modal_alignment_score" in metrics


class TestLosses:
    """Test cases for loss functions."""
    
    def test_multimodal_ner_loss(self):
        """Test multimodal NER loss."""
        loss_fn = MultimodalNERLoss()
        
        # Create dummy inputs
        batch_size, seq_len, num_labels = 2, 10, 9
        ner_logits = torch.randn(batch_size, seq_len, num_labels)
        ner_labels = torch.randint(0, num_labels, (batch_size, seq_len))
        
        losses = loss_fn(ner_logits, ner_labels)
        
        assert "ner_loss" in losses
        assert "total_loss" in losses
        assert losses["total_loss"].item() > 0
    
    def test_focal_loss(self):
        """Test focal loss."""
        loss_fn = FocalLoss()
        
        # Create dummy inputs
        batch_size, num_classes = 10, 5
        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        loss = loss_fn(inputs, targets)
        
        assert loss.item() > 0
        assert loss.item() < 10  # Reasonable range
    
    def test_contrastive_loss(self):
        """Test contrastive loss."""
        loss_fn = ContrastiveLoss()
        
        # Create dummy features
        batch_size, dim = 4, 128
        text_features = torch.randn(batch_size, dim)
        vision_features = torch.randn(batch_size, dim)
        
        loss = loss_fn(text_features, vision_features)
        
        assert loss.item() > 0
        assert loss.item() < 10  # Reasonable range


class TestUtils:
    """Test cases for utility functions."""
    
    def test_device_setup(self):
        """Test device setup."""
        device = setup_device()
        assert isinstance(device, torch.device)
    
    def test_seed_setting(self):
        """Test seed setting."""
        set_seed(42, True)
        
        # Generate some random numbers
        torch_rand = torch.rand(1).item()
        np_rand = np.random.rand()
        
        # Reset seed and generate again
        set_seed(42, True)
        torch_rand2 = torch.rand(1).item()
        np_rand2 = np.random.rand()
        
        # Should be the same (approximately)
        assert abs(torch_rand - torch_rand2) < 1e-6
        assert abs(np_rand - np_rand2) < 1e-6
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = create_default_config()
        
        assert config.get("model.name") == "multimodal_ner"
        assert config.get("training.batch_size") == 16
        assert config.get("seed") == 42
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            # Create and save config
            config = create_default_config()
            config.save_config(config_path)
            
            # Load config
            loaded_config = Config(config_path)
            
            assert loaded_config.get("model.name") == config.get("model.name")
            assert loaded_config.get("training.batch_size") == config.get("training.batch_size")


if __name__ == "__main__":
    pytest.main([__file__])
