"""Evaluation script for multimodal NER models."""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.device import setup_device, set_seed
from src.data.loaders import MultimodalNERDataset, create_dataloader
from src.models.multimodal_ner import MultimodalNERModel, TextNERModel
from src.eval.metrics import MultimodalNEREvaluator
from src.viz.visualization import create_metrics_dashboard

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalNEREvaluator:
    """Evaluator for multimodal NER models."""
    
    def __init__(self, config_path: str, model_path: str):
        """
        Initialize evaluator.
        
        Args:
            config_path: Path to configuration file
            model_path: Path to model checkpoint
        """
        self.config = load_config(config_path)
        
        # Setup device
        self.device = setup_device(
            fallback_order=self.config.get("device.fallback_order", ["cuda", "mps", "cpu"])
        )
        set_seed(self.config.get("seed", 42), self.config.get("deterministic", True))
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize data loader
        self.test_loader = self._initialize_data_loader()
        
        # Initialize metrics
        self.evaluator = MultimodalNEREvaluator(self.model.text_encoder.label2id)
        
        logger.info(f"Evaluator initialized on device: {self.device}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model
        model_config = self.config.get("model", {})
        
        if model_config.get("name") == "text_ner":
            model = TextNERModel(
                model_name=model_config.get("encoder.name", "bert-base-uncased"),
                num_labels=model_config.get("encoder.num_labels", 9),
                dropout=model_config.get("encoder.dropout", 0.1)
            )
        else:
            model = MultimodalNERModel(
                text_model_name=model_config.get("text_encoder.name", "bert-base-uncased"),
                vision_model_name=model_config.get("vision_encoder.name", "ViT-B/32"),
                num_labels=model_config.get("text_encoder.num_labels", 9),
                fusion_method=model_config.get("fusion.method", "cross_attention"),
                dropout=model_config.get("fusion.dropout", 0.1)
            )
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def _initialize_data_loader(self) -> DataLoader:
        """Initialize test data loader."""
        data_config = self.config.get("data", {})
        
        test_dataset = MultimodalNERDataset(
            data_path=data_config.get("test_path", "data/annotations/test.json"),
            tokenizer_name=self.config.get("model.text_encoder.name", "bert-base-uncased"),
            max_length=self.config.get("model.text_encoder.max_length", 512),
            image_size=self.config.get("model.vision_encoder.image_size", 224),
            split="test"
        )
        
        test_loader = create_dataloader(
            test_dataset,
            batch_size=self.config.get("training.batch_size", 16),
            shuffle=False,
            num_workers=4
        )
        
        logger.info(f"Test data loader initialized with {len(test_dataset)} samples")
        return test_loader
    
    def evaluate(self) -> Dict[str, float]:
        """Run comprehensive evaluation."""
        logger.info("Starting evaluation...")
        
        all_predictions = []
        all_labels = []
        all_attention_masks = []
        all_input_ids = []
        all_images = []
        all_predicted_entities = []
        all_visual_entities = []
        all_text_features = []
        all_vision_features = []
        
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.config.get("model.name") == "text_ner":
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                else:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        images=batch["images"],
                        labels=batch["labels"]
                    )
                
                loss = outputs.loss
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # Store for evaluation
                total_loss += loss.item()
                all_predictions.append(predictions)
                all_labels.append(batch["labels"])
                all_attention_masks.append(batch["attention_mask"])
                all_input_ids.append(batch["input_ids"])
                
                if "images" in batch:
                    all_images.append(batch["images"])
                
                # Store entities
                all_predicted_entities.append(batch.get("entities", []))
                all_visual_entities.append(batch.get("visual_entities", []))
                
                # Store features for alignment metrics
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    all_text_features.append(outputs.hidden_states[-1])
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)
        all_input_ids = torch.cat(all_input_ids, dim=0)
        
        if all_images:
            all_images = torch.cat(all_images, dim=0)
        
        if all_text_features:
            all_text_features = torch.cat(all_text_features, dim=0)
        
        # Create dummy vision features for alignment metrics
        if all_text_features.size(0) > 0:
            all_vision_features = torch.randn(all_text_features.size(0), 512).to(self.device)
        
        # Compute metrics
        metrics = self.evaluator.evaluate(
            predictions=all_predictions,
            labels=all_labels,
            attention_mask=all_attention_masks,
            input_ids=all_input_ids,
            images=all_images[0] if all_images else None,
            text_features=all_text_features[0] if all_text_features else torch.zeros(1, 1, 768),
            vision_features=all_vision_features[0] if all_vision_features else torch.zeros(1, 512),
            predicted_entities=all_predicted_entities,
            visual_entities=all_visual_entities,
            tokenizer=self.test_loader.dataset.tokenizer
        )
        
        metrics["test_loss"] = total_loss / len(self.test_loader)
        
        logger.info("Evaluation completed")
        return metrics
    
    def save_results(self, metrics: Dict[str, float], output_dir: str):
        """Save evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        with open(output_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Create metrics dashboard
        fig = create_metrics_dashboard(metrics)
        fig.savefig(output_path / "metrics_dashboard.png", dpi=300, bbox_inches='tight')
        
        # Save detailed report
        report_path = output_path / "evaluation_report.txt"
        with open(report_path, "w") as f:
            f.write("Multimodal NER Evaluation Report\n")
            f.write("=" * 40 + "\n\n")
            
            f.write("NER Metrics:\n")
            f.write(f"  Token F1: {metrics.get('token_f1', 0):.4f}\n")
            f.write(f"  Entity F1: {metrics.get('entity_f1', 0):.4f}\n")
            f.write(f"  Token Precision: {metrics.get('token_precision', 0):.4f}\n")
            f.write(f"  Token Recall: {metrics.get('token_recall', 0):.4f}\n")
            f.write(f"  Entity Precision: {metrics.get('entity_precision', 0):.4f}\n")
            f.write(f"  Entity Recall: {metrics.get('entity_recall', 0):.4f}\n\n")
            
            f.write("Visual Grounding Metrics:\n")
            f.write(f"  Visual Grounding F1: {metrics.get('visual_grounding_f1', 0):.4f}\n")
            f.write(f"  Visual Grounding Precision: {metrics.get('visual_grounding_precision', 0):.4f}\n")
            f.write(f"  Visual Grounding Recall: {metrics.get('visual_grounding_recall', 0):.4f}\n")
            f.write(f"  Visual Grounding Matches: {metrics.get('visual_grounding_matches', 0)}\n\n")
            
            f.write("Cross-Modal Alignment:\n")
            f.write(f"  Cross-Modal Similarity: {metrics.get('cross_modal_similarity', 0):.4f}\n")
            f.write(f"  Cross-Modal Alignment Score: {metrics.get('cross_modal_alignment_score', 0):.4f}\n\n")
            
            f.write(f"Test Loss: {metrics.get('test_loss', 0):.4f}\n")
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate multimodal NER model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MultimodalNEREvaluator(args.config, args.model)
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print results
    logger.info("Evaluation Results:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save results
    evaluator.save_results(metrics, args.output)


if __name__ == "__main__":
    main()
