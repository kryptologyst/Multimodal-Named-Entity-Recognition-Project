"""Training script for multimodal NER."""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb

from src.utils.config import load_config
from src.utils.device import setup_device, set_seed, get_device_info
from src.data.loaders import MultimodalNERDataset, create_dataloader
from src.models.multimodal_ner import MultimodalNERModel, TextNERModel
from src.eval.metrics import MultimodalNEREvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalNERTrainer:
    """Trainer for multimodal NER models."""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        
        # Setup device and reproducibility
        self.device = setup_device(
            fallback_order=self.config.get("device.fallback_order", ["cuda", "mps", "cpu"])
        )
        set_seed(
            self.config.get("seed", 42),
            self.config.get("deterministic", True)
        )
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._initialize_data_loaders()
        
        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = self._initialize_optimizer()
        
        # Initialize evaluator
        self.evaluator = MultimodalNEREvaluator(self.model.text_encoder.label2id)
        
        # Training state
        self.current_epoch = 0
        self.best_f1 = 0.0
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Device info: {get_device_info()}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.get("logging.log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup wandb if enabled
        if self.config.get("logging.wandb.enabled", False):
            wandb.init(
                project=self.config.get("logging.wandb.project", "multimodal-ner"),
                config=self.config.to_dict()
            )
    
    def _initialize_model(self) -> nn.Module:
        """Initialize model."""
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
        
        model = model.to(self.device)
        return model
    
    def _initialize_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Initialize data loaders."""
        data_config = self.config.get("data", {})
        training_config = self.config.get("training", {})
        
        # Training dataset
        train_dataset = MultimodalNERDataset(
            data_path=data_config.get("train_path", "data/annotations/train.json"),
            tokenizer_name=self.config.get("model.text_encoder.name", "bert-base-uncased"),
            max_length=self.config.get("model.text_encoder.max_length", 512),
            image_size=self.config.get("model.vision_encoder.image_size", 224),
            split="train"
        )
        
        # Validation dataset
        val_dataset = MultimodalNERDataset(
            data_path=data_config.get("val_path", "data/annotations/val.json"),
            tokenizer_name=self.config.get("model.text_encoder.name", "bert-base-uncased"),
            max_length=self.config.get("model.text_encoder.max_length", 512),
            image_size=self.config.get("model.vision_encoder.image_size", 224),
            split="val"
        )
        
        # Create data loaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=training_config.get("batch_size", 16),
            shuffle=True,
            num_workers=4
        )
        
        val_loader = create_dataloader(
            val_dataset,
            batch_size=training_config.get("batch_size", 16),
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def _initialize_optimizer(self) -> tuple[torch.optim.Optimizer, Any]:
        """Initialize optimizer and scheduler."""
        training_config = self.config.get("training", {})
        
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=training_config.get("learning_rate", 2e-5),
            weight_decay=training_config.get("weight_decay", 0.01)
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * training_config.get("num_epochs", 10)
        warmup_steps = training_config.get("warmup_steps", 500)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return optimizer, scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
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
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("training.gradient_clip_norm", 1.0)
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_attention_masks = []
        all_input_ids = []
        all_images = []
        all_text_features = []
        all_vision_features = []
        all_predicted_entities = []
        all_visual_entities = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
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
                
                # Store entities for evaluation
                all_predicted_entities.append(batch.get("entities", []))
                all_visual_entities.append(batch.get("visual_entities", []))
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)
        all_input_ids = torch.cat(all_input_ids, dim=0)
        
        if all_images:
            all_images = torch.cat(all_images, dim=0)
        
        # Compute metrics
        metrics = self.evaluator.evaluate(
            predictions=all_predictions,
            labels=all_labels,
            attention_mask=all_attention_masks,
            input_ids=all_input_ids,
            images=all_images[0] if all_images else None,
            text_features=torch.zeros(1, 1, 768),  # Placeholder
            vision_features=torch.zeros(1, 512),   # Placeholder
            predicted_entities=all_predicted_entities,
            visual_entities=all_visual_entities,
            tokenizer=self.train_loader.dataset.tokenizer
        )
        
        metrics["val_loss"] = total_loss / len(self.val_loader)
        return metrics
    
    def train(self):
        """Main training loop."""
        training_config = self.config.get("training", {})
        num_epochs = training_config.get("num_epochs", 10)
        patience = training_config.get("patience", 3)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            all_metrics = {**train_metrics, **val_metrics}
            logger.info(f"Epoch {epoch}: {all_metrics}")
            
            if self.config.get("logging.wandb.enabled", False):
                wandb.log(all_metrics, step=epoch)
            
            # Check for improvement
            current_f1 = val_metrics.get("entity_f1", 0.0)
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.patience_counter = 0
                self._save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info(f"Training completed. Best F1: {self.best_f1:.4f}")
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.get("checkpointing.save_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_f1": self.best_f1,
            "config": self.config.to_dict()
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / "latest.pt")
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best.pt")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train multimodal NER model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = MultimodalNERTrainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # TODO: Implement checkpoint loading
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
