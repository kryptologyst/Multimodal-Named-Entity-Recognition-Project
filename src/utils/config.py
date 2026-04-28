"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from omegaconf import OmegaConf
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = {}
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
        
        logger.info(f"Loaded configuration from {config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """
        Save configuration to file.
        
        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {output_path}")
    
    def update(self, other_config: Dict[str, Any]) -> None:
        """
        Update configuration with another config.
        
        Args:
            other_config: Configuration to merge
        """
        self.config.update(other_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.copy()


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)


def create_default_config() -> Config:
    """Create default configuration."""
    default_config = {
        "model": {
            "name": "multimodal_ner",
            "text_encoder": {
                "name": "bert-base-uncased",
                "max_length": 512,
                "num_labels": 9
            },
            "vision_encoder": {
                "name": "openai/clip-vit-base-patch32",
                "image_size": 224
            },
            "fusion": {
                "method": "cross_attention",
                "hidden_size": 768,
                "num_attention_heads": 12,
                "dropout": 0.1
            }
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 2e-5,
            "num_epochs": 10,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "use_amp": True,
            "patience": 3,
            "min_delta": 0.001
        },
        "data": {
            "train_path": "data/annotations/train.json",
            "val_path": "data/annotations/val.json",
            "test_path": "data/annotations/test.json"
        },
        "evaluation": {
            "metrics": ["token_f1", "entity_f1", "visual_grounding"],
            "eval_steps": 500,
            "save_steps": 1000
        },
        "logging": {
            "log_level": "INFO",
            "log_dir": "logs",
            "tensorboard": True
        },
        "checkpointing": {
            "save_dir": "checkpoints",
            "save_total_limit": 3,
            "load_best_model_at_end": True
        },
        "device": {
            "auto_detect": True,
            "fallback_order": ["cuda", "mps", "cpu"]
        },
        "seed": 42,
        "deterministic": True
    }
    
    config = Config()
    config.config = default_config
    return config
