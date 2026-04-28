"""Utility functions for device management and reproducibility."""

import os
import random
from typing import Optional, List, Union
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def setup_device(
    device_preference: Optional[str] = None,
    fallback_order: List[str] = ["cuda", "mps", "cpu"]
) -> torch.device:
    """
    Setup device with automatic detection and fallback.
    
    Args:
        device_preference: Preferred device ("cuda", "mps", "cpu")
        fallback_order: Order of devices to try
        
    Returns:
        torch.device: Selected device
    """
    if device_preference:
        if device_preference == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif device_preference == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon MPS device")
        elif device_preference == "cpu":
            device = torch.device("cpu")
            logger.info("Using CPU device")
        else:
            logger.warning(f"Preferred device {device_preference} not available, falling back")
            device = _auto_detect_device(fallback_order)
    else:
        device = _auto_detect_device(fallback_order)
    
    return device


def _auto_detect_device(fallback_order: List[str]) -> torch.device:
    """Auto-detect the best available device."""
    for device_type in fallback_order:
        if device_type == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name()}")
            return device
        elif device_type == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Auto-detected Apple Silicon MPS device")
            return device
        elif device_type == "cpu":
            device = torch.device("cpu")
            logger.info("Auto-detected CPU device")
            return device
    
    # Fallback to CPU if nothing else works
    device = torch.device("cpu")
    logger.warning("No suitable device found, using CPU")
    return device


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variables for reproducibility
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        
        # Enable deterministic behavior in PyTorch
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    logger.info(f"Random seed set to {seed}, deterministic={deterministic}")


def get_device_info() -> dict:
    """Get information about available devices."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_count": torch.get_num_threads(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_memory_allocated": torch.cuda.memory_allocated(),
            "cuda_memory_reserved": torch.cuda.memory_reserved(),
        })
    
    return info


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")


def move_to_device(
    obj: Union[torch.Tensor, dict, list, tuple], 
    device: torch.device
) -> Union[torch.Tensor, dict, list, tuple]:
    """
    Move object to specified device.
    
    Args:
        obj: Object to move (tensor, dict, list, tuple)
        device: Target device
        
    Returns:
        Object moved to device
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    else:
        return obj
