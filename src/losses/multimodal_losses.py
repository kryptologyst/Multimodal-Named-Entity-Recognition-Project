"""Loss functions for multimodal NER training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class MultimodalNERLoss(nn.Module):
    """Combined loss function for multimodal NER."""
    
    def __init__(
        self,
        ner_weight: float = 1.0,
        alignment_weight: float = 0.1,
        grounding_weight: float = 0.1,
        label_smoothing: float = 0.0
    ):
        """
        Initialize multimodal NER loss.
        
        Args:
            ner_weight: Weight for NER loss
            alignment_weight: Weight for cross-modal alignment loss
            grounding_weight: Weight for visual grounding loss
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        
        self.ner_weight = ner_weight
        self.alignment_weight = alignment_weight
        self.grounding_weight = grounding_weight
        
        # NER loss with label smoothing
        self.ner_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Alignment loss (contrastive)
        self.alignment_loss = nn.MSELoss()
        
        # Grounding loss (IoU-based)
        self.grounding_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        ner_logits: torch.Tensor,
        ner_labels: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        grounding_logits: Optional[torch.Tensor] = None,
        grounding_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            ner_logits: NER prediction logits
            ner_labels: NER ground truth labels
            text_features: Text features for alignment
            vision_features: Vision features for alignment
            grounding_logits: Visual grounding logits
            grounding_labels: Visual grounding labels
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # NER loss
        ner_loss = self.ner_loss(ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1))
        losses["ner_loss"] = ner_loss
        
        total_loss = self.ner_weight * ner_loss
        
        # Cross-modal alignment loss
        if text_features is not None and vision_features is not None:
            # Compute cosine similarity between text and vision features
            text_norm = F.normalize(text_features, p=2, dim=-1)
            vision_norm = F.normalize(vision_features, p=2, dim=-1)
            
            # Global average pooling for alignment
            text_pooled = torch.mean(text_norm, dim=1)  # [batch_size, dim]
            vision_pooled = torch.mean(vision_norm, dim=1)  # [batch_size, dim]
            
            # Alignment loss (encourage high similarity)
            alignment_loss = self.alignment_loss(text_pooled, vision_pooled)
            losses["alignment_loss"] = alignment_loss
            total_loss += self.alignment_weight * alignment_loss
        
        # Visual grounding loss
        if grounding_logits is not None and grounding_labels is not None:
            grounding_loss = self.grounding_loss(grounding_logits, grounding_labels)
            losses["grounding_loss"] = grounding_loss
            total_loss += self.grounding_weight * grounding_loss
        
        losses["total_loss"] = total_loss
        return losses


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in NER."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            inputs: Input logits
            targets: Target labels
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for cross-modal alignment."""
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.5):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter
            margin: Margin for negative pairs
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            text_features: Text features
            vision_features: Vision features
            labels: Optional labels for supervised contrastive learning
            
        Returns:
            Contrastive loss
        """
        # Normalize features
        text_norm = F.normalize(text_features, p=2, dim=-1)
        vision_norm = F.normalize(vision_features, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(text_norm, vision_norm.T) / self.temperature
        
        if labels is not None:
            # Supervised contrastive learning
            batch_size = similarity.size(0)
            labels = labels.view(-1, 1)
            
            # Create positive mask
            positive_mask = torch.eq(labels, labels.T).float()
            
            # Compute loss
            exp_sim = torch.exp(similarity)
            sum_exp_sim = torch.sum(exp_sim, dim=1, keepdim=True)
            
            log_prob = similarity - torch.log(sum_exp_sim)
            mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
            
            loss = -mean_log_prob_pos.mean()
        else:
            # Unsupervised contrastive learning (InfoNCE)
            batch_size = similarity.size(0)
            labels = torch.arange(batch_size).to(similarity.device)
            
            loss = F.cross_entropy(similarity, labels)
        
        return loss
