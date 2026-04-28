"""Multimodal Named Entity Recognition models."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertModel, BertConfig
)
from transformers.modeling_outputs import TokenClassifierOutput
import open_clip

logger = logging.getLogger(__name__)


class TextNERModel(nn.Module):
    """Text-based Named Entity Recognition model."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 9,
        dropout: float = 0.1
    ):
        """
        Initialize text NER model.
        
        Args:
            model_name: Name of pre-trained model
            num_labels: Number of NER labels
            dropout: Dropout rate
        """
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        self.num_labels = num_labels
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> TokenClassifierOutput:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels
            
        Returns:
            TokenClassifierOutput
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


class VisionEncoder(nn.Module):
    """Vision encoder using CLIP."""
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        pretrained: str = "openai"
    ):
        """
        Initialize vision encoder.
        
        Args:
            model_name: CLIP model name
            pretrained: Pretrained weights
        """
        super().__init__()
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = self.model.visual
        
        # Freeze CLIP parameters
        for param in self.visual.parameters():
            param.requires_grad = False
            
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Input images
            
        Returns:
            Visual features
        """
        return self.visual(images)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module."""
    
    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 512,
        hidden_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        """
        Initialize cross-attention fusion.
        
        Args:
            text_dim: Text feature dimension
            vision_dim: Vision feature dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            text_features: Text features [batch_size, seq_len, text_dim]
            vision_features: Vision features [batch_size, vision_dim]
            text_mask: Text attention mask
            
        Returns:
            Fused features
        """
        # Project features to same dimension
        text_proj = self.text_proj(text_features)
        vision_proj = self.vision_proj(vision_features).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Cross-attention: text attends to vision
        attn_output, _ = self.cross_attention(
            query=text_proj,
            key=vision_proj,
            value=vision_proj,
            key_padding_mask=None
        )
        
        # Residual connection and layer norm
        text_features = self.norm1(text_proj + self.dropout(attn_output))
        
        return text_features


class MultimodalNERModel(nn.Module):
    """Multimodal Named Entity Recognition model."""
    
    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        vision_model_name: str = "ViT-B/32",
        num_labels: int = 9,
        fusion_method: str = "cross_attention",
        dropout: float = 0.1
    ):
        """
        Initialize multimodal NER model.
        
        Args:
            text_model_name: Text model name
            vision_model_name: Vision model name
            num_labels: Number of NER labels
            fusion_method: Fusion method (late_fusion, early_fusion, cross_attention)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # Text encoder
        self.text_encoder = TextNERModel(text_model_name, num_labels, dropout)
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(vision_model_name)
        
        # Fusion module
        if fusion_method == "cross_attention":
            self.fusion = CrossAttentionFusion(
                text_dim=768,
                vision_dim=512,
                hidden_dim=768,
                dropout=dropout
            )
        elif fusion_method == "late_fusion":
            self.fusion = nn.Linear(768 + 512, 768)
        elif fusion_method == "early_fusion":
            self.fusion = nn.Linear(768 + 512, 768)
        
        # Final classifier
        self.classifier = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(dropout)
        
        self.num_labels = num_labels
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> TokenClassifierOutput:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            images: Input images
            labels: Ground truth labels
            
        Returns:
            TokenClassifierOutput
        """
        # Get text features
        text_outputs = self.text_encoder.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state  # [batch_size, seq_len, 768]
        
        # Get vision features
        vision_features = self.vision_encoder(images)  # [batch_size, 512]
        
        # Fuse features
        if self.fusion_method == "cross_attention":
            fused_features = self.fusion(text_features, vision_features, attention_mask)
        elif self.fusion_method == "late_fusion":
            # Repeat vision features for each token
            vision_repeated = vision_features.unsqueeze(1).repeat(1, text_features.size(1), 1)
            # Concatenate text and vision features
            combined = torch.cat([text_features, vision_repeated], dim=-1)
            fused_features = self.fusion(combined)
        elif self.fusion_method == "early_fusion":
            # Global average pooling of text features
            text_pooled = torch.mean(text_features, dim=1)  # [batch_size, 768]
            # Concatenate pooled text and vision features
            combined = torch.cat([text_pooled, vision_features], dim=-1)
            fused_features = self.fusion(combined)
            # Repeat for each token
            fused_features = fused_features.unsqueeze(1).repeat(1, text_features.size(1), 1)
        
        # Apply dropout and classify
        fused_features = self.dropout(fused_features)
        logits = self.classifier(fused_features)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions
        )
    
    @classmethod
    def from_pretrained(
        cls,
        text_model_name: str = "bert-base-uncased",
        vision_model_name: str = "ViT-B/32",
        **kwargs
    ):
        """Create model from pre-trained weights."""
        return cls(text_model_name, vision_model_name, **kwargs)
    
    def get_entity_predictions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        tokenizer
    ) -> List[Dict[str, Any]]:
        """
        Get entity predictions with text.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            images: Input images
            tokenizer: Tokenizer for decoding
            
        Returns:
            List of predicted entities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, images)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Convert predictions to entities
        entities = []
        batch_size = input_ids.size(0)
        
        for i in range(batch_size):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
            preds = predictions[i].cpu().numpy()
            mask = attention_mask[i].cpu().numpy()
            
            current_entity = None
            for j, (token, pred, m) in enumerate(zip(tokens, preds, mask)):
                if m == 0:  # Padding token
                    break
                
                label = self.text_encoder.id2label.get(pred, "O")
                
                if label.startswith("B-"):
                    # Start new entity
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "text": token,
                        "label": label[2:],
                        "start": j,
                        "end": j + 1
                    }
                elif label.startswith("I-") and current_entity:
                    # Continue entity
                    current_entity["text"] += token
                    current_entity["end"] = j + 1
                else:
                    # End entity
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            
            if current_entity:
                entities.append(current_entity)
        
        return entities
