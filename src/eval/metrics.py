"""Evaluation metrics for multimodal NER."""

import logging
from typing import Dict, List, Tuple, Any, Optional
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report as seqeval_report

logger = logging.getLogger(__name__)


class NERMetrics:
    """Metrics for Named Entity Recognition evaluation."""
    
    def __init__(self, label2id: Dict[str, int]):
        """
        Initialize NER metrics.
        
        Args:
            label2id: Label to ID mapping
        """
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        
    def compute_token_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute token-level metrics.
        
        Args:
            predictions: Predicted token labels
            labels: Ground truth token labels
            attention_mask: Attention mask
            
        Returns:
            Dictionary of metrics
        """
        # Flatten predictions and labels
        preds_flat = predictions.view(-1)
        labels_flat = labels.view(-1)
        mask_flat = attention_mask.view(-1)
        
        # Remove padding tokens
        valid_indices = mask_flat == 1
        preds_valid = preds_flat[valid_indices]
        labels_valid = labels_flat[valid_indices]
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_valid.cpu().numpy(),
            preds_valid.cpu().numpy(),
            average='weighted',
            zero_division=0
        )
        
        return {
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": f1
        }
    
    def compute_entity_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        tokenizer
    ) -> Dict[str, float]:
        """
        Compute entity-level metrics.
        
        Args:
            predictions: Predicted token labels
            labels: Ground truth token labels
            attention_mask: Attention mask
            input_ids: Input token IDs
            tokenizer: Tokenizer for decoding
            
        Returns:
            Dictionary of metrics
        """
        batch_size = predictions.size(0)
        all_pred_entities = []
        all_true_entities = []
        
        for i in range(batch_size):
            # Get tokens and predictions for this sample
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
            preds = predictions[i].cpu().numpy()
            true_labels = labels[i].cpu().numpy()
            mask = attention_mask[i].cpu().numpy()
            
            # Extract predicted entities
            pred_entities = self._extract_entities(tokens, preds, mask)
            true_entities = self._extract_entities(tokens, true_labels, mask)
            
            all_pred_entities.append(pred_entities)
            all_true_entities.append(true_entities)
        
        # Convert to seqeval format
        pred_labels = self._entities_to_labels(all_pred_entities)
        true_labels = self._entities_to_labels(all_true_entities)
        
        # Compute entity-level metrics
        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        
        return {
            "entity_precision": precision,
            "entity_recall": recall,
            "entity_f1": f1
        }
    
    def _extract_entities(
        self,
        tokens: List[str],
        labels: np.ndarray,
        mask: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Extract entities from token predictions."""
        entities = []
        current_entity = None
        
        for i, (token, label, m) in enumerate(zip(tokens, labels, mask)):
            if m == 0:  # Padding token
                break
            
            label_str = self.id2label.get(label, "O")
            
            if label_str.startswith("B-"):
                # Start new entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "label": label_str[2:],
                    "start": i,
                    "end": i + 1
                }
            elif label_str.startswith("I-") and current_entity:
                # Continue entity
                current_entity["text"] += token
                current_entity["end"] = i + 1
            else:
                # End entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def _entities_to_labels(self, entities_list: List[List[Dict[str, Any]]]) -> List[List[str]]:
        """Convert entities to seqeval label format."""
        labels = []
        for entities in entities_list:
            # Create label sequence for this sample
            sample_labels = []
            for entity in entities:
                # Add B- and I- labels for entity
                sample_labels.append(f"B-{entity['label']}")
                for _ in range(entity['start'] + 1, entity['end']):
                    sample_labels.append(f"I-{entity['label']}")
            labels.append(sample_labels)
        return labels


class VisualGroundingMetrics:
    """Metrics for visual grounding evaluation."""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize visual grounding metrics.
        
        Args:
            iou_threshold: IoU threshold for positive detection
        """
        self.iou_threshold = iou_threshold
    
    def compute_grounding_metrics(
        self,
        predicted_entities: List[Dict[str, Any]],
        visual_entities: List[Dict[str, Any]],
        text_entities: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute visual grounding metrics.
        
        Args:
            predicted_entities: Predicted text entities
            visual_entities: Visual entities with bounding boxes
            text_entities: Ground truth text entities
            
        Returns:
            Dictionary of grounding metrics
        """
        # Match predicted entities with visual entities
        matches = self._match_entities(predicted_entities, visual_entities)
        
        # Compute metrics
        total_predicted = len(predicted_entities)
        total_visual = len(visual_entities)
        matched = len(matches)
        
        precision = matched / total_predicted if total_predicted > 0 else 0.0
        recall = matched / total_visual if total_visual > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "visual_grounding_precision": precision,
            "visual_grounding_recall": recall,
            "visual_grounding_f1": f1,
            "visual_grounding_matches": matched
        }
    
    def _match_entities(
        self,
        text_entities: List[Dict[str, Any]],
        visual_entities: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Match text entities with visual entities."""
        matches = []
        used_visual = set()
        
        for text_entity in text_entities:
            best_match = None
            best_score = 0.0
            
            for i, visual_entity in enumerate(visual_entities):
                if i in used_visual:
                    continue
                
                # Simple label matching (can be enhanced with semantic similarity)
                if text_entity["label"].lower() in visual_entity["label"].lower():
                    score = visual_entity.get("confidence", 0.5)
                    if score > best_score:
                        best_score = score
                        best_match = (text_entity, visual_entity)
            
            if best_match and best_score > 0.3:  # Confidence threshold
                matches.append(best_match)
                used_visual.add(visual_entities.index(best_match[1]))
        
        return matches


class CrossModalAlignmentMetrics:
    """Metrics for cross-modal alignment evaluation."""
    
    def compute_alignment_metrics(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute cross-modal alignment metrics.
        
        Args:
            text_features: Text features
            vision_features: Vision features
            attention_mask: Attention mask
            
        Returns:
            Dictionary of alignment metrics
        """
        batch_size = text_features.size(0)
        
        # Compute cosine similarity between text and vision features
        similarities = []
        
        for i in range(batch_size):
            # Get valid text features (remove padding)
            valid_mask = attention_mask[i] == 1
            text_feat = text_features[i][valid_mask]  # [seq_len, dim]
            vision_feat = vision_features[i].unsqueeze(0)  # [1, dim]
            
            # Compute similarities
            sim = F.cosine_similarity(
                text_feat.unsqueeze(1),  # [seq_len, 1, dim]
                vision_feat.unsqueeze(0),  # [1, 1, dim]
                dim=-1
            ).squeeze(-1)  # [seq_len]
            
            similarities.append(sim.mean().item())
        
        avg_similarity = np.mean(similarities)
        
        return {
            "cross_modal_similarity": avg_similarity,
            "cross_modal_alignment_score": avg_similarity
        }


class MultimodalNEREvaluator:
    """Comprehensive evaluator for multimodal NER."""
    
    def __init__(self, label2id: Dict[str, int]):
        """
        Initialize evaluator.
        
        Args:
            label2id: Label to ID mapping
        """
        self.ner_metrics = NERMetrics(label2id)
        self.visual_metrics = VisualGroundingMetrics()
        self.alignment_metrics = CrossModalAlignmentMetrics()
    
    def evaluate(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        predicted_entities: List[List[Dict[str, Any]]],
        visual_entities: List[List[Dict[str, Any]]],
        tokenizer
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation.
        
        Args:
            predictions: Predicted token labels
            labels: Ground truth token labels
            attention_mask: Attention mask
            input_ids: Input token IDs
            images: Input images
            text_features: Text features
            vision_features: Vision features
            predicted_entities: Predicted entities
            visual_entities: Visual entities
            tokenizer: Tokenizer
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Token-level metrics
        token_metrics = self.ner_metrics.compute_token_metrics(
            predictions, labels, attention_mask
        )
        metrics.update(token_metrics)
        
        # Entity-level metrics
        entity_metrics = self.ner_metrics.compute_entity_metrics(
            predictions, labels, attention_mask, input_ids, tokenizer
        )
        metrics.update(entity_metrics)
        
        # Visual grounding metrics
        grounding_metrics = self.visual_metrics.compute_grounding_metrics(
            predicted_entities[0] if predicted_entities else [],
            visual_entities[0] if visual_entities else [],
            []
        )
        metrics.update(grounding_metrics)
        
        # Cross-modal alignment metrics
        alignment_metrics = self.alignment_metrics.compute_alignment_metrics(
            text_features, vision_features, attention_mask
        )
        metrics.update(alignment_metrics)
        
        return metrics
