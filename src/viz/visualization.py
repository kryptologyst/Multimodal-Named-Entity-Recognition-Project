"""Visualization utilities for multimodal NER."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import cv2


class NERVisualizer:
    """Visualization utilities for NER results."""
    
    def __init__(self, label_colors: Optional[Dict[str, str]] = None):
        """
        Initialize NER visualizer.
        
        Args:
            label_colors: Custom colors for entity labels
        """
        self.label_colors = label_colors or {
            "PER": "#e3f2fd",  # Light blue
            "ORG": "#f3e5f5",  # Light purple
            "LOC": "#e8f5e8",  # Light green
            "MISC": "#fff3e0"  # Light orange
        }
    
    def visualize_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        entities: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize attention heatmap for entity recognition.
        
        Args:
            attention_weights: Attention weights [num_heads, seq_len, seq_len]
            tokens: Input tokens
            entities: Detected entities
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Average attention across heads
        avg_attention = attention_weights.mean(dim=0).cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot heatmap
        im = ax.imshow(avg_attention, cmap='Blues', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
        
        # Highlight entity regions
        for entity in entities:
            start, end = entity['start'], entity['end']
            # Draw rectangle around entity
            rect = plt.Rectangle(
                (start-0.5, start-0.5), 
                end-start, 
                end-start,
                fill=False, 
                edgecolor='red', 
                linewidth=2
            )
            ax.add_patch(rect)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        plt.title('Attention Heatmap for Entity Recognition')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_entity_distribution(
        self,
        entities: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize distribution of entity types.
        
        Args:
            entities: List of detected entities
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Count entity types
        entity_counts = {}
        for entity in entities:
            label = entity['label']
            entity_counts[label] = entity_counts.get(label, 0) + 1
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        labels = list(entity_counts.keys())
        counts = list(entity_counts.values())
        colors = [self.label_colors.get(label, "#cccccc") for label in labels]
        
        bars = ax1.bar(labels, counts, color=colors)
        ax1.set_title('Entity Type Distribution')
        ax1.set_xlabel('Entity Type')
        ax1.set_ylabel('Count')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # Pie chart
        if counts:
            ax2.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%')
            ax2.set_title('Entity Type Proportions')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_cross_modal_alignment(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        tokens: List[str],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize cross-modal alignment between text and vision features.
        
        Args:
            text_features: Text features [seq_len, dim]
            vision_features: Vision features [dim]
            tokens: Input tokens
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Compute similarities
        text_norm = F.normalize(text_features, p=2, dim=-1)
        vision_norm = F.normalize(vision_features, p=2, dim=-1)
        
        similarities = F.cosine_similarity(
            text_norm, vision_norm.unsqueeze(0), dim=-1
        ).cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot similarities
        bars = ax.bar(range(len(tokens)), similarities, 
                     color=plt.cm.RdYlBu_r(similarities))
        
        # Set labels
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Cross-Modal Alignment: Text-Vision Similarity')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, 
                                  norm=plt.Normalize(vmin=similarities.min(), 
                                                   vmax=similarities.max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Similarity Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ImageVisualizer:
    """Visualization utilities for images and visual entities."""
    
    def __init__(self):
        """Initialize image visualizer."""
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
    
    def visualize_visual_entities(
        self,
        image: Image.Image,
        visual_entities: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> Image.Image:
        """
        Visualize visual entities on image.
        
        Args:
            image: Input image
            visual_entities: Visual entities with bounding boxes
            save_path: Path to save the image
            
        Returns:
            Annotated image
        """
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes
        for i, entity in enumerate(visual_entities):
            bbox = entity['bbox']
            label = entity['label']
            confidence = entity.get('confidence', 0.0)
            
            # Get color
            color = self.colors[i % len(self.colors)]
            
            # Draw rectangle
            cv2.rectangle(img_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(img_cv, label_text, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Convert back to PIL
        annotated_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        if save_path:
            annotated_img.save(save_path)
        
        return annotated_img
    
    def create_entity_comparison_grid(
        self,
        images: List[Image.Image],
        entities_lists: List[List[Dict[str, Any]]],
        titles: List[str],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a grid comparing entity recognition across images.
        
        Args:
            images: List of images
            entities_lists: List of entity lists for each image
            titles: Titles for each subplot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_images = len(images)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (img, entities, title) in enumerate(zip(images, entities_lists, titles)):
            ax = axes[i]
            
            # Show image
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
            
            # Add entity annotations
            for entity in entities:
                bbox = entity['bbox']
                label = entity['label']
                
                # Draw rectangle
                rect = plt.Rectangle(
                    (bbox[0], bbox[1]), 
                    bbox[2] - bbox[0], 
                    bbox[3] - bbox[1],
                    fill=False, 
                    edgecolor='red', 
                    linewidth=2
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(bbox[0], bbox[1] - 5, label, 
                       color='red', fontsize=10, weight='bold')
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_metrics_dashboard(
    metrics: Dict[str, float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a dashboard showing evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Categorize metrics
    ner_metrics = {k: v for k, v in metrics.items() if 'token' in k or 'entity' in k}
    visual_metrics = {k: v for k, v in metrics.items() if 'visual' in k or 'grounding' in k}
    alignment_metrics = {k: v for k, v in metrics.items() if 'alignment' in k or 'similarity' in k}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # NER metrics
    if ner_metrics:
        ax1 = axes[0, 0]
        labels = list(ner_metrics.keys())
        values = list(ner_metrics.values())
        bars = ax1.bar(labels, values, color='skyblue')
        ax1.set_title('NER Metrics')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # Visual metrics
    if visual_metrics:
        ax2 = axes[0, 1]
        labels = list(visual_metrics.keys())
        values = list(visual_metrics.values())
        bars = ax2.bar(labels, values, color='lightgreen')
        ax2.set_title('Visual Grounding Metrics')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # Alignment metrics
    if alignment_metrics:
        ax3 = axes[1, 0]
        labels = list(alignment_metrics.keys())
        values = list(alignment_metrics.values())
        bars = ax3.bar(labels, values, color='lightcoral')
        ax3.set_title('Cross-Modal Alignment')
        ax3.set_ylabel('Score')
        
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # Overall summary
    ax4 = axes[1, 1]
    overall_score = np.mean(list(metrics.values()))
    ax4.pie([overall_score, 1 - overall_score], 
            labels=['Performance', 'Remaining'],
            colors=['lightblue', 'lightgray'],
            autopct='%1.1f%%')
    ax4.set_title(f'Overall Performance: {overall_score:.3f}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
