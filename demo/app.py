"""Interactive demo for multimodal NER using Streamlit."""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import logging
from pathlib import Path
import json
from typing import Dict, List, Any, Optional

from src.models.multimodal_ner import MultimodalNERModel, TextNERModel
from src.utils.device import setup_device, set_seed
from src.utils.config import load_config
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Multimodal Named Entity Recognition",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .entity-highlight {
        background-color: #ffeb3b;
        padding: 0.2rem 0.4rem;
        border-radius: 0.3rem;
        margin: 0.1rem;
        display: inline-block;
    }
    .person { background-color: #e3f2fd; }
    .org { background-color: #f3e5f5; }
    .loc { background-color: #e8f5e8; }
    .misc { background-color: #fff3e0; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "device" not in st.session_state:
    st.session_state.device = None


@st.cache_resource
def load_model_and_tokenizer(model_type: str = "multimodal"):
    """Load model and tokenizer with caching."""
    try:
        # Setup device
        device = setup_device()
        set_seed(42, True)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Load model based on type
        if model_type == "multimodal":
            model = MultimodalNERModel.from_pretrained(
                text_model_name="bert-base-uncased",
                vision_model_name="ViT-B/32"
            )
        else:
            model = TextNERModel(
                model_name="bert-base-uncased",
                num_labels=9,
                dropout=0.1
            )
        
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        return model, tokenizer, device
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None, None, None


def preprocess_image(image: Image.Image, size: int = 224) -> torch.Tensor:
    """Preprocess image for model input."""
    # Resize image
    image = image.resize((size, size))
    
    # Convert to tensor and normalize
    img_array = np.array(image)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    img_tensor = (img_tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
    
    return img_tensor.unsqueeze(0)  # Add batch dimension


def predict_entities(text: str, image: Optional[Image.Image] = None, model_type: str = "multimodal") -> Dict[str, Any]:
    """Predict entities from text and optionally image."""
    if not st.session_state.model_loaded:
        return {"error": "Model not loaded"}
    
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    device = st.session_state.device
    
    try:
        # Tokenize text
        encoding = tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        # Prepare inputs based on model type
        if model_type == "multimodal" and image is not None:
            image_tensor = preprocess_image(image).to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=image_tensor
            )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get predictions
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Extract entities
        entities = extract_entities_from_predictions(
            predictions[0], input_ids[0], attention_mask[0], tokenizer
        )
        
        return {
            "entities": entities,
            "logits": outputs.logits[0].cpu().numpy(),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": str(e)}


def extract_entities_from_predictions(
    predictions: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer
) -> List[Dict[str, Any]]:
    """Extract entities from model predictions."""
    # Label mapping
    id2label = {
        0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
        5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"
    }
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    preds = predictions.cpu().numpy()
    mask = attention_mask.cpu().numpy()
    
    entities = []
    current_entity = None
    
    for i, (token, pred, m) in enumerate(zip(tokens, preds, mask)):
        if m == 0:  # Padding token
            break
        
        label = id2label.get(pred, "O")
        
        if label.startswith("B-"):
            # Start new entity
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "text": token,
                "label": label[2:],
                "start": i,
                "end": i + 1,
                "confidence": 0.8  # Placeholder confidence
            }
        elif label.startswith("I-") and current_entity:
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


def highlight_entities(text: str, entities: List[Dict[str, Any]]) -> str:
    """Create HTML with highlighted entities."""
    highlighted_text = text
    
    # Sort entities by start position (reverse order to avoid index issues)
    entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)
    
    for entity in entities_sorted:
        start = entity["start"]
        end = entity["end"]
        label = entity["label"].lower()
        
        # Create highlighted span
        span = f'<span class="entity-highlight {label}">{entity["text"]}</span>'
        
        # Replace text with highlighted version
        highlighted_text = highlighted_text[:start] + span + highlighted_text[end:]
    
    return highlighted_text


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🔍 Multimodal Named Entity Recognition</h1>', unsafe_allow_html=True)
    
    # Sidebar for model selection
    st.sidebar.title("Model Configuration")
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["multimodal", "text_only"],
        help="Choose between multimodal (text + image) or text-only NER"
    )
    
    # Load model button
    if st.sidebar.button("Load Model", type="primary"):
        with st.spinner("Loading model..."):
            model, tokenizer, device = load_model_and_tokenizer(model_type)
            
            if model is not None:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.device = device
                st.session_state.model_loaded = True
                st.sidebar.success("Model loaded successfully!")
            else:
                st.sidebar.error("Failed to load model")
    
    # Main content
    if not st.session_state.model_loaded:
        st.info("Please load a model from the sidebar to start.")
        
        # Show example
        st.markdown("### Example Usage")
        st.code("""
        Text: "Elon Musk founded SpaceX in California."
        
        Expected Entities:
        - Elon Musk (PERSON)
        - SpaceX (ORGANIZATION)  
        - California (LOCATION)
        """)
        return
    
    # Input section
    st.markdown("## Input")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text for entity recognition:",
            value="Elon Musk founded SpaceX in California.",
            height=100
        )
    
    with col2:
        if model_type == "multimodal":
            uploaded_image = st.file_uploader(
                "Upload an image (optional):",
                type=["jpg", "jpeg", "png"],
                help="Upload an image to enhance entity recognition"
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            else:
                image = None
        else:
            image = None
            st.info("Text-only mode selected")
    
    # Prediction button
    if st.button("Extract Entities", type="primary"):
        if text_input.strip():
            with st.spinner("Processing..."):
                result = predict_entities(text_input, image, model_type)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    entities = result["entities"]
                    
                    # Display results
                    st.markdown("## Results")
                    
                    if entities:
                        # Highlighted text
                        st.markdown("### Highlighted Text")
                        highlighted_html = highlight_entities(text_input, entities)
                        st.markdown(highlighted_html, unsafe_allow_html=True)
                        
                        # Entity table
                        st.markdown("### Detected Entities")
                        
                        entity_data = []
                        for entity in entities:
                            entity_data.append({
                                "Entity": entity["text"],
                                "Type": entity["label"],
                                "Position": f"{entity['start']}-{entity['end']}",
                                "Confidence": f"{entity['confidence']:.2f}"
                            })
                        
                        st.table(entity_data)
                        
                        # Metrics
                        st.markdown("### Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Entities", len(entities))
                        
                        with col2:
                            entity_types = set(e["label"] for e in entities)
                            st.metric("Entity Types", len(entity_types))
                        
                        with col3:
                            if model_type == "multimodal" and image:
                                st.metric("Mode", "Multimodal")
                            else:
                                st.metric("Mode", "Text-only")
                    else:
                        st.info("No entities detected in the input text.")
        else:
            st.warning("Please enter some text to analyze.")
    
    # Examples section
    st.markdown("## Examples")
    
    example_texts = [
        "Apple Inc. is located in Cupertino, California.",
        "Tim Cook is the CEO of Apple.",
        "Microsoft was founded by Bill Gates in Seattle.",
        "Tesla Motors is based in Palo Alto, California.",
        "Google's headquarters are in Mountain View, California."
    ]
    
    selected_example = st.selectbox("Choose an example:", example_texts)
    
    if st.button("Use Example"):
        st.session_state.example_text = selected_example
        st.rerun()
    
    if hasattr(st.session_state, 'example_text'):
        st.text_area("Example text:", value=st.session_state.example_text, height=100)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About This Demo
    
    This interactive demo showcases **Multimodal Named Entity Recognition (NER)** capabilities:
    
    - **Text-based NER**: Extracts named entities from text using transformer models
    - **Multimodal NER**: Combines text and visual information for enhanced entity recognition
    - **Entity Types**: Person (PER), Organization (ORG), Location (LOC), Miscellaneous (MISC)
    
    **Note**: This is a research/educational demo. Results should not be used for production applications without proper validation.
    """)


if __name__ == "__main__":
    main()
