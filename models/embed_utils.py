from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import clip
import re

# Initialize models lazily to avoid loading at import time
_text_model = None
_clip_model = None
_clip_preprocess = None

def get_text_model():
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _text_model



def get_clip_model():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device="cpu")
        _clip_model.eval()
    return _clip_model, _clip_preprocess

# Legacy ResNet preprocess transform removed - CLIP handles its own preprocessing

def get_text_embedding(text):
    """
    Get the embedding for a given text.
    """
    text_model = get_text_model()
    return text_model.encode(text)

# Legacy function removed - use get_image_embedding_simple() instead
# def get_image_embedding(image_input): ... [REMOVED]

def get_image_embedding_simple(image_input):
    """
    Get image embedding using CLIP that's compatible with text embeddings.
    CLIP produces 512-dimensional image embeddings.
    Fixed version to prevent kernel crashes.
    """
    try:
        # Import torch here to ensure it's available
        import torch
        import torch.nn.functional as F
        
        clip_model, clip_preprocess = get_clip_model()
        
        # Check if models are loaded properly
        if clip_model is None or clip_preprocess is None:
            raise ValueError("CLIP model not loaded properly")
        
        # Handle different input types with robust error handling
        try:
            if isinstance(image_input, str):
                # Handle file path
                image = Image.open(image_input).convert('RGB')
            elif hasattr(image_input, 'read'):
                # Handle file-like object (BytesIO, etc.)
                image = Image.open(image_input).convert('RGB')
            elif isinstance(image_input, Image.Image):
                # Handle PIL Image directly
                image = image_input.convert('RGB')
            else:
                # Try to treat as file-like object
                image = Image.open(image_input).convert('RGB')
        except Exception as img_error:
            print(f"Image loading failed: {img_error}")
            return np.zeros(512, dtype=np.float32)
        
        # Preprocess image for CLIP with dimension validation
        try:
            image_tensor = clip_preprocess(image)
            
            # Ensure tensor is properly formatted
            if not isinstance(image_tensor, torch.Tensor):
                print("Warning: CLIP preprocess didn't return tensor, converting...")
                image_tensor = transforms.ToTensor()(image_tensor)
            
            # Ensure correct dimensions: [1, 3, 224, 224] for CLIP
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            elif len(image_tensor.shape) != 4:
                raise ValueError(f"Unexpected tensor shape: {image_tensor.shape}")
            
            # Validate tensor dimensions match CLIP requirements
            if image_tensor.shape != torch.Size([1, 3, 224, 224]):
                print(f"Warning: Tensor shape {image_tensor.shape} doesn't match CLIP requirements [1,3,224,224]")
                
        except Exception as tensor_error:
            print(f"Image tensor preprocessing failed: {tensor_error}")
            return np.zeros(512, dtype=np.float32)
        
        # Generate embeddings with proper error handling
        try:
            with torch.no_grad():
                # Get CLIP image features (512 dimensions)
                image_features = clip_model.encode_image(image_tensor)
                
                # Validate output dimensions
                if image_features.shape[-1] != 512:
                    print(f"Warning: Unexpected embedding dimension: {image_features.shape}")
                    return np.zeros(512, dtype=np.float32)
                
                # Normalize features
                image_features = F.normalize(image_features, p=2, dim=1)
                
                # Convert to numpy with proper dtype
                result = image_features.squeeze().numpy().astype(np.float32)
                
                # Final validation
                if result.shape != (512,):
                    print(f"Warning: Final embedding shape {result.shape} != (512,)")
                    return np.zeros(512, dtype=np.float32)
                
                return result
                
        except Exception as embed_error:
            print(f"CLIP embedding generation failed: {embed_error}")
            return np.zeros(512, dtype=np.float32)
        
    except Exception as e:
        print(f"CLIP image processing completely failed: {e}")
        # Return zero vector with CLIP dimensions (512) and proper dtype
        return np.zeros(512, dtype=np.float32)
