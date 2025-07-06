#!/usr/bin/env python3
"""
Minimal CLIP test to isolate the segmentation fault issue
"""

import os
import sys
sys.path.append(os.path.abspath('.'))

print("Testing CLIP functionality...")

try:
    # Test basic imports
    import torch
    import clip
    print("✅ Imports successful")
    
    # Test model loading
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("✅ CLIP model loaded")
    
    # Test our embed_utils
    from models.embed_utils import get_text_embedding, get_image_embedding_simple
    print("✅ embed_utils imported")
    
    # Test text embedding
    text_embedding = get_text_embedding("test shirt")
    print(f"✅ Text embedding generated: shape {text_embedding.shape}")
    
    # Test with a simple test image (create a dummy image bytes)
    from PIL import Image
    from io import BytesIO
    
    # Create a simple test image
    test_img = Image.new('RGB', (224, 224), color='red')
    img_bytes = BytesIO()
    test_img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    image_embedding = get_image_embedding_simple(img_bytes)
    print(f"✅ Image embedding generated: shape {image_embedding.shape}")
    
    print("🎉 All CLIP functionality working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
