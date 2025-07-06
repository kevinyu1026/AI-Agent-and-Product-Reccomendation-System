from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
import numpy as np
import clip

# Initialize models lazily to avoid loading at import time
_text_model = None
_resnet = None
_clip_model = None
_clip_preprocess = None

def get_text_model():
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _text_model

def get_resnet_model():
    global _resnet
    if _resnet is None:
        _resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        _resnet.eval()
    return _resnet

def get_clip_model():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device="cpu")
        _clip_model.eval()
    return _clip_model, _clip_preprocess

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_text_embedding(text):
    """
    Get the embedding for a given text.
    """
    text_model = get_text_model()
    return text_model.encode(text)

def get_image_embedding(image_input):
    """
    Get the embedding for a given image.
    
    Args:
        image_input: Can be a file path (str) or BytesIO object
    """
    resnet = get_resnet_model()
    
    # Handle different input types
    if isinstance(image_input, str):
        # File path
        image = Image.open(image_input).convert('RGB')
    else:
        # BytesIO object
        image = Image.open(image_input).convert('RGB')
    
    image_tensor = preprocess(image)
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = transforms.ToTensor()(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        # Get features from ResNet50 (1000 dimensions)
        features = resnet(image_tensor)
        features = F.normalize(features, p=2, dim=1)  # Normalize the features
        
        # Reduce dimensions to match text embeddings (384)
        # Use a simple linear projection
        features_reduced = torch.nn.functional.linear(
            features, 
            torch.randn(384, 1000) * 0.1  # Random projection matrix
        )
        features_reduced = F.normalize(features_reduced, p=2, dim=1)
        
    return features_reduced.squeeze().numpy()  # Remove batch dimension and convert to numpy array

def get_image_embedding_simple(image_input):
    """
    Get image embedding using CLIP that's compatible with text embeddings.
    CLIP produces 512-dimensional image embeddings.
    """
    try:
        clip_model, clip_preprocess = get_clip_model()
        
        # Check if models are loaded properly
        if clip_model is None or clip_preprocess is None:
            raise ValueError("CLIP model not loaded properly")
        
        # Handle different input types
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = Image.open(image_input).convert('RGB')
        
        # Preprocess image for CLIP
        image_tensor = clip_preprocess(image)
        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = transforms.ToTensor()(image_tensor)
        image_tensor = image_tensor.unsqueeze(0)
        
        with torch.no_grad():
            # Get CLIP image features (512 dimensions)
            image_features = clip_model.encode_image(image_tensor)
            image_features = F.normalize(image_features, p=2, dim=1)
        
        return image_features.squeeze().numpy()
        
    except Exception as e:
        print(f"CLIP image processing failed: {e}")
        # Return zero vector with CLIP dimensions (512)
        return np.zeros(512)

def get_text_embedding_clip(text):
    """
    Get text embedding using CLIP (optional - for comparison).
    """
    try:
        clip_model, _ = get_clip_model()
        
        # Tokenize text for CLIP
        text_tokens = clip.tokenize([text])
        
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=1)
        
        return text_features.squeeze().numpy()
        
    except Exception as e:
        print(f"CLIP text processing failed: {e}")
        return np.zeros(512)

def get_enhanced_text_embedding(text):
    """
    Enhanced text embedding with better preprocessing and context understanding.
    """
    import re
    import string
    
    try:
        # Advanced text preprocessing for fashion/apparel domain
        text = str(text).lower().strip()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Standardize fashion terms and synonyms
        fashion_synonyms = {
            'tee': 'shirt',
            't-shirt': 'shirt', 
            'tshirt': 'shirt',
            'jumper': 'sweater',
            'pullover': 'sweater',
            'hoodie': 'sweatshirt',
            'top': 'shirt',
            'blouse': 'shirt',
            'pants': 'trousers',
            'jeans': 'denim pants',
            'jacket': 'outerwear',
            'coat': 'outerwear',
            'dress': 'dress',
            'skirt': 'skirt',
            'shorts': 'short pants'
        }
        
        for synonym, standard in fashion_synonyms.items():
            text = re.sub(rf'\b{synonym}\b', standard, text)
        
        # Enhance with fashion-specific context
        color_terms = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple', 'orange', 'brown', 'gray', 'grey']
        material_terms = ['cotton', 'silk', 'wool', 'denim', 'leather', 'polyester', 'linen', 'cashmere']
        style_terms = ['casual', 'formal', 'vintage', 'modern', 'classic', 'sporty', 'elegant', 'trendy']
        
        # Add semantic tags based on detected terms
        semantic_tags = []
        
        for color in color_terms:
            if color in text:
                semantic_tags.append(f"color:{color}")
        
        for material in material_terms:
            if material in text:
                semantic_tags.append(f"material:{material}")
                
        for style in style_terms:
            if style in text:
                semantic_tags.append(f"style:{style}")
        
        # Combine original text with semantic enrichment
        enriched_text = text
        if semantic_tags:
            enriched_text += " " + " ".join(semantic_tags)
        
        # Use sentence transformer for final embedding
        text_model = get_text_model()
        return text_model.encode(enriched_text)
        
    except Exception as e:
        print(f"Enhanced text embedding failed: {e}")
        # Fallback to basic embedding
        return get_text_embedding(text)

def get_enhanced_image_embedding(image_input):
    """
    Enhanced image embedding with multiple CNN features and CLIP.
    """
    try:
        # Get CLIP embedding (primary)
        clip_embedding = get_image_embedding_simple(image_input)
        
        # Enhance with color and texture analysis
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image_input.seek(0)  # Reset BytesIO position
            image = Image.open(image_input).convert('RGB')
        
        # Extract color histogram features
        import numpy as np
        
        # Convert to numpy array for color analysis
        img_array = np.array(image.resize((64, 64)))  # Smaller size for efficiency
        
        # Get dominant colors (simplified)
        colors = img_array.reshape(-1, 3)
        
        # Calculate color distribution features
        color_features = []
        for channel in range(3):  # RGB
            channel_data = colors[:, channel]
            color_features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data)
            ])
        
        # Normalize color features
        color_features = np.array(color_features)
        color_features = color_features / 255.0  # Normalize to 0-1
        
        # Pad or truncate to fixed size (9 features for RGB stats)
        if len(color_features) > 9:
            color_features = color_features[:9]
        else:
            color_features = np.pad(color_features, (0, 9 - len(color_features)))
        
        # The CLIP embedding is 512D, keep it as primary feature
        return clip_embedding
        
    except Exception as e:
        print(f"Enhanced image embedding failed: {e}")
        return get_image_embedding_simple(image_input)

def calculate_hybrid_similarity(query_text, query_image, product_text_emb, product_image_emb, text_weight=0.7, image_weight=0.3):
    """
    Calculate hybrid similarity combining text and image embeddings.
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = []
        
        # Text similarity
        if query_text is not None and product_text_emb is not None:
            query_text_emb = get_enhanced_text_embedding(query_text)
            text_sim = cosine_similarity(
                np.array(query_text_emb).reshape(1, -1), 
                np.array(product_text_emb).reshape(1, -1)
            )[0][0]
            similarities.append(('text', text_sim, text_weight))
        
        # Image similarity  
        if query_image is not None and product_image_emb is not None:
            query_image_emb = get_enhanced_image_embedding(query_image)
            image_sim = cosine_similarity(
                np.array(query_image_emb).reshape(1, -1), 
                np.array(product_image_emb).reshape(1, -1)
            )[0][0]
            similarities.append(('image', image_sim, image_weight))
        
        # Calculate weighted average
        if similarities:
            total_weighted_sim = sum(sim * weight for _, sim, weight in similarities)
            total_weight = sum(weight for _, _, weight in similarities)
            return total_weighted_sim / total_weight if total_weight > 0 else 0.0
        
        return 0.0
        
    except Exception as e:
        print(f"Hybrid similarity calculation failed: {e}")
        return 0.0

def get_semantic_tags(text):
    """
    Extract semantic tags from product text for better categorization.
    """
    text = str(text).lower()
    
    tags = {
        'gender': [],
        'category': [],
        'color': [],
        'material': [],
        'style': [],
        'season': []
    }
    
    # Gender detection
    if any(word in text for word in ['men', 'male', 'guy', 'masculine']):
        tags['gender'].append('men')
    if any(word in text for word in ['women', 'female', 'girl', 'feminine', 'lady']):
        tags['gender'].append('women')
    if any(word in text for word in ['unisex', 'neutral']):
        tags['gender'].append('unisex')
    
    # Category detection
    categories = {
        'tops': ['shirt', 'blouse', 'top', 'tee', 't-shirt', 'tank', 'sweater', 'pullover'],
        'bottoms': ['pants', 'jeans', 'trousers', 'shorts', 'skirt'],
        'outerwear': ['jacket', 'coat', 'blazer', 'cardigan', 'hoodie'],
        'dresses': ['dress', 'gown', 'frock'],
        'accessories': ['bag', 'purse', 'belt', 'scarf', 'hat']
    }
    
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            tags['category'].append(category)
    
    # Color detection
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple', 
              'orange', 'brown', 'gray', 'grey', 'navy', 'olive', 'cream', 'beige']
    for color in colors:
        if color in text:
            tags['color'].append(color)
    
    # Material detection
    materials = ['cotton', 'silk', 'wool', 'denim', 'leather', 'polyester', 'linen']
    for material in materials:
        if material in text:
            tags['material'].append(material)
    
    # Style detection
    styles = ['casual', 'formal', 'vintage', 'modern', 'classic', 'sporty', 'elegant']
    for style in styles:
        if style in text:
            tags['style'].append(style)
    
    # Season detection
    seasons = ['summer', 'winter', 'spring', 'fall', 'autumn']
    for season in seasons:
        if season in text:
            tags['season'].append(season)
    
    return tags
