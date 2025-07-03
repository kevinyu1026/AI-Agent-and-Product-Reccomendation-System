from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

# Initialize models lazily to avoid loading at import time
_text_model = None
_resnet = None

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

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

def get_text_embedding(text):
    """
    Get the embedding for a given text.
    """
    text_model = get_text_model()
    return text_model.encode(text)

def get_image_embedding(image_path):
    """
    Get the embedding for a given image.
    """
    resnet = get_resnet_model()
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = resnet(image_tensor)
        features = F.normalize(features, p=2, dim=1)  # Normalize the features
    return features.squeeze().numpy()  # Remove batch dimension and convert to numpy array
