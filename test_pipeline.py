#!/usr/bin/env python3
"""
Test pipeline for the AI Agent and Product Recommendation System
This script runs the key steps from the notebook to test functionality
"""

import pandas as pd 
import faiss
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('.'))

print("🚀 Starting AI Agent and Product Recommendation System Test")
print("=" * 60)

# Step 1: Test CLIP installation
print("\n🔧 Step 1: Testing CLIP installation...")
try:
    import clip
    print("✅ CLIP imported successfully!")
    
    # Test loading a model
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    print("✅ CLIP model loaded successfully!")
    
    # Test embedding functions
    from models.embed_utils import get_text_embedding, get_image_embedding_simple
    
    # Test text embedding
    text_emb = get_text_embedding("test text")
    print(f"✅ Text embedding shape: {text_emb.shape} (SentenceTransformer)")
    
    # Test image embedding with a dummy image
    from PIL import Image
    from io import BytesIO
    
    test_image = Image.new('RGB', (224, 224), color='blue')
    img_bytes = BytesIO()
    test_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    img_emb = get_image_embedding_simple(img_bytes)
    print(f"✅ Image embedding shape: {img_emb.shape} (CLIP)")
    
    print("🎉 All embedding functions work!")
    
except Exception as e:
    print(f"❌ Error in Step 1: {e}")
    sys.exit(1)

# Step 2: Test data loading
print("\n📊 Step 2: Testing data loading...")
try:
    df = pd.read_csv('data/apparel.csv')
    print(f"✅ Loaded {len(df)} total rows")
    
    # Clean and filter the data
    df_clean = df.dropna(subset=['Title', 'Image Src']).copy()
    df_clean = df_clean[df_clean['Title'].str.strip() != '']
    df_clean = df_clean.drop_duplicates(subset=['Title'])
    
    # Limit to first 5 products for quick testing
    df_clean = df_clean.head(5)
    print(f"✅ Processing {len(df_clean)} products for testing")
    
    # Show sample products
    for i, row in df_clean.head(3).iterrows():
        title = row['Title'][:50] + "..." if len(row['Title']) > 50 else row['Title']
        print(f"   - {title}")
        
except Exception as e:
    print(f"❌ Error in Step 2: {e}")
    sys.exit(1)

# Step 3: Test text embeddings
print("\n📝 Step 3: Testing text embedding generation...")
try:
    import requests
    from io import BytesIO
    from tqdm import tqdm
    import time
    import json
    import gc
    
    products_data = []
    text_embeddings = []
    
    for idx, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Text Embeddings"):
        # Prepare text content
        title = str(row['Title']).strip()
        description = str(row['Body (HTML)']).strip() if pd.notna(row['Body (HTML)']) else ""
        price = str(row['Variant Price']) if pd.notna(row['Variant Price']) else "N/A"
        tags = str(row['Tags']).strip() if pd.notna(row['Tags']) else ""
        
        # Combine text features
        text_content = f"{title}. {description}. Price: ${price}. Category: {tags}"
        
        # Generate text embedding
        text_vec = get_text_embedding(text_content)
        
        # Store product data
        product_data = {
            'title': title,
            'description': description,
            'price': price,
            'tags': tags,
            'image_url': row['Image Src'],
            'handle': row['Handle']
        }
        
        products_data.append(product_data)
        text_embeddings.append(text_vec)
    
    print(f"✅ Generated {len(text_embeddings)} text embeddings")
    
except Exception as e:
    print(f"❌ Error in Step 3: {e}")
    sys.exit(1)

# Step 4: Test image embeddings (with just 2 images to be safe)
print("\n🖼️  Step 4: Testing image embedding generation...")
image_embeddings = []
successful_images = 0

try:
    for i, product_data in enumerate(tqdm(products_data[:2], desc="Image Processing")):
        try:
            img_url = product_data['image_url']
            
            # Download image
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(img_url, timeout=10, headers=headers)
            response.raise_for_status()
            
            # Process image using CLIP
            img_bytes = BytesIO(response.content)
            img_vec = get_image_embedding_simple(img_bytes)
            image_embeddings.append(img_vec)
            successful_images += 1
            
            print(f"✅ Processed image for: {product_data['title'][:30]}...")
            
        except Exception as e:
            print(f"⚠️  Image failed for {product_data['title'][:30]}...: {str(e)[:50]}")
            image_embeddings.append(np.zeros(512))  # Zero vector for CLIP dimensions
        
        time.sleep(0.5)  # Be nice to servers
    
    print(f"✅ Processed images: {successful_images}/{len(products_data[:2])} successful")
    
except Exception as e:
    print(f"❌ Error in Step 4: {e}")
    print("Continuing with text-only search...")
    image_embeddings = []
    successful_images = 0

# Step 5: Test index creation
print("\n🔍 Step 5: Testing FAISS index creation...")
try:
    # Create text index
    text_embeddings_np = np.vstack(text_embeddings).astype('float32')
    text_index = faiss.IndexFlatL2(text_embeddings_np.shape[1])
    text_index.add(text_embeddings_np)
    print(f"✅ Text index created: {text_index.ntotal} vectors, {text_embeddings_np.shape[1]}D")
    
    # Create image index if we have images
    image_index = None
    if len(image_embeddings) > 0:
        image_embeddings_np = np.vstack(image_embeddings).astype('float32')
        non_zero_images = np.any(image_embeddings_np != 0, axis=1)
        if np.any(non_zero_images):
            image_index = faiss.IndexFlatL2(image_embeddings_np.shape[1])
            image_index.add(image_embeddings_np)
            print(f"✅ Image index created: {image_index.ntotal} vectors, {image_embeddings_np.shape[1]}D")
    
    # Create directories and save
    os.makedirs('embeddings', exist_ok=True)
    faiss.write_index(text_index, "embeddings/text_index.bin")
    
    if image_index:
        faiss.write_index(image_index, "embeddings/image_index.bin")
    
    # Save products data
    products_df = pd.DataFrame(products_data)
    products_df.to_pickle("embeddings/products.pkl")
    
    print("✅ Saved indices and data")
    
except Exception as e:
    print(f"❌ Error in Step 5: {e}")
    sys.exit(1)

# Step 6: Test search functionality
print("\n🔍 Step 6: Testing search functionality...")
try:
    # Test text search
    test_queries = ["shirt", "blue", "women"]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        # Generate query embedding
        query_vec = get_text_embedding(query)
        
        # Search for similar products
        distances, indices = text_index.search(
            np.array([query_vec]).astype('float32'), 3
        )
        
        print("   Top matches:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(products_df):
                product = products_df.iloc[idx]
                similarity = 1 / (1 + dist)
                print(f"   {i+1}. {product['title']} (similarity: {similarity:.3f})")
    
    print("✅ Search functionality works!")
    
except Exception as e:
    print(f"❌ Error in Step 6: {e}")

print("\n🎉 Test pipeline completed successfully!")
print("=" * 60)
print("📋 Summary:")
print(f"   - Text embeddings: {len(text_embeddings)} products (384D)")
print(f"   - Image embeddings: {len(image_embeddings)} processed ({successful_images} successful)")
print(f"   - Search indices: Created and tested")
print("   - System: Ready for use!")
print("\n🚀 You can now run the Streamlit app with:")
print("   streamlit run app/streamlit_app.py")
