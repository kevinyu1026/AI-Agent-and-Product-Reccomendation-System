#!/usr/bin/env python3
"""
Test script to demonstrate separate text and image thresholds with 55% text threshold
"""

import sys
import os
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.rag_utils import ProductSearchEngine
    from models.embed_utils import get_text_embedding
    
    print("🔍 Testing Separate Text and Image Thresholds")
    print("=" * 60)
    
    # Initialize search engine
    print("\n1. Loading search engine...")
    search_engine = ProductSearchEngine("embeddings")
    
    if search_engine.text_index is None:
        print("❌ No text index found!")
        sys.exit(1)
    
    print(f"✅ Loaded text index with {search_engine.text_index.ntotal} vectors")
    
    if search_engine.image_index is not None:
        print(f"✅ Loaded image index with {search_engine.image_index.ntotal} vectors")
    else:
        print("⚠️ Image index not available")
        
    if search_engine.products_df is not None:
        print(f"✅ Loaded {len(search_engine.products_df)} products")
    else:
        print("❌ No products data found!")
        sys.exit(1)
    
    # Test text search with higher threshold
    print("\n2. Testing TEXT search with VERY HIGH threshold (55%)...")
    query = "blue shirt"
    query_embedding = get_text_embedding(query)
    
    text_results, _ = search_engine.search_similar(
        query_embedding,
        search_type="text",
        top_k=10,
        text_threshold=55.0,    # Higher threshold for text (very precise)
        image_threshold=25.0    # Lower threshold for image (not used here)
    )
    
    print(f"🎯 Text search results for '{query}' (threshold: 55%):")
    print("-" * 50)
    if len(text_results) > 0:
        for i, (_, product) in enumerate(text_results.iterrows(), 1):
            similarity = product.get('similarity_score', 0)
            print(f"{i}. {product['title']} - {similarity:.2f}% similarity")
    else:
        print("❌ No results found with 55% threshold - too restrictive!")
    
    # Compare with lower thresholds
    print("\n3. Comparison with lower text thresholds...")
    
    # Test with 35% threshold
    print("\n3a. Text search with 35% threshold:")
    text_results_35, _ = search_engine.search_similar(
        query_embedding,
        search_type="text",
        top_k=10,
        text_threshold=35.0,
        image_threshold=25.0
    )
    print(f"📊 Found {len(text_results_35)} results with 35% threshold")
    
    # Test with 25% threshold
    print("\n3b. Text search with 25% threshold:")
    text_results_25, _ = search_engine.search_similar(
        query_embedding,
        search_type="text",
        top_k=10,
        text_threshold=25.0,
        image_threshold=25.0
    )
    print(f"📊 Found {len(text_results_25)} results with 25% threshold")
    
    print("\n" + "=" * 60)
    print("📊 THRESHOLD COMPARISON RESULTS:")
    print(f"• 55% threshold: {len(text_results)} results (very precise)")
    print(f"• 35% threshold: {len(text_results_35)} results (balanced)")  
    print(f"• 25% threshold: {len(text_results_25)} results (high recall)")
    
    print("\n🎯 55% Text Threshold Analysis:")
    if len(text_results) == 0:
        print("⚠️  TOO RESTRICTIVE: No results found")
        print("💡 Recommendation: Use 35-45% for better balance")
    elif len(text_results) < 3:
        print("⚠️  VERY RESTRICTIVE: Very few results")
        print("💡 Good for finding only the most relevant matches")
    else:
        print("✅ WORKING: Found relevant high-quality matches")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Threshold analysis completed!")
