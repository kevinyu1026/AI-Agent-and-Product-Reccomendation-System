#!/usr/bin/env python3
"""
Debug script to test threshold filtering in the search engine
"""

import sys
import os
sys.path.append('.')

import numpy as np
from models.rag_utils import ProductSearchEngine
from models.embed_utils import get_text_embedding

def debug_search_threshold():
    """Test the threshold filtering logic"""
    
    print("🔍 Debugging Threshold Filtering...")
    
    # Initialize search engine
    search_engine = ProductSearchEngine()
    
    if search_engine.products_df is None:
        print("❌ No products loaded. Please run the embedding generation notebook first.")
        return
    
    # Test query
    query = "black leather bag"
    print(f"\n📝 Query: '{query}'")
    print(f"🎯 Threshold: 49%")
    
    try:
        # Get embedding
        query_embedding = get_text_embedding(query)
        print(f"✅ Generated embedding: {query_embedding.shape}")
        
        # Perform manual search to debug
        index = search_engine.text_index
        if index is None:
            print("❌ Text index not available")
            return
        
        # Search for candidates
        search_k = 20  # Get more candidates to see what's happening
        query_vec = np.array([query_embedding]).astype('float32')
        distances, indices = index.search(query_vec, search_k)
        
        print(f"\n🔍 Raw search results (top {search_k}):")
        print("Index | Distance | Similarity% | Product Title")
        print("-" * 80)
        
        valid_count = 0
        threshold = 49.0
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(search_engine.products_df):
                # Calculate similarity exactly as in the main code
                similarity_score = 1.0 / (1.0 + dist)
                similarity_percentage = similarity_score * 100
                
                # Get product title
                product_title = search_engine.products_df.iloc[idx]['title']
                
                # Check if it meets threshold
                meets_threshold = similarity_percentage >= threshold
                if meets_threshold:
                    valid_count += 1
                
                status = "✅" if meets_threshold else "❌"
                print(f"{idx:5d} | {dist:8.4f} | {similarity_percentage:9.2f}% | {status} {product_title[:50]}")
        
        print(f"\n📊 Summary:")
        print(f"   Total candidates: {len(distances[0])}")
        print(f"   Above threshold: {valid_count}")
        print(f"   Below threshold: {len(distances[0]) - valid_count}")
        
        # Now test the actual search_similar method
        print(f"\n🧪 Testing search_similar method:")
        results, _ = search_engine.search_similar(
            query_embedding, 
            search_type="text", 
            top_k=10,
            similarity_threshold=49.0
        )
        
        print(f"   Results returned: {len(results)}")
        if len(results) > 0:
            print(f"\n📋 Returned products:")
            for i, (_, row) in enumerate(results.iterrows()):
                similarity = row.get('similarity_score', 'N/A')
                print(f"   {i+1}. {row['title'][:50]} | Similarity: {similarity:.2f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_search_threshold()
