import os
import numpy as np
import faiss
import pandas as pd
import json
from typing import List, Dict, Optional, Tuple

class ProductSearchEngine:
    """Advanced product search engine with multimodal capabilities"""
    
    def __init__(self, embeddings_path: str = "embeddings"):
        self.embeddings_path = embeddings_path
        self.products_df = None
        self.text_index = None
        self.combined_index = None
        self.image_index = None
        self.metadata = None
        self._load_indices()
    
    def _load_indices(self):
        """Load all FAISS indices and product data"""
        try:
            # Load products data
            products_path = os.path.join(self.embeddings_path, "products.pkl")
            if os.path.exists(products_path):
                self.products_df = pd.read_pickle(products_path)
                print(f"✅ Loaded {len(self.products_df)} products")
            
            # Load metadata
            metadata_path = os.path.join(self.embeddings_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            # Load FAISS indices
            text_index_path = os.path.join(self.embeddings_path, "text_index.bin")
            if os.path.exists(text_index_path):
                self.text_index = faiss.read_index(text_index_path)
                print("✅ Loaded text index")
            
            combined_index_path = os.path.join(self.embeddings_path, "combined_index.bin")
            if os.path.exists(combined_index_path):
                self.combined_index = faiss.read_index(combined_index_path)
                print("✅ Loaded combined multimodal index")
            
            image_index_path = os.path.join(self.embeddings_path, "image_index.bin")
            if os.path.exists(image_index_path):
                self.image_index = faiss.read_index(image_index_path)
                print("✅ Loaded image index")
                
        except Exception as e:
            print(f"⚠️ Error loading indices: {e}")
            print("Please run the embedding generation notebook first.")
    
    def search_similar(self, query_embedding: np.ndarray, search_type: str = "combined", 
                      top_k: int = 5) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Search for similar products
        
        Args:
            query_embedding: Query vector
            search_type: 'text', 'combined', or 'image'
            top_k: Number of results to return
            
        Returns:
            Tuple of (product_dataframe, distances)
        """
        if self.products_df is None:
            raise ValueError("No products loaded. Run embedding generation first.")
        
        # Select appropriate index
        if search_type == "text" and self.text_index:
            index = self.text_index
        elif search_type == "combined" and self.combined_index:
            index = self.combined_index
        elif search_type == "image" and self.image_index:
            index = self.image_index
        else:
            # Fallback to combined or text
            index = self.combined_index or self.text_index
            if index is None:
                raise ValueError("No search index available")
        
        # Perform search
        query_vec = np.array([query_embedding]).astype('float32')
        distances, indices = index.search(query_vec, top_k)
        
        # Get products
        valid_indices = indices[0][indices[0] < len(self.products_df)]
        results = self.products_df.iloc[valid_indices].copy()
        result_distances = distances[0][:len(valid_indices)]
        
        # Add similarity scores (convert distance to similarity)
        results['similarity_score'] = 1 / (1 + result_distances)
        results['distance'] = result_distances
        
        return results, result_distances
    
    def get_product_recommendations(self, query: str, query_embedding: np.ndarray, 
                                  top_k: int = 3) -> Dict:
        """Get comprehensive product recommendations"""
        
        # Search different modalities
        recommendations = {}
        
        if self.combined_index:
            combined_results, _ = self.search_similar(query_embedding, "combined", top_k)
            recommendations['multimodal'] = combined_results
        
        if self.text_index:
            text_results, _ = self.search_similar(query_embedding, "text", top_k)
            recommendations['text_only'] = text_results
        
        # Generate insights
        recommendations['query'] = query
        recommendations['total_products'] = len(self.products_df) if self.products_df is not None else 0
        
        return recommendations


def generate_rag_description(query: str, product_info: pd.DataFrame, 
                           use_openai: bool = False) -> str:
    """
    Generate AI-powered product descriptions and recommendations
    
    Args:
        query: User search query
        product_info: DataFrame with product results
        use_openai: Whether to use OpenAI API (requires API key)
    
    Returns:
        Generated description/recommendation
    """
    
    if len(product_info) == 0:
        return "No products found matching your search criteria."
    
    # Create a structured summary
    summary_parts = []
    summary_parts.append(f"Based on your search for '{query}', here are the top recommendations:\n")
    
    for i, (_, product) in enumerate(product_info.head(3).iterrows(), 1):
        price = f"${product['price']}" if product['price'] != 'N/A' else "Price not available"
        similarity = f"{product.get('similarity_score', 0):.2f}" if 'similarity_score' in product else "N/A"
        
        summary_parts.append(
            f"{i}. **{product['title']}** ({price})\n"
            f"   - {product['description'][:150]}{'...' if len(product['description']) > 150 else ''}\n"
            f"   - Category: {product['tags']}\n"
            f"   - Match Score: {similarity}\n"
        )
    
    # Add general insights
    if len(product_info) > 1:
        price_range = product_info['price'].replace('N/A', np.nan).dropna()
        if len(price_range) > 0:
            try:
                price_range = pd.to_numeric(price_range, errors='coerce').dropna()
                if len(price_range) > 0:
                    summary_parts.append(f"\n💰 Price range: ${price_range.min():.0f} - ${price_range.max():.0f}")
            except:
                pass
        
        categories = product_info['tags'].value_counts()
        if len(categories) > 0:
            summary_parts.append(f"\n🏷️ Main categories: {', '.join(categories.head(3).index.tolist())}")
    
    summary_parts.append(f"\n🔍 Found {len(product_info)} total matches for your search.")
    
    if use_openai:
        try:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            
            if openai.api_key:
                prompt = f"""You are a helpful shopping assistant. A user searched for: {query}

Here are the top matching products:
{product_info[['title', 'description', 'price', 'tags']].head(3).to_string(index=False)}

Generate a friendly, informative recommendation that highlights the best options and explains why they match the user's search. Keep it concise but helpful."""
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.7
                )
                return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"OpenAI API failed: {e}")
    
    return "\n".join(summary_parts)


# Global search engine instance
_search_engine = None

def get_search_engine() -> ProductSearchEngine:
    """Get or create global search engine instance"""
    global _search_engine
    if _search_engine is None:
        _search_engine = ProductSearchEngine()
    return _search_engine

def search_similar(query_embedding: np.ndarray, top_k: int = 3) -> pd.DataFrame:
    """Backward compatibility function"""
    engine = get_search_engine()
    results, _ = engine.search_similar(query_embedding, "combined", top_k)
    return results
