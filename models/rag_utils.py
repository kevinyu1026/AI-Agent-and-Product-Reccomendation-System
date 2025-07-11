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
                      top_k: int = 10, similarity_threshold: Optional[float] = None, 
                      text_threshold: float = 55.0, image_threshold: float = 25.0) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Search for similar products using quality-based filtering
        
        Args:
            query_embedding: Query vector
            search_type: 'text', 'combined', or 'image'
            top_k: Maximum number of results to consider (not guaranteed return count)
            similarity_threshold: Legacy parameter for backward compatibility (1-100 percentage scale)
            text_threshold: Similarity threshold for text search (default: 55%)
            image_threshold: Similarity threshold for image search (default: 25%)
            
        Returns:
            Tuple of (product_dataframe, distances) - ONLY relevant products (0-N results)
        """
        if self.products_df is None:
            raise ValueError("No products loaded. Run embedding generation first.")
        
        # Determine which threshold to use based on search type and parameters
        if similarity_threshold is not None:
            # Legacy mode: use the provided threshold
            effective_threshold = similarity_threshold
        else:
            # New mode: use search-type specific thresholds
            if search_type == "text":
                effective_threshold = text_threshold
            elif search_type == "image":
                effective_threshold = image_threshold
            else:  # combined/multimodal - use text threshold since it's primarily text-based
                effective_threshold = text_threshold  # Use text threshold for combined search (55%)
        
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
        
        # Perform search - get more candidates to properly filter
        search_k = min(top_k * 2, len(self.products_df))
        query_vec = np.array([query_embedding]).astype('float32')
        distances, indices = index.search(query_vec, search_k)
        
        # Filter for valid indices and similarity threshold - QUALITY OVER QUANTITY
        valid_results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.products_df):
                # Convert L2 distance to similarity score (simple and intuitive)
                # For FAISS L2 distance: smaller distance = higher similarity
                # Use 1 / (1 + distance) formula for intuitive 0-1 similarity
                similarity_score = 1.0 / (1.0 + dist)
                
                # Convert to percentage (0-100) for better interpretability
                similarity_percentage = similarity_score * 100
                
                # Only include products that meet similarity threshold (threshold is already in percentage)
                if similarity_percentage >= effective_threshold:
                    valid_results.append((idx, dist, similarity_percentage))
        
        # Return ONLY relevant products (no padding to top_k)
        print(f"🎯 Found {len(valid_results)} relevant products (threshold: {effective_threshold}%, type: {search_type})")
        valid_results = valid_results  # Return all relevant results, don't limit artificially
        
        if not valid_results:
            # Return empty DataFrame with same structure
            empty_df = pd.DataFrame(columns=self.products_df.columns.tolist() + ['similarity_score', 'distance'])
            return empty_df, np.array([])
        
        # Extract results
        result_indices = [r[0] for r in valid_results]
        result_distances = [r[1] for r in valid_results]
        result_similarities = [r[2] for r in valid_results]
        
        # Get products
        results = self.products_df.iloc[result_indices].copy()
        
        # Add similarity scores
        results['similarity_score'] = result_similarities
        results['distance'] = result_distances
        
        return results, np.array(result_distances)
    
    def get_product_recommendations(self, query: str, query_embedding: np.ndarray, 
                                  top_k: int = 10, text_threshold: float = 55.0, 
                                  image_threshold: float = 25.0) -> Dict:
        """Get comprehensive product recommendations with smart filtering using separate thresholds"""
        
        # Search different modalities with improved thresholds
        recommendations = {}
        
        if self.combined_index:
            combined_results, _ = self.search_similar(query_embedding, "combined", top_k, 
                                                    text_threshold=text_threshold, 
                                                    image_threshold=image_threshold)
            recommendations['multimodal'] = combined_results
        
        if self.text_index:
            text_results, _ = self.search_similar(query_embedding, "text", top_k, 
                                                text_threshold=text_threshold, 
                                                image_threshold=image_threshold)
            recommendations['text_only'] = text_results
        
        # Generate insights
        recommendations['query'] = query
        recommendations['total_products'] = len(self.products_df) if self.products_df is not None else 0
        recommendations['text_threshold'] = text_threshold
        recommendations['image_threshold'] = image_threshold
        
        # Add information about actual matches found (quality-based)
        if 'multimodal' in recommendations:
            recommendations['matches_found'] = len(recommendations['multimodal'])
            recommendations['search_method'] = 'multimodal'
        elif 'text_only' in recommendations:
            recommendations['matches_found'] = len(recommendations['text_only'])
            recommendations['search_method'] = 'text_only'
        else:
            recommendations['matches_found'] = 0
            recommendations['search_method'] = 'none'
        
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
            
            # Try using OpenAI for enhanced descriptions (optional)
            try:
                import openai
                if hasattr(openai, 'api_key') and openai.api_key:
                    prompt = f"""You are a helpful shopping assistant. A user searched for: {query}

Here are the top matching products:
{product_info[['title', 'description', 'price', 'tags']].head(3).to_string(index=False)}

Generate a friendly, informative recommendation that highlights the best options and explains why they match the user's search. Keep it concise but helpful."""
                    
                    # Use the newer OpenAI client format
                    client = openai.OpenAI()
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,
                        temperature=0.7
                    )
                    return response.choices[0].message.content or "AI response unavailable"
            except Exception as e:
                print(f"OpenAI API unavailable: {e}")
                # Continue with fallback response
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
