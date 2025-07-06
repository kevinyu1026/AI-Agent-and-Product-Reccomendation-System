import streamlit as st
import sys
import os

# Add the parent directory to the path to import models
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import plotly.express as px

try:
    from models.embed_utils import (
        get_text_embedding, 
        get_image_embedding_simple,
        get_enhanced_text_embedding,
        get_enhanced_image_embedding,
        get_semantic_tags
    )
    from models.rag_utils import get_search_engine, generate_rag_description
    MODELS_AVAILABLE = True
    get_text_embedding_func = get_text_embedding
    get_image_embedding_simple_func = get_image_embedding_simple
    get_enhanced_text_embedding_func = get_enhanced_text_embedding
    get_enhanced_image_embedding_func = get_enhanced_image_embedding
    get_semantic_tags_func = get_semantic_tags
    get_search_engine_func = get_search_engine
    generate_rag_description_func = generate_rag_description
except ImportError as e:
    st.error(f"⚠️ Models not available: {e}")
    st.info("Please run the Jupyter notebook first to set up the embedding models.")
    MODELS_AVAILABLE = False
    get_text_embedding_func = None
    get_image_embedding_simple_func = None
    get_enhanced_text_embedding_func = None
    get_enhanced_image_embedding_func = None
    get_semantic_tags_func = None
    get_search_engine_func = None
    generate_rag_description_func = None

# Configure Streamlit page
st.set_page_config(
    page_title="🛍️ AI Product Recommendation Agent",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .product-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .similarity-score {
        background-color: #e1f5fe;
        padding: 5px 10px;
        border-radius: 15px;
        display: inline-block;
        margin: 5px 0;
    }
    .price-tag {
        background-color: #c8e6c9;
        padding: 5px 10px;
        border-radius: 10px;
        font-weight: bold;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Initialize search engine
@st.cache_resource
def load_search_engine():
    """Load and cache the search engine"""
    if not MODELS_AVAILABLE:
        return None
    
    try:
        # Check if required files exist
        embeddings_dir = os.path.join(parent_dir, 'embeddings')
        required_files = ['text_index.bin', 'products.pkl']
        
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(embeddings_dir, f))]
        if missing_files:
            st.error(f"Missing required files: {missing_files}")
            return None
        
        # Load the search engine with absolute path
        from models.rag_utils import ProductSearchEngine
        absolute_embeddings_path = os.path.join(parent_dir, 'embeddings')
        engine = ProductSearchEngine(embeddings_path=absolute_embeddings_path)
        
        if engine is not None and engine.products_df is not None:
            st.success(f"✅ Search engine loaded with {len(engine.products_df)} products")
            return engine
        else:
            st.error("Search engine failed to load products")
            return None
            
    except Exception as e:
        st.error(f"Failed to load search engine: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def display_product_card(product, similarity_score=None):
    """Display a product in a card format"""
    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display product image
            try:
                if product['image_url']:
                    response = requests.get(product['image_url'], timeout=10)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        st.image(img, use_column_width=True)
                    else:
                        st.write("🖼️ Image not available")
                else:
                    st.write("🖼️ No image")
            except Exception:
                st.write("🖼️ Image loading failed")
        
        with col2:
            # Product details
            st.markdown(f"### {product['title']}")
            
            # Price and similarity score
            col2a, col2b = st.columns(2)
            with col2a:
                price = product['price'] if product['price'] != 'N/A' else 'Price not available'
                st.markdown(f'<div class="price-tag">${price}</div>', unsafe_allow_html=True)
            
            with col2b:
                if similarity_score:
                    st.markdown(f'<div class="similarity-score">Match: {similarity_score:.1%}</div>', 
                              unsafe_allow_html=True)
            
            # Description
            st.write(f"**Description:** {product['description']}")
            
            # Tags
            if product['tags']:
                st.write(f"**Category:** {product['tags']}")
            
def enhanced_search_products(query, search_engine, top_k=5, use_enhanced=True):
    """Enhanced search with improved accuracy."""
    try:
        if use_enhanced and get_enhanced_text_embedding_func is not None:
            # Use enhanced embedding
            query_embedding = get_enhanced_text_embedding_func(query)
            
            # Get semantic tags for query understanding
            if get_semantic_tags_func is not None:
                query_tags = get_semantic_tags_func(query)
                st.info(f"🎯 Query understanding: {', '.join([f'{k}: {v}' for k, v in query_tags.items() if v])}")
        else:
            # Fallback to basic embedding
            if get_text_embedding_func is not None:
                query_embedding = get_text_embedding_func(query)
            else:
                st.error("Text embedding function not available")
                return pd.DataFrame()
        
        # Search for similar products
        results, _ = search_engine.search_similar(
            query_embedding, 
            search_type="text", 
            top_k=top_k
        )
        
        return results
        
    except Exception as e:
        st.error(f"Enhanced search failed: {e}")
        return pd.DataFrame()

def enhanced_image_search(image_input, search_engine, top_k=5, use_enhanced=True):
    """Enhanced image search with better visual understanding."""
    try:
        if use_enhanced and get_enhanced_image_embedding_func is not None:
            # Use enhanced image embedding
            image_embedding = get_enhanced_image_embedding_func(image_input)
        else:
            # Fallback to basic embedding
            if get_image_embedding_simple_func is not None:
                image_embedding = get_image_embedding_simple_func(image_input)
            else:
                st.error("Image embedding function not available")
                return pd.DataFrame()
        
        # Search for similar products
        results, _ = search_engine.search_similar(
            image_embedding,
            search_type="image",
            top_k=top_k
        )
        
        return results
        
    except Exception as e:
        st.error(f"Enhanced image search failed: {e}")
        return pd.DataFrame()

def display_enhanced_product_card(product, similarity_score=None, show_semantic_tags=False):
    """Enhanced product card with semantic information."""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Product image
            if product.get('image_url'):
                try:
                    response = requests.get(product['image_url'], timeout=5)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        st.image(image, width=150)
                    else:
                        st.info("📷 Image not available")
                except:
                    st.info("📷 Image not available")
            else:
                st.info("📷 No image")
        
        with col2:
            # Product title
            st.markdown(f"### {product['title']}")
            
            # Price with enhanced styling
            if product.get('price'):
                st.markdown(f"**💰 Price:** `${product['price']}`")
            
            # Similarity score
            if similarity_score is not None:
                confidence_color = "green" if similarity_score > 0.8 else "orange" if similarity_score > 0.6 else "red"
                st.markdown(f"**📊 Match Confidence:** :{confidence_color}[{similarity_score:.1%}]")
            
            # Description
            if product.get('description'):
                st.write(f"**Description:** {product['description']}")
            
            # Tags
            if product.get('tags'):
                st.write(f"**Category:** {product['tags']}")
            
            # Show semantic tags if available
            if show_semantic_tags and get_semantic_tags_func is not None:
                try:
                    full_text = f"{product['title']} {product.get('description', '')} {product.get('tags', '')}"
                    semantic_tags = get_semantic_tags_func(full_text)
                    
                    if any(semantic_tags.values()):
                        st.markdown("**🏷️ Product Attributes:**")
                        for category, values in semantic_tags.items():
                            if values:
                                st.markdown(f"  • {category.title()}: {', '.join(values)}")
                except:
                    pass
            
            st.markdown("---")

def main():
    # Header
    st.title("🛍️ AI Product Recommendation Agent")
    st.markdown("### Discover products using AI-powered multimodal search")
    
    # Load search engine
    search_engine = load_search_engine()
    
    if search_engine is None:
        st.error("❌ Search engine not available.")
        
        # Setup instructions
        st.markdown("""
        ## 🔧 Setup Required
        
        Please complete the following steps to use the AI recommendation system:
        
        ### Step 1: Generate Embeddings
        1. Open the Jupyter notebook: `notebooks/01_vector_search_rag.ipynb`
        2. Run the cells **one by one** in order:
           - Cell 1: Import libraries
           - Cell 3: Examine dataset  
           - Cell 4: Data preparation
           - Cell 5: Text embeddings
           - Cell 7: Create indices (skip cell 6 for now)
           - Cell 8: Test search
           - Cell 9: RAG demo
        
        ### Step 2: Refresh This Page
        Once the notebook has completed successfully, refresh this page to access the AI agent.
        
        ### 📁 Expected Files
        The notebook will create these files in the `embeddings/` folder:
        - `text_index.bin` - Search index
        - `products.pkl` - Product database
        - `metadata.json` - Configuration
        
        ---
        
        💡 **Tip**: Run the notebook cells step-by-step to avoid memory issues.
        """)
        
        # Show current status
        embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'embeddings')
        st.markdown("### 📋 Current Status")
        
        required_files = ['text_index.bin', 'products.pkl', 'metadata.json']
        for file in required_files:
            file_path = os.path.join(embeddings_dir, file)
            if os.path.exists(file_path):
                st.success(f"✅ {file} - Ready")
            else:
                st.error(f"❌ {file} - Missing")
        
        return
    
    # Sidebar with statistics
    with st.sidebar:
        st.header("📊 Dataset Info")
        if search_engine.products_df is not None:
            total_products = len(search_engine.products_df)
            st.metric("Total Products", total_products)
            
            # Category distribution
            if 'tags' in search_engine.products_df.columns:
                categories = search_engine.products_df['tags'].value_counts()
                st.write("**Categories:**")
                for cat, count in categories.head(5).items():
                    st.write(f"• {cat}: {count}")
            
            # Price distribution
            if 'price' in search_engine.products_df.columns:
                prices = pd.to_numeric(search_engine.products_df['price'], errors='coerce').dropna()
                if len(prices) > 0:
                    st.write(f"**Price Range:** ${prices.min():.0f} - ${prices.max():.0f}")
        
        st.markdown("---")
        st.header("🔧 Search Settings")
        search_type = st.selectbox(
            "Search Mode",
            ["multimodal", "text_only"],
            help="Multimodal uses both text and image features"
        )
        
        num_results = st.slider("Number of Results", 1, 10, 5)
        
        use_openai = st.checkbox(
            "Use OpenAI for descriptions",
            help="Requires OPENAI_API_KEY environment variable"
        )
    
    # Main search interface
    st.header("🔍 Search Products")
    
    # Create tabs for different search methods
    tab1, tab2, tab3 = st.tabs(["📝 Text Search", "🖼️ Image Search", "📊 Analytics"])
    
    with tab1:
        st.subheader("Search by Description")
        
        # Enhanced search toggle
        use_enhanced = st.checkbox("🚀 Use Enhanced Search", value=True, 
                                  help="Enhanced search uses improved text processing and semantic understanding")
        
        query = st.text_input(
            "What are you looking for?",
            placeholder="e.g., blue leather jacket, women's summer top, casual shoes..."
        )
        
        if query:
            with st.spinner("🔍 Searching..."):
                try:
                    if use_enhanced:
                        # Use enhanced search
                        results = enhanced_search_products(query, search_engine, num_results, use_enhanced=True)
                    else:
                        # Use basic search
                        if get_text_embedding_func is None:
                            st.error("Text embedding function not available")
                            return
                        
                        query_embedding = get_text_embedding_func(query)
                        
                        # Search for similar products
                        results, _ = search_engine.search_similar(
                            query_embedding, 
                            search_type=search_type.replace("_only", ""), 
                            top_k=num_results
                        )
                    
                    if len(results) > 0:
                        st.success(f"✅ Found {len(results)} matching products!")
                        
                        # Display results with enhanced cards
                        st.subheader("🎯 Search Results")
                        for _, product in results.iterrows():
                            similarity_score = product.get('similarity_score', 0)
                            display_enhanced_product_card(product, similarity_score, show_semantic_tags=use_enhanced)
                        
                        # Generate AI description
                        st.subheader("🧠 AI-Powered Recommendation")
                        with st.spinner("Generating AI insights..."):
                            if generate_rag_description_func is not None:
                                ai_description = generate_rag_description_func(query, results, use_openai)
                                st.markdown(ai_description)
                            else:
                                st.warning("AI description generation not available")
                    
                    else:
                        st.warning("No products found matching your search.")
                        
                        # Suggest alternative searches
                        if use_enhanced and get_semantic_tags_func is not None:
                            try:
                                query_tags = get_semantic_tags_func(query)
                                suggestions = []
                                if query_tags.get('color'):
                                    suggestions.append(f"Try searching without color: '{query.replace(query_tags['color'][0], '').strip()}'")
                                if query_tags.get('category'):
                                    suggestions.append(f"Try broader category search")
                                
                                if suggestions:
                                    st.info("💡 Search suggestions:\n" + "\n".join([f"• {s}" for s in suggestions]))
                            except:
                                pass
                        
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                    st.info("💡 Try using simpler keywords or check if the search index is properly loaded.")
    
    with tab2:
        st.subheader("Search by Image")
        
        # Enhanced image search toggle
        use_enhanced_img = st.checkbox("🚀 Use Enhanced Image Search", value=True,
                                      help="Enhanced image search uses CLIP for better visual understanding")
        
        uploaded_file = st.file_uploader(
            "Upload an image to find similar products",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of the product you're looking for"
        )
        
        if uploaded_file:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", width=300)
            
            with st.spinner("🔄 Analyzing image..."):
                try:
                    # Check if image index is available
                    if not hasattr(search_engine, 'image_index') or search_engine.image_index is None:
                        st.error("Image search index not available. Image embeddings may not have been generated.")
                        st.info("To enable image search, re-run the notebook cells that generate image embeddings.")
                        return
                    
                    if use_enhanced_img:
                        # Use enhanced image search
                        results = enhanced_image_search(uploaded_file, search_engine, num_results, use_enhanced=True)
                    else:
                        # Use basic image search
                        if get_image_embedding_simple_func is None:
                            st.error("Image embedding function not available")
                            return
                        
                        image_embedding = get_image_embedding_simple_func(uploaded_file)
                        st.success(f"✅ Image processed successfully (embedding: {image_embedding.shape[0]}D)")
                        
                        results, _ = search_engine.search_similar(
                            image_embedding,
                            search_type="image",  # Use image index for image search
                            top_k=num_results
                        )
                    
                    if len(results) > 0:
                        st.success(f"✅ Found {len(results)} visually similar products!")
                        
                        # Display results with enhanced cards
                        for _, product in results.iterrows():
                            similarity_score = product.get('similarity_score', 0)
                            display_enhanced_product_card(product, similarity_score, show_semantic_tags=use_enhanced_img)
                        
                        # Generate AI description
                        if generate_rag_description_func is not None:
                            ai_description = generate_rag_description_func(
                                "similar products to uploaded image", 
                                results, 
                                use_openai
                            )
                            st.markdown("### 🧠 AI Analysis")
                            st.markdown(ai_description)
                    else:
                        st.warning("No visually similar products found.")
                        st.info("💡 Try uploading a different image or ensure the image shows the product clearly.")
                        
                except Exception as e:
                    st.error(f"Image search failed: {str(e)}")
                    st.info("💡 Make sure the image is clear and shows a product similar to those in our catalog.")
    
    with tab3:
        st.subheader("📊 Product Analytics")
        
        if search_engine.products_df is not None:
            df = search_engine.products_df
            
            # Price distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Price Distribution**")
                prices = pd.to_numeric(df['price'], errors='coerce').dropna()
                if len(prices) > 0:
                    fig = px.histogram(
                        x=prices, 
                        nbins=10, 
                        title="Product Price Distribution",
                        labels={'x': 'Price ($)', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Category Distribution**")
                if 'tags' in df.columns:
                    categories = df['tags'].value_counts()
                    fig = px.pie(
                        values=categories.values,
                        names=categories.index,
                        title="Products by Category"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Product table
            st.write("**All Products**")
            display_df = df[['title', 'price', 'tags']].copy()
            display_df['price'] = display_df['price'].apply(lambda x: f"${x}" if x != 'N/A' else x)
            st.dataframe(display_df, use_container_width=True)
        
        else:
            st.warning("No product data available.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        🛍️ AI Product Recommendation System | 
        Built with Streamlit, FAISS, SentenceTransformers & ResNet
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
