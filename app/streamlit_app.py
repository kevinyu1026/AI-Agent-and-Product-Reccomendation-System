import streamlit as st
import sys
import os

# Add the parent directory to the path to import models
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

try:
    from models.embed_utils import get_text_embedding, get_image_embedding
    from models.rag_utils import get_search_engine, generate_rag_description
    MODELS_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Models not available: {e}")
    st.info("Please run the Jupyter notebook first to set up the embedding models.")
    MODELS_AVAILABLE = False

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
        # Check if embeddings exist
        embeddings_dir = os.path.join(parent_dir, 'embeddings')
        required_files = ['text_index.bin', 'products.pkl']
        
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(embeddings_dir, f))]
        
        if missing_files:
            st.warning(f"⚠️ Missing embedding files: {missing_files}")
            st.info("Please run the Jupyter notebook first to generate embeddings.")
            return None
            
        engine = get_search_engine()
        if engine.products_df is not None:
            return engine
        else:
            return None
    except Exception as e:
        st.error(f"Failed to load search engine: {e}")
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
        query = st.text_input(
            "What are you looking for?",
            placeholder="e.g., blue leather jacket, women's summer top, casual shoes...",
            help="Describe the product you're looking for in natural language"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("🔍 Search", type="primary")
        
        if search_button and query:
            with st.spinner("🔄 Searching products..."):
                try:
                    # Generate embedding for query
                    query_embedding = get_text_embedding(query)
                    
                    # Search for similar products
                    results, distances = search_engine.search_similar(
                        query_embedding, 
                        search_type=search_type.replace("_only", ""), 
                        top_k=num_results
                    )
                    
                    if len(results) > 0:
                        st.success(f"✅ Found {len(results)} matching products!")
                        
                        # Display results
                        st.subheader("🎯 Search Results")
                        for idx, (_, product) in enumerate(results.iterrows()):
                            similarity_score = product.get('similarity_score', 0)
                            display_product_card(product, similarity_score)
                        
                        # Generate AI description
                        st.subheader("🧠 AI-Powered Recommendation")
                        with st.spinner("Generating AI insights..."):
                            ai_description = generate_rag_description(query, results, use_openai)
                            st.markdown(ai_description)
                    
                    else:
                        st.warning("No products found matching your search.")
                        
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
    
    with tab2:
        st.subheader("Search by Image")
        uploaded_file = st.file_uploader(
            "Upload a product image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of a product to find similar items"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 2])
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                if st.button("🔍 Find Similar Products", type="primary"):
                    with st.spinner("🔄 Analyzing image..."):
                        try:
                            # Generate embedding for uploaded image
                            image_embedding = get_image_embedding(uploaded_file)
                            
                            # Search for similar products
                            results, distances = search_engine.search_similar(
                                image_embedding,
                                search_type="combined",  # Use combined for image search
                                top_k=num_results
                            )
                            
                            if len(results) > 0:
                                st.success(f"✅ Found {len(results)} visually similar products!")
                                
                                # Display results
                                for idx, (_, product) in enumerate(results.iterrows()):
                                    similarity_score = product.get('similarity_score', 0)
                                    display_product_card(product, similarity_score)
                                
                                # Generate AI description
                                ai_description = generate_rag_description(
                                    "similar products to uploaded image", 
                                    results, 
                                    use_openai
                                )
                                st.markdown("### 🧠 AI Analysis")
                                st.markdown(ai_description)
                            
                            else:
                                st.warning("No visually similar products found.")
                                
                        except Exception as e:
                            st.error(f"Image search failed: {str(e)}")
    
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
