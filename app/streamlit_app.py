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
        get_image_embedding_simple
    )
    from models.rag_utils import get_search_engine, generate_rag_description
    MODELS_AVAILABLE = True
    get_text_embedding_func = get_text_embedding
    get_image_embedding_simple_func = get_image_embedding_simple
    get_search_engine_func = get_search_engine
    generate_rag_description_func = generate_rag_description
except ImportError as e:
    st.error(f"⚠️ Models not available: {e}")
    st.info("Please run the Jupyter notebook first to set up the embedding models.")
    MODELS_AVAILABLE = False
    get_text_embedding_func = None
    get_image_embedding_simple_func = None
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
    
    /* Dark mode improvements for chat */
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    
    /* Chat message styling for better visibility */
    [data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    
    /* User messages */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background-color: rgba(100, 150, 255, 0.1) !important;
        border-left: 3px solid #4CAF50 !important;
    }
    
    /* Assistant messages */
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background-color: rgba(150, 100, 255, 0.1) !important;
        border-left: 3px solid #FF9800 !important;
    }
    
    /* Chat input styling - make text visible in dark mode */
    .stChatInput > div > div > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: #ffffff !important;
    }
    
    .stChatInput input {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    .stChatInput input::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    
    /* Ensure chat message text is visible */
    [data-testid="stChatMessage"] p {
        color: inherit !important;
    }
    
    /* Fix for chat input text color in dark mode */
    .stChatInput textarea {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
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

# Legacy function removed - use display_enhanced_product_card() instead
# def display_product_card(product, similarity_score=None): ... [REMOVED]
            


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
            
            # Similarity score (now using percentage format since search engine returns 0-100)
            if similarity_score is not None:
                confidence_color = "green" if similarity_score > 80 else "orange" if similarity_score > 60 else "red"
                st.markdown(f"**📊 Match Confidence:** :{confidence_color}[{similarity_score:.1f}%]")
            
            # Description
            if product.get('description'):
                st.write(f"**Description:** {product['description']}")
            
            # Tags
            if product.get('tags'):
                st.write(f"**Category:** {product['tags']}")
            
            # Show semantic tags if available
            # Semantic tags functionality removed for simplification
            
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
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Search", "🖼️ Image Search", "📊 Analytics", "🤖 AI Assistant"])
    
    with tab1:
        st.subheader("Search by Description")
        
        query = st.text_input(
            "What are you looking for?",
            placeholder="e.g., blue leather jacket, women's summer top, casual shoes..."
        )
        
        if query:
            with st.spinner("🔍 Searching..."):
                try:
                    if get_text_embedding_func is None:
                        st.error("Text embedding function not available")
                        return
                    
                    query_embedding = get_text_embedding_func(query)
                    
                    # Search for similar products
                    results, _ = search_engine.search_similar(
                        query_embedding, 
                        search_type=search_type.replace("_only", ""), 
                        top_k=num_results,
                        text_threshold=55.0,   # Higher threshold for text search (very precise)
                        image_threshold=25.0   # Lower threshold for image search (better recall)
                    )
                    
                    if len(results) > 0:
                        st.success(f"✅ Found {len(results)} matching products!")
                        
                        # Display results
                        st.subheader("🎯 Search Results")
                        for _, product in results.iterrows():
                            similarity_score = product.get('similarity_score', 0)
                            display_enhanced_product_card(product, similarity_score, show_semantic_tags=False)
                        
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
                        st.info("💡 Try using simpler keywords or different search terms.")
                        
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                    st.info("💡 Try using simpler keywords or check if the search index is properly loaded.")
    
    with tab2:
        st.subheader("Search by Image")
        
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
                    
                    if get_image_embedding_simple_func is None:
                        st.error("Image embedding function not available")
                        return
                    
                    image_embedding = get_image_embedding_simple_func(uploaded_file)
                    st.success(f"✅ Image processed successfully (embedding: {image_embedding.shape[0]}D)")
                    
                    results, _ = search_engine.search_similar(
                        image_embedding,
                        search_type="image",  # Use image index for image search
                        top_k=num_results,
                        text_threshold=55.0,   # Higher threshold for text search
                        image_threshold=25.0   # Lower threshold for image search (optimized for visual similarity)
                    )
                    
                    if len(results) > 0:
                        st.success(f"✅ Found {len(results)} visually similar products!")
                        
                        # Display results
                        for _, product in results.iterrows():
                            similarity_score = product.get('similarity_score', 0)
                            display_enhanced_product_card(product, similarity_score, show_semantic_tags=False)
                        
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
    
    with tab4:
        st.subheader("🤖 AI Shopping Assistant")
        
        # Chat controls
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🗑️ Clear Chat", help="Clear conversation history"):
                if "ai_chat_messages" in st.session_state:
                    st.session_state.ai_chat_messages = []
                if "conversational_agent" in st.session_state and st.session_state.conversational_agent:
                    st.session_state.conversational_agent.clear_context()
                if "input_counter" in st.session_state:
                    st.session_state.input_counter = 0
                st.rerun()
        
        # Import and display conversational commerce interface
        st.markdown("### 🤖 AI Shopping Assistant (Advanced Conversational AI)")
        st.markdown("*I can have natural conversations and help you find products! Ask me anything.*")
        
        try:
            # Initialize conversational agent directly
            if "conversational_agent" not in st.session_state:
                try:
                    from models.conversational_agent import ConversationalAgent
                    st.session_state.conversational_agent = ConversationalAgent(search_engine)
                    st.success("🤖 Advanced Conversational AI loaded!")
                except ImportError as e:
                    st.error(f"❌ Failed to import ConversationalAgent: {e}")
                    st.session_state.conversational_agent = None
                except Exception as e:
                    st.error(f"❌ Failed to initialize ConversationalAgent: {e}")
                    st.session_state.conversational_agent = None
            
            # Show status
            if st.session_state.conversational_agent:
                st.success("🤖 Advanced Conversational AI: **ACTIVE**")
            else:
                st.warning("⚠️ Basic Responses Only: **Advanced AI not available**")
            
            # Initialize chat history with unique key
            if "ai_chat_messages" not in st.session_state:
                st.session_state.ai_chat_messages = []
                # Add welcome message
                if st.session_state.conversational_agent:
                    welcome = st.session_state.conversational_agent.respond("hello")
                    st.session_state.ai_chat_messages.append({"role": "assistant", "content": welcome})
                else:
                    st.session_state.ai_chat_messages.append({
                        "role": "assistant", 
                        "content": "Hi! 👋 I'm your AI shopping assistant. I can help you find products and have casual conversations. What are you looking for today?"
                    })
            
            # Initialize input counter for clearing
            if "input_counter" not in st.session_state:
                st.session_state.input_counter = 0
            
            # Display chat messages
            for message in st.session_state.ai_chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Display uploaded image if user uploaded one
                    if message["role"] == "user" and message.get("uploaded_image"):
                        st.image(message["uploaded_image"], caption="Your uploaded image", width=200)
                    
                    # Display product images if available (from AI recommendations)
                    if message.get("products"):
                        st.markdown("**🛍️ Product Recommendations:**")
                        cols = st.columns(min(len(message["products"]), 3))  # Max 3 columns
                        
                        for i, product in enumerate(message["products"]):
                            with cols[i % 3]:
                                if product.get("image_url"):
                                    try:
                                        # Display product image
                                        st.image(
                                            product["image_url"], 
                                            caption=f"{product['title']}\n{product['price']}", 
                                            use_container_width=True
                                        )
                                    except Exception as e:
                                        st.warning(f"⚠️ Could not load image for {product['title']}")
                                        st.text(f"🏷️ {product['title']}\n💰 {product['price']}")
                                else:
                                    # Fallback for products without images
                                    st.markdown(f"""
                                    **{product['title']}**  
                                    💰 {product['price']}  
                                    🏷️ {product['tags']}
                                    """)
                        
                        st.markdown("---")
            
            # Image upload for visual search in chat
            st.markdown("**📸 Upload an image to search for similar products:**")
            uploaded_image = st.file_uploader(
                "Choose an image file (JPG, PNG, JPEG)", 
                type=['jpg', 'jpeg', 'png'],
                help="Upload a photo and I'll find visually similar products!",
                key=f"chat_image_upload_{st.session_state.input_counter}"
            )
            
            # Chat input with key that changes to clear the input
            chat_key = f"ai_chat_input_{st.session_state.input_counter}"
            user_input = st.chat_input("Type anything - ask about products, say hi, or just chat...", key=chat_key)
            
            # Handle image upload
            if uploaded_image is not None:
                # Increment counter to clear input field on next render
                st.session_state.input_counter += 1
                
                # Read image data
                image_data = uploaded_image.read()
                
                # Add user message showing image was uploaded WITH the actual image
                st.session_state.ai_chat_messages.append({
                    "role": "user", 
                    "content": f"📸 *Uploaded image: {uploaded_image.name}*\n\nCan you find products similar to this image?",
                    "uploaded_image": uploaded_image  # Store the uploaded image to display
                })
                
                # Generate response using image search
                with st.spinner("🔍 Analyzing your image..."):
                    if st.session_state.conversational_agent:
                        # Use the conversational agent's image search capability
                        response = st.session_state.conversational_agent.respond_with_image(
                            "Find products similar to this image", 
                            image_data
                        )
                        
                        # Handle structured response for image search
                        if isinstance(response, dict):
                            response_text = response.get('text', str(response))
                            products = response.get('products', [])
                            st.session_state.ai_chat_messages.append({
                                "role": "assistant", 
                                "content": response_text,
                                "products": products
                            })
                        else:
                            st.session_state.ai_chat_messages.append({"role": "assistant", "content": response})
                    else:
                        # Fallback response
                        response = """I can see you've uploaded an image! 📸 

For the best image search results, I recommend using the **Image Search tab** above where you can:
• Upload your photo
• See visual similarity scores
• Get detailed product comparisons
• View full-size images

What type of product are you looking for in your image?"""
                        st.session_state.ai_chat_messages.append({"role": "assistant", "content": response})
                
                st.rerun()
            
            if user_input:
                # Increment counter to clear input field on next render
                st.session_state.input_counter += 1
                
                # Add user message
                st.session_state.ai_chat_messages.append({"role": "user", "content": user_input})
                
                # Generate response
                with st.spinner("🤔 Thinking..."):
                    if st.session_state.conversational_agent:
                        # Use the conversational agent's enhanced response method
                        response = st.session_state.conversational_agent.respond_with_images(user_input)
                        
                        # Handle structured response
                        if isinstance(response, dict):
                            response_text = response.get('text', str(response))
                            products = response.get('products', [])
                            response_type = response.get('type', 'text')
                            
                            st.session_state.ai_chat_messages.append({
                                "role": "assistant", 
                                "content": response_text,
                                "products": products if response_type == 'product_search' else []
                            })
                        else:
                            # Fallback for simple string response
                            st.session_state.ai_chat_messages.append({"role": "assistant", "content": response})
                    else:
                        # Fallback to basic responses
                        if user_input.lower() in ['hello', 'hi', 'hey']:
                            response = "Hi there! 👋 I'm your AI shopping assistant. What can I help you find today?"
                        elif 'thank' in user_input.lower():
                            response = "You're welcome! 😊 Is there anything else I can help you with?"
                        elif search_engine and get_text_embedding_func and any(word in user_input.lower() for word in ['shirt', 'jacket', 'pants', 'dress', 'shoes']):
                            # Try to search
                            try:
                                query_embedding = get_text_embedding_func(user_input)
                                results, _ = search_engine.search_similar(query_embedding, search_type="text", top_k=3, text_threshold=55.0, image_threshold=25.0)
                                
                                if len(results) > 0:
                                    response = f"I found some great options for '{user_input}'! 🎉\n\n"
                                    for _, product in results.head(2).iterrows():
                                        response += f"• **{product['title']}** - ${product['price']}\n"
                                    response += "\n💡 Check the Text Search tab above for full details and images!"
                                else:
                                    response = f"I couldn't find exact matches for '{user_input}', but I'd love to help! Try using broader terms like 'shirt' or 'jacket'. What style are you going for?"
                            except:
                                response = f"I'd love to help you find '{user_input}'! For the best results with images and details, try the Text Search tab above. What specific style or features are you looking for?"
                        else:
                            response = f"That's interesting! I'm here to help you find products. Are you looking for something specific to shop for, or would you like me to show you what's available?"
                        
                        st.session_state.ai_chat_messages.append({"role": "assistant", "content": response})
                
                st.rerun()
            
        except Exception as e:
            st.error(f"Error loading conversational interface: {e}")
            st.info("Please ensure all required dependencies are installed.")
    
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
