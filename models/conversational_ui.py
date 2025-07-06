"""
Conversational Commerce Interface for Streamlit
Implements chat-based product discovery and shopping
"""

import streamlit as st
from typing import Dict, List, Optional
import json

def initialize_conversational_interface():
    """Initialize conversation state in Streamlit session"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'commerce_engine' not in st.session_state:
        st.session_state.commerce_engine = None

def display_conversational_commerce(search_engine, enhanced_search=None, advanced_engine=None):
    """Display conversational commerce interface"""
    
    # Add dark mode friendly CSS
    st.markdown("""
    <style>
    /* Dark mode improvements for conversational UI */
    .chat-message-user {
        background-color: rgba(100, 150, 255, 0.2) !important;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
        color: #ffffff !important;
        border-left: 3px solid #4CAF50;
    }
    
    .chat-message-assistant {
        background-color: rgba(150, 100, 255, 0.2) !important;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        color: #ffffff !important;
        border-left: 3px solid #FF9800;
    }
    
    /* Ensure text inputs are visible */
    .stTextInput input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    .stTextInput input::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("💬 Chat with AI Shopping Assistant")
    st.markdown("Ask me anything about products! I can help you find, compare, and recommend items.")
    
    # Initialize conversation if needed
    initialize_conversational_interface()
    
    # Try to import and initialize commerce engine
    try:
        if st.session_state.commerce_engine is None:
            from models.agentic_ai import ConversationalCommerceEngine
            
            # Use enhanced search if available, fallback to basic
            search_to_use = enhanced_search if enhanced_search else search_engine
            st.session_state.commerce_engine = ConversationalCommerceEngine(
                search_to_use, advanced_engine
            )
            
            # Start new session
            st.session_state.session_id = st.session_state.commerce_engine.start_conversation()
            
            # Add welcome message
            if not st.session_state.conversation_history:
                welcome_msg = {
                    "role": "assistant",
                    "content": "👋 Hi! I'm your AI shopping assistant. I can help you find products, get recommendations, compare items, and more. What are you looking for today?",
                    "timestamp": "now"
                }
                st.session_state.conversation_history.append(welcome_msg)
        
        commerce_engine = st.session_state.commerce_engine
        
    except ImportError:
        st.error("❌ Conversational AI not available. Please ensure all dependencies are installed.")
        st.info("💡 You can still use the regular search interface below.")
        return False
    
    # Display conversation history
    display_conversation_history()
    
    # Chat input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message...",
            key="commerce_chat_input",
            placeholder="e.g., 'Show me blue shirts under $50' or 'I need a jacket for winter'"
        )
    
    with col2:
        send_button = st.button("Send 📨", type="primary")
    
    # Image upload for visual search in chat
    uploaded_image = st.file_uploader(
        "🖼️ Or upload an image to search",
        type=['png', 'jpg', 'jpeg'],
        key="chat_image_upload"
    )
    
    # Process user input
    if (send_button and user_input) or uploaded_image:
        process_chat_message(commerce_engine, user_input, uploaded_image)
    
    # Quick action buttons
    st.markdown("---")
    st.markdown("**🚀 Quick Actions:**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Browse Popular"):
            process_chat_message(commerce_engine, "Show me popular products", None)
    
    with col2:
        if st.button("💰 Deals Under $50"):
            process_chat_message(commerce_engine, "Show me products under $50", None)
    
    with col3:
        if st.button("👕 Men's Clothing"):
            process_chat_message(commerce_engine, "Show me men's clothing", None)
    
    with col4:
        if st.button("👗 Women's Fashion"):
            process_chat_message(commerce_engine, "Show me women's fashion", None)
    
    # Conversation insights
    if st.session_state.conversation_history:
        with st.expander("📊 Conversation Insights"):
            display_conversation_insights(commerce_engine)
    
    return True

def display_conversation_history():
    """Display the conversation history"""
    
    # Container for messages
    message_container = st.container()
    
    with message_container:
        for message in st.session_state.conversation_history:
            if message["role"] == "user":
                # User message with better dark mode styling
                st.markdown(
                    f"""
                    <div class="chat-message-user">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Show uploaded image if present
                if message.get("metadata", {}).get("has_image"):
                    st.markdown("🖼️ *Image uploaded*")
            
            else:
                # Assistant message with better dark mode styling
                st.markdown(
                    f"""
                    <div class="chat-message-assistant">
                        <strong>🤖 Assistant:</strong> {message["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display search results if present
                if message.get("metadata", {}).get("results"):
                    display_chat_search_results(message["metadata"]["results"])

def process_chat_message(commerce_engine, user_input: str, uploaded_image=None):
    """Process user chat message"""
    
    if not user_input and not uploaded_image:
        return
    
    # Prepare message content
    message_content = user_input if user_input else "I uploaded an image"
    
    # Prepare image data
    image_data = None
    if uploaded_image:
        image_data = uploaded_image.read()
    
    # Add user message to history
    user_message = {
        "role": "user",
        "content": message_content,
        "timestamp": "now",
        "metadata": {"has_image": uploaded_image is not None}
    }
    st.session_state.conversation_history.append(user_message)
    
    # Process through commerce engine
    try:
        with st.spinner("🤔 Thinking..."):
            response = commerce_engine.chat(
                st.session_state.session_id,
                message_content,
                image_data
            )
        
        # Add assistant response to history
        assistant_message = {
            "role": "assistant",
            "content": response["response"]["content"],
            "timestamp": "now",
            "metadata": response["response"].get("metadata", {})
        }
        st.session_state.conversation_history.append(assistant_message)
        
        # Clear input (remove the problematic line)
        # st.session_state.chat_input = ""  # This was causing the error
        
        # Rerun to update display
        st.rerun()
        
    except Exception as e:
        st.error(f"Sorry, I encountered an error: {str(e)}")
        st.info("💡 Try rephrasing your request or use the search interface above.")

def display_chat_search_results(results_metadata: Dict):
    """Display search results within chat"""
    
    if not results_metadata or not results_metadata.get("results"):
        return
    
    # Get results from different tools
    for tool_name, tool_results in results_metadata["results"].items():
        if tool_name in ["text_search", "image_search"] and tool_results:
            
            st.markdown("**🎯 Search Results:**")
            
            # Limit to top 3 results in chat
            top_results = tool_results[:3] if isinstance(tool_results, list) else []
            
            for i, result in enumerate(top_results, 1):
                with st.expander(f"{i}. {result.get('title', 'Product')} - ${result.get('price', 'N/A')}"):
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Product image
                        if result.get('image_url'):
                            try:
                                st.image(result['image_url'], width=150)
                            except:
                                st.info("📷 Image not available")
                        else:
                            st.info("📷 No image")
                    
                    with col2:
                        # Product details
                        st.markdown(f"**Price:** ${result.get('price', 'N/A')}")
                        if result.get('tags'):
                            st.markdown(f"**Category:** {result['tags']}")
                        if result.get('description'):
                            st.markdown(f"**Description:** {result['description'][:100]}...")
                        
                        # Similarity score
                        if result.get('similarity'):
                            confidence = result['similarity']
                            st.markdown(f"**Match Confidence:** {confidence:.1%}")
                        
                        # Action buttons
                        button_col1, button_col2 = st.columns(2)
                        with button_col1:
                            if st.button(f"Tell me more", key=f"more_{i}"):
                                more_info_query = f"Tell me more about {result.get('title', 'this product')}"
                                process_chat_message(st.session_state.commerce_engine, more_info_query, None)
                        
                        with button_col2:
                            if st.button(f"Find similar", key=f"similar_{i}"):
                                similar_query = f"Find products similar to {result.get('title', 'this')}"
                                process_chat_message(st.session_state.commerce_engine, similar_query, None)

def display_conversation_insights(commerce_engine):
    """Display insights about the conversation"""
    
    try:
        # Get conversation stats
        history = st.session_state.conversation_history
        user_messages = [msg for msg in history if msg["role"] == "user"]
        assistant_messages = [msg for msg in history if msg["role"] == "assistant"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Messages", len(history))
        
        with col2:
            st.metric("Your Questions", len(user_messages))
        
        with col3:
            st.metric("AI Responses", len(assistant_messages))
        
        # Get user preferences if available
        if st.session_state.session_id:
            preferences = commerce_engine.get_user_preferences(st.session_state.session_id)
            if preferences:
                st.markdown("**🎯 Learned Preferences:**")
                for key, value in preferences.items():
                    st.markdown(f"• **{key.title()}:** {value}")
        
        # Conversation actions
        st.markdown("**💬 Conversation Actions:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 New Conversation"):
                st.session_state.conversation_history = []
                st.session_state.session_id = None
                st.session_state.commerce_engine = None
                st.rerun()
        
        with col2:
            if st.button("📥 Download Chat"):
                chat_data = json.dumps(st.session_state.conversation_history, indent=2)
                st.download_button(
                    "Download as JSON",
                    chat_data,
                    "conversation.json",
                    "application/json"
                )
    
    except Exception as e:
        st.warning(f"Could not load conversation insights: {e}")

# Sample conversation starters
CONVERSATION_STARTERS = [
    "Show me blue shirts under $60",
    "I need a jacket for cold weather",
    "What's trending in women's fashion?",
    "Find me casual clothes for weekend",
    "I'm looking for formal wear",
    "Show me leather accessories",
    "What do you recommend for summer?",
    "Compare different jacket styles"
]

def display_conversation_starters():
    """Display suggested conversation starters"""
    
    st.markdown("**💡 Try asking:**")
    
    # Display starters as clickable buttons
    cols = st.columns(2)
    for i, starter in enumerate(CONVERSATION_STARTERS):
        col_idx = i % 2
        with cols[col_idx]:
            if st.button(starter, key=f"starter_{i}"):
                if st.session_state.commerce_engine:
                    process_chat_message(st.session_state.commerce_engine, starter, None)
