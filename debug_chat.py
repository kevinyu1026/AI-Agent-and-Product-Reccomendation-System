import streamlit as st
import sys
import os

# Add the parent directory to the path to import models
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

st.title("🧪 Conversational Agent Debug Test")

# Test import
try:
    from models.conversational_agent import ConversationalAgent
    st.success("✅ ConversationalAgent imported successfully!")
    
    # Test initialization
    try:
        agent = ConversationalAgent()
        st.success("✅ ConversationalAgent initialized successfully!")
        
        # Test response
        test_message = st.text_input("Test message:", value="Hello")
        if st.button("Test Response"):
            response = agent.respond(test_message)
            st.write("**Response:**", response)
            
            # Show intent detection
            intent = agent.get_intent(test_message)
            st.write("**Detected Intent:**", intent)
            
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {e}")
        
except ImportError as e:
    st.error(f"❌ Failed to import ConversationalAgent: {e}")
    st.info("Make sure you're running from the correct directory and the models folder exists.")

# Show current working directory
st.write("**Current working directory:**", os.getcwd())
st.write("**Python path:**", sys.path[-3:])  # Show last 3 entries
