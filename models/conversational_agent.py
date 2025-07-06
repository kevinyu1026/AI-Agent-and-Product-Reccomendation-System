"""
Advanced Conversational AI Agent for Product Recommendations
Provides natural conversation capabilities similar to ChatGPT
"""

import random
from typing import List, Dict, Optional
import re

class ConversationalAgent:
    """
    Advanced conversational AI agent that can handle casual conversation
    and provide intelligent product recommendations
    """
    
    def __init__(self, search_engine=None):
        self.search_engine = search_engine
        self.conversation_context = []
        self.user_preferences = {}
        
        # Casual conversation starters and responses
        self.greetings = [
            "Hi there! 👋 I'm your AI shopping assistant. What can I help you find today?",
            "Hello! 😊 I'm here to help you discover amazing products. What are you looking for?",
            "Hey! 🛍️ Ready to find some great products? What's on your shopping list?",
            "Hi! I'm your personal shopping AI. Tell me what you need and I'll help you find it!",
        ]
        
        self.casual_responses = {
            "how are you": [
                "I'm doing great, thanks for asking! 😊 I'm excited to help you find some awesome products today. What are you shopping for?",
                "I'm fantastic! Ready to help you discover some amazing items. What's caught your interest lately?",
                "I'm doing wonderful! I love helping people find exactly what they're looking for. What can I help you with?"
            ],
            "thank you": [
                "You're so welcome! 😊 Is there anything else I can help you find?",
                "My pleasure! I'm here whenever you need shopping advice.",
                "Happy to help! Let me know if you want to explore more products."
            ],
            "hello": self.greetings,
            "hi": self.greetings,
            "hey": self.greetings,
        }
        
        # Shopping conversation patterns
        self.shopping_keywords = {
            'clothing': ['shirt', 'pants', 'dress', 'jacket', 'top', 'clothing', 'wear', 'outfit'],
            'accessories': ['shoes', 'bag', 'belt', 'hat', 'jewelry', 'watch'],
            'colors': ['red', 'blue', 'green', 'black', 'white', 'yellow', 'pink', 'purple', 'orange'],
            'occasions': ['work', 'party', 'casual', 'formal', 'wedding', 'date', 'gym', 'beach'],
            'seasons': ['summer', 'winter', 'spring', 'fall', 'autumn'],
            'materials': ['cotton', 'leather', 'silk', 'wool', 'denim']
        }
    
    def get_intent(self, message: str) -> str:
        """Analyze user message to determine intent"""
        message_lower = message.lower()
        
        # Check for questions about the system/AI FIRST (most specific)
        if any(pattern in message_lower for pattern in ['what are you', 'what do you do', 'tell me about yourself', 'who are you', 'what is this']):
            return 'question'
        
        # Check for help requests
        if any(question in message_lower for question in ['help', 'what can you do', 'how does this work', 'can you help']):
            return 'help'
        
        # Check for thanks
        if any(thanks in message_lower for thanks in ['thank', 'thanks', 'appreciate', 'thx']):
            return 'thanks'
        
        # Check for casual conversation
        if any(casual in message_lower for casual in ['how are you', 'what\'s up', 'how\'s it going', 'how do you do', 'what up']):
            return 'casual'
        
        # Check for greetings (simple ones)
        if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'sup', 'yo']) and len(message.split()) <= 2:
            return 'greeting'
        
        # Check for product search (specific fashion terms)
        if any(keyword in message_lower for category in self.shopping_keywords.values() for keyword in category):
            return 'product_search'
        
        # Check for general questions with question words
        if any(word in message_lower for word in ['what', 'how', 'why', 'where', 'when', 'who']) and not any(keyword in message_lower for category in self.shopping_keywords.values() for keyword in category):
            return 'question'
        
        # If it's a short message, treat as casual
        if len(message.split()) <= 3:
            return 'casual'
        
        # Default to product search for longer messages
        return 'product_search'
    
    def generate_casual_response(self, intent: str, message: str) -> str:
        """Generate natural, casual responses"""
        
        if intent == 'greeting':
            return random.choice(self.greetings)
        
        elif intent == 'thanks':
            return random.choice(self.casual_responses['thank you'])
        
        elif intent == 'casual':
            return random.choice(self.casual_responses['how are you'])
        
        elif intent == 'help':
            return """I'm your AI shopping assistant! Here's what I can do:

🔍 **Product Search**: Tell me what you're looking for (e.g., "blue shirt", "leather jacket")
💬 **Casual Chat**: I love chatting! Ask me about products, styles, or just say hi
🎯 **Smart Recommendations**: I'll suggest products based on your preferences
📊 **Product Details**: I can tell you about prices, descriptions, and features

Just tell me what you're interested in and I'll help you find it! What would you like to explore today?"""
        
        elif intent == 'question':
            return self.handle_general_question(message)
        
        else:
            return self.handle_product_search(message)
    
    def handle_general_question(self, message: str) -> str:
        """Handle general questions and conversation"""
        message_lower = message.lower()
        
        if 'you' in message_lower and any(word in message_lower for word in ['are', 'do', 'can', 'what']):
            return """I'm an AI shopping assistant created to help you find amazing products! 🤖

I can:
• Search for products you describe
• Have casual conversations 
• Give shopping advice and recommendations
• Help you explore different styles and options

I'm here to make shopping fun and easy! What would you like to chat about or shop for? 😊"""
        
        elif any(word in message_lower for word in ['boring', 'stupid', 'dumb', 'bad']):
            return "I'm sorry you feel that way! 😔 I'm always trying to improve. What would make our conversation more helpful or interesting for you?"
        
        elif any(word in message_lower for word in ['good', 'great', 'awesome', 'cool', 'nice']):
            return "Thank you so much! 😊 That really means a lot. I love helping people find great products. Is there anything specific you'd like to shop for today?"
        
        elif 'weather' in message_lower:
            return "I don't know about the weather, but I can help you find clothes for any weather! ☀️🌧️ Are you looking for something for hot, cold, or rainy weather?"
        
        elif any(word in message_lower for word in ['time', 'date', 'day']):
            return "I'm not great with time, but I'm great with timeless fashion! 😄 Are you looking for something for a special occasion or just everyday wear?"
        
        else:
            return f"That's an interesting question! I'm focused on helping with shopping and product recommendations. Speaking of which, is there anything you'd like to shop for? I can help you find '{message.lower()}' or something similar! 🛍️"
    
    def handle_product_search(self, query: str) -> str:
        """Handle product search queries with natural conversation"""
        
        try:
            if not self.search_engine:
                return f"""I'd love to help you find '{query}'! 

Unfortunately, I need the search engine to be connected to give you specific results. You can:

🔍 **Use the Text Search tab** above to get detailed product results with images
📱 **Try the Image Search** if you have a photo of what you want
📊 **Check Analytics** to see what products are available

What specific type of product are you most interested in? I can give you some general shopping advice!"""
            
            # Try to perform actual search
            from models.embed_utils import get_text_embedding
            
            try:
                query_embedding = get_text_embedding(query)
                results, _ = self.search_engine.search_similar(
                    query_embedding, 
                    search_type="text", 
                    top_k=3
                )
                
                if len(results) > 0:
                    response = f"Great choice! I found some awesome options for '{query}' 🎉\n\n"
                    
                    for i, (_, product) in enumerate(results.head(3).iterrows(), 1):
                        price_text = f"${product['price']}" if product['price'] != 'N/A' else "Price varies"
                        response += f"**{i}. {product['title']}**\n"
                        response += f"   💰 {price_text}\n"
                        response += f"   🏷️ {product['tags']}\n\n"
                    
                    response += "💡 **Want to see images and more details?** Check out the Text Search tab above!\n\n"
                    response += "What do you think of these options? Need help narrowing it down or looking for something else?"
                    
                    return response
                
                else:
                    return self.generate_no_results_response(query)
                    
            except Exception as e:
                return f"I understand you're looking for '{query}' - that sounds interesting! 🤔\n\nTo get the best results with images and detailed info, try using the **Text Search tab** above. \n\nIn the meantime, what specific features are most important to you? Color, style, price range?"
        
        except Exception as e:
            return self.generate_fallback_response(query)
    
    def generate_no_results_response(self, query: str) -> str:
        """Generate helpful response when no products found"""
        suggestions = []
        
        # Analyze query for suggestions
        query_lower = query.lower()
        
        if any(color in query_lower for color in self.shopping_keywords['colors']):
            suggestions.append("Try searching without the color")
        
        if 'for' in query_lower:
            suggestions.append("Try using simpler keywords")
        
        response = f"Hmm, I couldn't find exact matches for '{query}' in our current catalog. 🤔\n\n"
        
        if suggestions:
            response += "**Here are some tips:**\n"
            for suggestion in suggestions:
                response += f"• {suggestion}\n"
            response += "\n"
        
        response += "**Try searching for:**\n"
        response += "• 'shirt' or 'jacket' (basic categories)\n"
        response += "• 'blue' or 'black' (colors)\n"
        response += "• 'men' or 'women' (gender)\n\n"
        response += "What type of item were you most interested in? I can help you explore our catalog! 😊"
        
        return response
    
    def generate_fallback_response(self, query: str) -> str:
        """Generate fallback response when search fails"""
        return f"""I'd love to help you find '{query}'! 😊

While I can chat with you here, for the best shopping experience with **images, detailed descriptions, and advanced search**, I recommend using:

🔍 **Text Search tab** - Full product search with images
🖼️ **Image Search tab** - Upload photos to find similar items  
📊 **Analytics tab** - Browse all available products

What kind of style or occasion are you shopping for? I'm happy to chat and give you shopping advice! 🛍️"""

    def respond(self, message: str) -> str:
        """Main response method - generates conversational responses"""
        
        # Add to conversation context
        self.conversation_context.append({"role": "user", "content": message})
        
        # Determine intent and generate response
        intent = self.get_intent(message)
        response = self.generate_casual_response(intent, message)
        
        # Add response to context
        self.conversation_context.append({"role": "assistant", "content": response})
        
        # Keep context manageable
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
        
        return response
    
    def clear_context(self):
        """Clear conversation context"""
        self.conversation_context = []
        self.user_preferences = {}
