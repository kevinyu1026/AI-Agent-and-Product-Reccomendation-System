"""
Agentic AI Framework for Product Recommendation System
Implements memory, tools, planning, and conversational capabilities
"""

from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid

@dataclass
class ConversationMemory:
    """Manages conversation history and context"""
    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    cart_items: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to conversation history"""
        message = {
            "role": role,  # "user" or "assistant"
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
    
    def add_search(self, query: str, results: List[Dict], search_type: str = "text"):
        """Record search in history"""
        search_record = {
            "query": query,
            "search_type": search_type,
            "results_count": len(results),
            "timestamp": datetime.now().isoformat(),
            "top_result": results[0] if results else None
        }
        self.search_history.append(search_record)
    
    def update_preferences(self, new_preferences: Dict[str, Any]):
        """Update user preferences based on interactions"""
        self.user_preferences.update(new_preferences)
    
    def get_recent_context(self, message_count: int = 5) -> List[Dict]:
        """Get recent conversation context"""
        return self.messages[-message_count:] if self.messages else []

@dataclass
class AgentTool:
    """Represents a tool the agent can use"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)

class ConversationalAgent:
    """Main conversational AI agent for product recommendations"""
    
    def __init__(self, search_engine, recommendation_engine=None):
        self.search_engine = search_engine
        self.recommendation_engine = recommendation_engine
        self.sessions: Dict[str, ConversationMemory] = {}
        self.tools = self._initialize_tools()
        
    def _initialize_tools(self) -> Dict[str, AgentTool]:
        """Initialize available tools for the agent"""
        tools = {}
        
        # Search Tools
        tools["text_search"] = AgentTool(
            name="text_search",
            description="Search products using text query",
            function=self._tool_text_search,
            parameters={"query": "string", "max_results": "integer"}
        )
        
        tools["image_search"] = AgentTool(
            name="image_search", 
            description="Search products using uploaded image",
            function=self._tool_image_search,
            parameters={"image_data": "binary", "max_results": "integer"}
        )
        
        tools["filter_search"] = AgentTool(
            name="filter_search",
            description="Search with specific filters (price, category, etc.)",
            function=self._tool_filter_search,
            parameters={"filters": "dict", "max_results": "integer"}
        )
        
        # Recommendation Tools
        tools["get_recommendations"] = AgentTool(
            name="get_recommendations",
            description="Get personalized product recommendations",
            function=self._tool_get_recommendations,
            parameters={"user_preferences": "dict", "context": "string"}
        )
        
        tools["compare_products"] = AgentTool(
            name="compare_products",
            description="Compare multiple products side by side",
            function=self._tool_compare_products,
            parameters={"product_ids": "list"}
        )
        
        # Commerce Tools
        tools["add_to_cart"] = AgentTool(
            name="add_to_cart",
            description="Add product to shopping cart",
            function=self._tool_add_to_cart,
            parameters={"product_id": "string", "quantity": "integer"}
        )
        
        tools["check_availability"] = AgentTool(
            name="check_availability",
            description="Check product availability and pricing",
            function=self._tool_check_availability,
            parameters={"product_id": "string"}
        )
        
        return tools
    
    def create_session(self) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = ConversationMemory(session_id=session_id)
        return session_id
    
    def process_message(self, session_id: str, user_message: str, 
                       image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """Process user message and generate response"""
        
        # Get or create session
        if session_id not in self.sessions:
            session_id = self.create_session()
        
        memory = self.sessions[session_id]
        
        # Add user message to memory
        memory.add_message("user", user_message, {"has_image": image_data is not None})
        
        # Analyze user intent
        intent_analysis = self._analyze_intent(user_message, memory)
        
        # Plan response using available tools
        response_plan = self._plan_response(intent_analysis, memory, image_data)
        
        # Execute plan and generate response
        response = self._execute_plan(response_plan, memory)
        
        # Add assistant response to memory
        memory.add_message("assistant", response["content"], response.get("metadata", {}))
        
        return {
            "session_id": session_id,
            "response": response,
            "intent": intent_analysis,
            "context": memory.get_recent_context()
        }
    
    def _analyze_intent(self, message: str, memory: ConversationMemory) -> Dict[str, Any]:
        """Analyze user intent from message and context"""
        message_lower = message.lower()
        
        intent = {
            "primary_intent": "unknown",
            "entities": {},
            "confidence": 0.5,
            "requires_tools": []
        }
        
        # Search intent patterns
        search_patterns = [
            "looking for", "find", "search", "want", "need", "show me",
            "recommend", "suggest", "help me find"
        ]
        
        if any(pattern in message_lower for pattern in search_patterns):
            intent["primary_intent"] = "search"
            intent["requires_tools"].append("text_search")
            intent["confidence"] = 0.8
            
            # Extract entities
            if "under" in message_lower or "$" in message:
                intent["entities"]["price_constraint"] = True
            if any(color in message_lower for color in ["red", "blue", "green", "black", "white"]):
                intent["entities"]["color_specified"] = True
            if any(cat in message_lower for cat in ["shirt", "jacket", "dress", "pants"]):
                intent["entities"]["category_specified"] = True
        
        # Compare intent
        elif any(word in message_lower for word in ["compare", "difference", "versus", "vs"]):
            intent["primary_intent"] = "compare"
            intent["requires_tools"].append("compare_products")
            intent["confidence"] = 0.9
        
        # Purchase intent
        elif any(word in message_lower for word in ["buy", "purchase", "add to cart", "order"]):
            intent["primary_intent"] = "purchase"
            intent["requires_tools"].append("add_to_cart")
            intent["confidence"] = 0.9
        
        # Recommendation intent
        elif any(word in message_lower for word in ["recommend", "suggest", "what should"]):
            intent["primary_intent"] = "recommendation"
            intent["requires_tools"].append("get_recommendations")
            intent["confidence"] = 0.8
        
        return intent
    
    def _plan_response(self, intent: Dict, memory: ConversationMemory, 
                      image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """Plan the response based on intent and available tools"""
        
        plan = {
            "steps": [],
            "tools_to_use": intent["requires_tools"],
            "response_type": "text",
            "data_needed": {}
        }
        
        if intent["primary_intent"] == "search":
            if image_data:
                plan["steps"].append(("image_search", {"image_data": image_data}))
            else:
                plan["steps"].append(("text_search", {"query": memory.messages[-1]["content"]}))
            
            # Add recommendation step if user has preferences
            if memory.user_preferences:
                plan["steps"].append(("get_recommendations", {"context": "search"}))
        
        elif intent["primary_intent"] == "recommendation":
            plan["steps"].append(("get_recommendations", {
                "user_preferences": memory.user_preferences,
                "context": "general"
            }))
        
        elif intent["primary_intent"] == "compare":
            # Extract product references from conversation
            plan["steps"].append(("compare_products", {"from_context": True}))
        
        elif intent["primary_intent"] == "purchase":
            plan["steps"].append(("add_to_cart", {"from_context": True}))
        
        return plan
    
    def _execute_plan(self, plan: Dict, memory: ConversationMemory) -> Dict[str, Any]:
        """Execute the planned response"""
        
        results = {}
        response_parts = []
        
        for step_name, step_params in plan["steps"]:
            if step_name in self.tools:
                tool = self.tools[step_name]
                try:
                    result = tool.function(**step_params)
                    results[step_name] = result
                    
                    # Generate response text for this step
                    response_text = self._format_tool_response(step_name, result)
                    response_parts.append(response_text)
                    
                except Exception as e:
                    response_parts.append(f"Sorry, I encountered an error with {step_name}: {str(e)}")
        
        # Combine response parts
        if response_parts:
            final_response = "\\n\\n".join(response_parts)
        else:
            final_response = "I understand your request, but I'm not sure how to help with that specific query. Could you please rephrase or ask for something specific like finding products, getting recommendations, or comparing items?"
        
        return {
            "content": final_response,
            "metadata": {
                "tools_used": list(results.keys()),
                "results": results
            }
        }
    
    def _format_tool_response(self, tool_name: str, result: Any) -> str:
        """Format tool results into natural language"""
        
        if tool_name == "text_search":
            if result and len(result) > 0:
                count = len(result)
                top_item = result[0]
                return f"I found {count} products matching your search. The top result is **{top_item.get('title', 'Unknown')}** for ${top_item.get('price', 'N/A')}."
            else:
                return "I couldn't find any products matching your search. Try using different keywords or browse our categories."
        
        elif tool_name == "image_search":
            if result and len(result) > 0:
                count = len(result)
                return f"Based on your image, I found {count} visually similar products. Let me show you the best matches."
            else:
                return "I couldn't find products similar to your uploaded image. Try a different image or describe what you're looking for."
        
        elif tool_name == "get_recommendations":
            if result and len(result) > 0:
                count = len(result)
                return f"Based on your preferences and browsing history, I have {count} personalized recommendations for you."
            else:
                return "I'd love to give you personalized recommendations! Could you tell me more about your style preferences?"
        
        return f"Completed {tool_name} successfully."
    
    # Tool Implementation Methods
    def _tool_text_search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Execute text search tool"""
        try:
            if hasattr(self.search_engine, 'enhanced_text_search'):
                return self.search_engine.enhanced_text_search(query, max_results)
            else:
                # Fallback to basic search
                from models.embed_utils import get_text_embedding
                query_emb = get_text_embedding(query)
                results, _ = self.search_engine.search_similar(query_emb, "text", max_results)
                return results.to_dict('records') if not results.empty else []
        except Exception as e:
            return []
    
    def _tool_image_search(self, image_data: bytes, max_results: int = 5) -> List[Dict]:
        """Execute image search tool"""
        try:
            if hasattr(self.search_engine, 'enhanced_image_search'):
                return self.search_engine.enhanced_image_search(image_data, max_results)
            else:
                # Fallback to basic image search
                from models.embed_utils import get_image_embedding_simple
                from io import BytesIO
                img_emb = get_image_embedding_simple(BytesIO(image_data))
                results, _ = self.search_engine.search_similar(img_emb, "image", max_results)
                return results.to_dict('records') if not results.empty else []
        except Exception as e:
            return []
    
    def _tool_filter_search(self, filters: Dict, max_results: int = 5) -> List[Dict]:
        """Execute filtered search tool"""
        # Implementation for filtered search
        return []
    
    def _tool_get_recommendations(self, user_preferences: Optional[Dict] = None, 
                                 context: str = "general") -> List[Dict]:
        """Execute recommendation tool"""
        try:
            if self.recommendation_engine and hasattr(self.recommendation_engine, 'get_hybrid_recommendations'):
                return self.recommendation_engine.get_hybrid_recommendations(
                    query="personalized recommendations",
                    user_preferences=user_preferences,
                    top_k=5
                )
            return []
        except Exception as e:
            return []
    
    def _tool_compare_products(self, product_ids: Optional[List] = None, from_context: bool = False) -> Dict:
        """Execute product comparison tool"""
        # Implementation for product comparison
        return {"comparison": "Feature not yet implemented"}
    
    def _tool_add_to_cart(self, product_id: Optional[str] = None, quantity: int = 1, 
                         from_context: bool = False) -> Dict:
        """Execute add to cart tool"""
        # Implementation for cart functionality
        return {"status": "Feature not yet implemented"}
    
    def _tool_check_availability(self, product_id: str) -> Dict:
        """Execute availability check tool"""
        # Implementation for availability checking
        return {"available": True, "stock": "In stock"}

class ConversationalCommerceEngine:
    """High-level conversational commerce interface"""
    
    def __init__(self, search_engine, recommendation_engine=None):
        self.agent = ConversationalAgent(search_engine, recommendation_engine)
        self.active_sessions = {}
    
    def start_conversation(self) -> str:
        """Start a new conversation session"""
        session_id = self.agent.create_session()
        self.active_sessions[session_id] = {
            "started_at": datetime.now(),
            "message_count": 0
        }
        return session_id
    
    def chat(self, session_id: str, message: str, image_data: Optional[bytes] = None) -> Dict:
        """Main chat interface"""
        
        # Update session stats
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["message_count"] += 1
        
        # Process message through agent
        response = self.agent.process_message(session_id, message, image_data)
        
        return response
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get full conversation history"""
        if session_id in self.agent.sessions:
            return self.agent.sessions[session_id].messages
        return []
    
    def get_user_preferences(self, session_id: str) -> Dict:
        """Get learned user preferences"""
        if session_id in self.agent.sessions:
            return self.agent.sessions[session_id].user_preferences
        return {}
    
    def update_user_preferences(self, session_id: str, preferences: Dict):
        """Update user preferences"""
        if session_id in self.agent.sessions:
            self.agent.sessions[session_id].update_preferences(preferences)
