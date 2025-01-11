import streamlit as st
import re
from openai import OpenAI
import datetime
from typing import Dict, TypedDict, Annotated, Sequence, Tuple, List, Union, Any, Callable
from langgraph.graph import Graph, StateGraph
import operator
import json
from tavily import TavilyClient
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage

# Configure page settings
st.set_page_config(
    page_title="ReAct Agent with LangGraph",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session states
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None
if 'tavily_client' not in st.session_state:
    st.session_state.tavily_client = None
if 'api_keys_submitted' not in st.session_state:
    st.session_state.api_keys_submitted = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

def initialize_clients(openai_api_key: str, tavily_api_key: str):
    """Initialize API clients with provided keys."""
    try:
        st.session_state.openai_client = OpenAI(api_key=openai_api_key)
        st.session_state.tavily_client = TavilyClient(api_key=tavily_api_key)
        st.session_state.api_keys_submitted = True
        return True
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        return False

# Define the agent state
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    next_step: str
    done: bool

# Function for tools
def run_tool(tool_name: str, tool_input: str) -> str:
    """Execute a tool with given input."""
    if tool_name == "calculate":
        try:
            if any(char in tool_input for char in "[];'\""):
                raise ValueError("Invalid characters in expression")
            return str(eval(tool_input))
        except Exception as e:
            return f"Error in calculation: {e}"
            
    elif tool_name == "average_dog_weight":
        # Define breeds with standardized weights in pounds
        breeds = {
            "scottish terrier": 20,
            "border collie": 37,
            "toy poodle": 7,
            "bulldog": 51
        }
        
        breed_input = tool_input.lower().strip()
        
        # Improved breed matching
        matched_breed = None
        for known_breed in breeds:
            if known_breed in breed_input or breed_input in known_breed:
                matched_breed = known_breed
                break
        
        if matched_breed:
            weight = breeds[matched_breed]
            # Return a clearly formatted string with both pounds and kilograms
            return f"A {matched_breed.title()} averages {weight} lbs ({round(weight * 0.453592, 1)} kg)"
        
        return "I don't have specific weight information for this breed. An average dog weights 50 lbs (22.7 kg)"
        
    elif tool_name == "web_search":
        try:
            search_result = st.session_state.tavily_client.search(tool_input)
            results = search_result[:3]
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", "No title"),
                    "content": result.get("content", "No content")[:200] + "...",
                    "url": result.get("url", "No URL")
                })
            return json.dumps(formatted_results, indent=2)
        except Exception as e:
            return f"Error during web search: {str(e)}"
            
    return f"Unknown tool: {tool_name}"

# Update the agent prompt to be more specific about units
AGENT_PROMPT = """You are a helpful AI assistant that can think and act to answer questions.
You run in a loop of Thought, Action, Observation until you have enough information to provide an Answer.

Use the following format:
Thought: your reasoning about what to do next
Action: tool_name: input
or
Answer: your final answer when you have enough information

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number

average_dog_weight:
e.g. average_dog_weight: Collie
Returns average weight of a dog breed in both pounds and kilograms

web_search:
e.g. web_search: what is the average lifespan of a Border Collie
Searches the web for information

Remember to:
1. Always specify both pounds and kilograms when discussing weight
2. Double-check the observation output before providing final answers
3. If a breed isn't found, conduct a web search for more information

Current conversation:
{messages}

What would you like to do next?
"""


def format_messages(messages: List[Dict[str, str]]) -> str:
    """Format messages for the prompt."""
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

def get_next_step(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the next step using the OpenAI API."""
    messages = state["messages"]
    formatted_messages = format_messages(messages)
    
    response = st.session_state.openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": AGENT_PROMPT.format(messages=formatted_messages)
        }],
        temperature=0
    )
    
    next_step = response.choices[0].message.content
    messages.append({"role": "assistant", "content": next_step})
    
    # Check if this is a final answer
    state["done"] = next_step.startswith("Answer:")
    state["next_step"] = next_step
    
    return state

def execute_step(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the current step and update state."""
    next_step = state["next_step"]
    action_match = re.match(r'^Action: (\w+): (.*)$', next_step)
    
    if action_match:
        tool_name, tool_input = action_match.groups()
        result = run_tool(tool_name, tool_input)
        state["messages"].append({"role": "function", "content": result})
    
    return state

def process_query(question: str, max_steps: int = 5) -> List[Dict[str, str]]:
    """Process a question through a series of steps."""
    state = {
        "messages": [{"role": "user", "content": question}],
        "next_step": "",
        "done": False
    }
    
    steps_taken = 0
    while not state["done"] and steps_taken < max_steps:
        try:
            # Get next action from the model
            state = get_next_step(state)
            
            # If we got an answer, we're done
            if state["done"]:
                break
                
            # Execute the action
            state = execute_step(state)
            steps_taken += 1
            
        except Exception as e:
            state["messages"].append({
                "role": "system",
                "content": f"Error during processing: {str(e)}"
            })
            break
    
    return state["messages"]

# Main UI code remains the same as in the original
def main():
    st.title("🤖 ReAct Agent with LangGraph")

    # API Key Input Form
    if not st.session_state.api_keys_submitted:
        st.markdown("### 🔑 API Key Setup")
        with st.form("api_keys_form"):
            openai_key = st.text_input("OpenAI API Key", type="password")
            tavily_key = st.text_input("Tavily API Key", type="password")
            submitted = st.form_submit_button("Submit API Keys")
            
            if submitted:
                if not openai_key or not tavily_key:
                    st.error("Please provide both API keys!")
                else:
                    if initialize_clients(openai_key, tavily_key):
                        st.success("API keys configured successfully!")
                        st.rerun()

    # Only show the main interface if API keys are configured
    if st.session_state.api_keys_submitted:
        st.markdown("""
        This AI agent can help you with:
        - Dog breed information and weight calculations
        - Web searches for additional information
        - Basic mathematical calculations
        - Complex reasoning tasks
        
        Try asking questions like:
        - "What's the average weight of a Border Collie and is it good for apartments?"
        - "If I have 2 Scottish Terriers and 1 Toy Poodle, what's their total weight?"
        - "Calculate the monthly food cost for 3 dogs if each dog eats $2.50 worth of food per day"
        """)

        # Sidebar settings
        with st.sidebar:
            st.title("⚙️ Settings")
            max_steps = st.slider(
                "Maximum reasoning steps",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum number of steps the agent can take"
            )
            
            if st.button("Clear History"):
                st.session_state.conversation_history = []
                st.success("Conversation history cleared!")
            
            if st.button("Reset API Keys"):
                st.session_state.api_keys_submitted = False
                st.session_state.openai_client = None
                st.session_state.tavily_client = None
                st.rerun()

        # Input area
        question = st.text_area(
            "Enter your question:", 
            height=100,
            placeholder="e.g., What's the combined weight of a Border Collie and a Scottish Terrier?"
        )

        if st.button("Ask Question", type="primary"):
            if not question:
                st.error("Please enter a question!")
            else:
                with st.spinner("Thinking..."):
                    messages = process_query(question, max_steps)
                    
                    st.session_state.conversation_history.append({
                        "question": question,
                        "messages": messages,
                        "timestamp": datetime.datetime.now().isoformat()
                    })

        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("### 📜 Conversation History")
            
            for idx, entry in enumerate(reversed(st.session_state.conversation_history)):
                with st.expander(f"Q: {entry['question']}", expanded=(idx == 0)):
                    st.markdown(f"Asked at: {entry['timestamp']}")
                    
                    for message in entry['messages']:
                        role = message['role']
                        content = message['content']
                        
                        if role == "user":
                            st.markdown("### 👤 Human")
                            st.markdown(content)
                        elif role == "assistant":
                            st.markdown("### 🤖 Assistant")
                            st.markdown(content)
                        elif role == "function":
                            st.markdown("### ⚙️ Function Output")
                            try:
                                # Try to parse as JSON for prettier display
                                content_json = json.loads(content)
                                st.json(content_json)
                            except:
                                # Fall back to code display if not JSON
                                st.code(content)

if __name__ == "__main__":
    main()

