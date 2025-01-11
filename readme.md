# ReAct Agent with LangGraph

A Streamlit-based web application that implements a ReAct (Reasoning and Action) agent using LangGraph. The agent can help with dog breed information, perform calculations, and search the web for additional information.

## Project Architecture

![alt text](image.png)

## Features

- 🐕 Dog breed weight information and comparisons
- 🔢 Basic mathematical calculations
- 🌐 Web search integration using Tavily
- 💭 Complex reasoning tasks using GPT-4
- 📝 Conversation history tracking
- ⚙️ Configurable maximum reasoning steps

## Requirements

- Python 3.8+
- OpenAI API key
- Tavily API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ravix007/reAct-langgraph.git
cd react-agent-langgraph
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Configuration

Before running the application, you need to obtain:

- OpenAI API key from [OpenAI Platform](https://platform.openai.com)
- Tavily API key from [Tavily](https://tavily.com)

## Running the Application

1. Start the Streamlit app:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Enter your API keys in the setup form when prompted

## Usage Examples

1. Dog Breed Information:

```
What's the average weight of a Border Collie and is it good for apartments?
```

2. Multiple Dog Weight Calculations:

```
If I have 2 Scottish Terriers and 1 Toy Poodle, what's their total weight?
```

3. Cost Calculations:

```
Calculate the monthly food cost for 3 dogs if each dog eats $2.50 worth of food per day
```

## Features in Detail

### Dog Breed Database

Currently includes information for:

- Border Collie
- Scottish Terrier
- Toy Poodle
- Bulldog

### Available Tools

1. **calculate**

   - Performs basic mathematical calculations
   - Supports standard arithmetic operations

2. **average_dog_weight**

   - Provides weight information for specific dog breeds
   - Returns weights in both pounds and kilograms

3. **web_search**
   - Searches the web using Tavily API
   - Returns top 3 relevant results

### Settings

The application allows users to:

- Adjust maximum reasoning steps (1-10)
- Clear conversation history
- Reset API keys

## Project Structure

```
react-agent-langgraph/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Technical Details

- Uses GPT-4 for reasoning and decision-making
- Implements ReAct pattern (Reasoning + Acting)
- Built with Streamlit for the user interface
- Integrates LangGraph for agent workflow management
- Uses Tavily for web searches
- Includes type hints and proper error handling

## Limitations

- Limited to predefined dog breeds in the codebase
- Requires active internet connection for web searches
- API keys needed for full functionality

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Uses [LangGraph](https://github.com/langchain-ai/langgraph) for agent workflows
- Powered by [OpenAI](https://openai.com/) GPT-4
- Search powered by [Tavily](https://tavily.com/)

## Screenshots of working application

![alt text](image-1.png)
![alt text](<Screenshot 2025-01-11 at 7.13.16 PM.png>)
![alt text](image-2.png)
