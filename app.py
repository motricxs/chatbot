import streamlit as st
import requests
import json
from transformers import AutoTokenizer

# --- Page Configuration ---
# The page title in the browser tab is now "Poulstar Chatbot"
st.set_page_config(page_title="Poulstar Chatbot", page_icon="✨")

# --- Constants and API Setup ---
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
# 1. Added the logo URL
POULSTAR_LOGO_URL = "https://raw.githubusercontent.com/poulstar/.github/main/logo.png"

# 1. Displaying the logo at the top of the page
st.image(POULSTAR_LOGO_URL, width=200)

# 2. Changed the title as requested
st.title("Poulstar chatbot")

# 3. The caption line has been completely removed.
# st.caption("A powerful chatbot using the Mixtral 8x7B model via Hugging Face API.") # This line is now deleted.


# --- Load the tokenizer once using Streamlit's cache ---
# This is lightweight and doesn't require torch or a GPU.
@st.cache_resource
def load_tokenizer():
    """Loads the tokenizer from Hugging Face."""
    return AutoTokenizer.from_pretrained(MODEL_ID)

tokenizer = load_tokenizer()

# --- Helper Function to Call the API ---
def get_mixtral_response(messages):
    """
    Formats the chat history using the official tokenizer and sends it to the API.
    """
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except FileNotFoundError:
        st.error("Hugging Face API token not found. Please add it to your Streamlit secrets.", icon="🔑")
        return

    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # --- THE CORRECT WAY TO FORMAT THE PROMPT ---
    # Use the tokenizer to apply the chat template. This is the official and robust method.
    # We set add_generation_prompt=True to ensure the template ends correctly for the model to generate a response.
    prompt_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    payload = {
        "inputs": prompt_string,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "return_full_text": False
        },
        "stream": True,
        "options": {
            "wait_for_model": True
        }
    }

    try:
        with requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=180) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        json_str = decoded_line[len('data:'):].strip()
                        if json_str:
                            data = json.loads(json_str)
                            yield data.get("token", {}).get("text", "")
    except requests.exceptions.RequestException as e:
        yield f"\n\n**Error:** Could not connect to the API. {e}"
    except json.JSONDecodeError as e:
        yield f"\n\n**Error:** Failed to parse the response from the API. {e}"


# --- Chat Interface Logic ---

if "messages" not in st.session_state:
    st.session_state.messages = [] # Start with an empty history

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get new user input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant's response
    with st.chat_message("assistant"):
        response_stream = get_mixtral_response(st.session_state.messages)
        full_response = st.write_stream(response_stream)

    # Add the full response to the history
    st.session_state.messages.append({"role": "assistant", "content": full_response})