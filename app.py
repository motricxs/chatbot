import streamlit as st
import requests
import json
from transformers import AutoTokenizer

# --- Page Configuration ---
# Using a wide layout for a more modern look
st.set_page_config(page_title="PoulStar AI", page_icon="✨", layout="wide")

# --- Constants and API Setup ---
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
POULSTAR_LOGO_URL = "https://raw.githubusercontent.com/poulstar/.github/main/logo.png"

# --- Header Section with Logo ---
# Placing the logo at the top and center of the page
col1, col2, col3 = st.columns([1, 0.5, 1]) # The middle column is for the logo
with col2:
    st.image(POULSTAR_LOGO_URL, width=200) # Enlarged logo

# --- Main Title and Caption ---
# All UI text is now in simple English.
st.title("PoulStar AI Assistant")
st.caption("✨ Powered by Advanced AI")
st.divider()

# --- Sidebar for Advanced Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Here you can control the AI assistant.")
    
    # Sliders for model parameters with simple English labels and help text
    max_new_tokens = st.slider(
        "Max Answer Length",
        min_value=128, max_value=1024, value=512, step=64,
        help="The maximum number of words in the AI's answer."
    )
    temperature = st.slider(
        "Creativity",
        min_value=0.1, max_value=1.0, value=0.7, step=0.1,
        help="High value = more creative. Low value = more direct."
    )
    st.divider()
    st.info("Made by PoulStar Institute")

# --- Load tokenizer (cached) ---
@st.cache_resource
def load_tokenizer():
    """Loads the tokenizer from Hugging Face."""
    return AutoTokenizer.from_pretrained(MODEL_ID)

tokenizer = load_tokenizer()

# --- Helper Function for API Call ---
def get_ai_response(messages):
    """
    Formats the chat history and sends it to the Hugging Face Inference API.
    """
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except FileNotFoundError:
        st.error("Hugging Face API token not found. Please add it to your Streamlit Secrets.", icon="🔑")
        return

    headers = {"Authorization": f"Bearer {hf_token}"}
    prompt_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    payload = {
        "inputs": prompt_string,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False
        },
        "stream": True,
        "options": {"wait_for_model": True}
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
        yield f"\n\n**Error:** Could not connect to the AI server. Please try again later. ({e})"
    except Exception as e:
        yield f"\n\n**An unexpected error occurred:** {e}"

# --- Chat Interface Logic ---

# Initialize chat history with a welcome message in English
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the PoulStar AI Assistant. How can I help you today?"}]

# Display previous messages
for message in st.session_state.messages:
    # Use the PoulStar logo as the assistant's avatar
    avatar = POULSTAR_LOGO_URL if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Get new user input with an English placeholder
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Get and display assistant's response
    with st.chat_message("assistant", avatar=POULSTAR_LOGO_URL):
        response_stream = get_ai_response(st.session_state.messages)
        full_response = st.write_stream(response_stream)

    st.session_state.messages.append({"role": "assistant", "content": full_response})