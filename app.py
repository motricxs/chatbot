import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(page_title="Llama 3 Chatbot", page_icon="🦙")
st.title("🦙 Llama 3 Chatbot")
st.caption("A powerful chatbot using the Llama 3 8B model via Hugging Face API.")

# --- Constants and API Setup ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# --- Helper Function to Call the API ---
def get_llama_response(messages):
    """
    Sends a request to the Hugging Face Inference API and streams the response.
    """
    # Fetch the API token from Streamlit secrets
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except FileNotFoundError:
        st.error("Hugging Face API token not found. Please add it to your Streamlit secrets.", icon="🔑")
        return

    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Prepare the payload for the API
    payload = {
        "inputs": messages,
        "parameters": {
            "max_new_tokens": 512,  # Limit the length of the response
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        },
        "stream": True,  # Enable streaming
        "options": {
            "wait_for_model": True  # Wait if the model is loading
        }
    }

    # Make the streaming POST request
    try:
        with requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=180) as response:
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            # Each line in a streaming response is a JSON object.
            # We need to parse it to get the token.
            for line in response.iter_lines():
                if line:
                    # The API sends lines like b'data:{"token": {"text": "..."}}'
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        json_str = decoded_line[len('data:'):].strip()
                        if json_str:
                            data = json.loads(json_str)
                            # Yield the actual text token
                            yield data.get("token", {}).get("text", "")

    except requests.exceptions.RequestException as e:
        yield f"\n\n**Error:** Could not connect to the API. {e}"
    except json.JSONDecodeError as e:
        yield f"\n\n**Error:** Failed to parse the response from the API. {e}"


# --- Chat Interface Logic ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

# Display chat messages from history
for message in st.session_state.messages:
    # Do not display the system message
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask Llama 3 anything..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant's response using the streaming function
    with st.chat_message("assistant"):
        # The history sent to the API should not contain previous assistant responses
        # to avoid confusion, or should be structured correctly. For simplicity here,
        # we can send a limited history. A more advanced app would manage this better.
        
        # We pass the conversation history to the model
        response_stream = get_llama_response(st.session_state.messages)
        
        # Use write_stream to display the streaming response
        full_response = st.write_stream(response_stream)

    # Add the full response to the history for future context
    st.session_state.messages.append({"role": "assistant", "content": full_response})