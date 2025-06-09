import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(page_title="Mixtral Chatbot", page_icon="🤖")
st.title("🤖 Mixtral Chatbot")
st.caption("A powerful chatbot using the Mixtral 8x7B model via Hugging Face API.")

# --- Constants and API Setup ---
# UPDATED: We are now using the Mixtral model
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

# --- Helper Function to Call the API ---
def get_mixtral_response(messages):
    """
    Sends a request to the Hugging Face Inference API and streams the response.
    """
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except FileNotFoundError:
        st.error("Hugging Face API token not found. Please add it to your Streamlit secrets.", icon="🔑")
        return

    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # --- START OF THE FIX ---
    # We filter out the 'system' message before sending it to the API
    # because some models on the Inference API don't handle it well.
    api_messages = [msg for msg in messages if msg.get("role") != "system"]
    # --- END OF THE FIX ---

    payload = {
        # UPDATED: We send the filtered messages
        "inputs": api_messages,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.1
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
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask Mixtral anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_stream = get_mixtral_response(st.session_state.messages)
        full_response = st.write_stream(response_stream)

    st.session_state.messages.append({"role": "assistant", "content": full_response})