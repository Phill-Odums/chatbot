import json
import os

import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
from streamlit.errors import StreamlitSecretNotFoundError

MODEL_NAME = "gemini-3.1-flash-lite"
key = ""
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")


def get_api_key():
    if key:
        return key

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key

    try:
        secrets = st.secrets
        gemini_secrets = secrets.get("gemini", {})
        return (
            secrets.get("GEMINI_API_KEY")
            or gemini_secrets.get("api_key")
            or gemini_secrets.get("key")
        )
    except StreamlitSecretNotFoundError:
        return None


def load_history():
    """Load conversation history from disk so messages persist across sessions."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(messages):
    """Save conversation history to disk."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def build_gemini_history(messages):
    """Convert stored messages to the format expected by the Gemini API."""
    gemini_history = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        try:
            gemini_history.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])],
                )
            )
        except Exception:
            gemini_history.append(
                {"role": role, "parts": [{"text": msg["content"]}]}
            )
    return gemini_history


def init_chat_session(client, messages):
    """Create a Gemini chat session with full past memory and memory instruction."""
    config = types.GenerateContentConfig(
        system_instruction=(
            "You are a helpful, conversational AI companion. "
            "Maintain strong conversation continuity, remembering user preferences, "
            "facts, and all details shared earlier in the conversation."
        )
    )
    return client.chats.create(
        model=MODEL_NAME,
        config=config,
        history=build_gemini_history(messages),
    )


st.set_page_config(
    page_title="PHILLZ Chat",
    page_icon="💬",
    layout="centered",
)

st.markdown(
    """
    <style>
    /* Global font & typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Base Chat Message Container */
    [data-testid="stChatMessage"],
    .stChatMessage {
        display: inline-flex !important;
        width: fit-content !important;
        min-width: 0 !important;
        max-width: 75% !important;
        box-sizing: border-box !important;
        gap: 8px !important;
        align-items: center !important;
        padding: 8px 14px !important;
        margin-bottom: 8px !important;
        border: none !important;
    }

    /* Content wrapper inside message - hugs text snugly */
    [data-testid="stChatMessageContent"] {
        flex: 0 1 auto !important;
        width: fit-content !important;
        min-width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    [data-testid="stChatMessageContent"] .stMarkdown,
    [data-testid="stChatMessageContent"] div {
        width: fit-content !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stChatMessageContent"] p {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.45 !important;
        font-size: 0.95rem !important;
    }

    /* ========================================================
       USER MESSAGE (Outgoing)
       Clean, solid classic Royal Blue - NO AI purple/neon gradient
       ======================================================== */
    [data-testid="stChatMessage"]:has([data-testid*="User"]),
    [data-testid="stChatMessage"]:has([data-testid*="user"]) {
        flex-direction: row-reverse !important;
        justify-content: flex-start !important;
        align-self: flex-end !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        background-color: #0084ff !important;
        color: #ffffff !important;
        border-radius: 18px 18px 4px 18px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
    }

    [data-testid="stChatMessage"]:has([data-testid*="User"]) p,
    [data-testid="stChatMessage"]:has([data-testid*="user"]) p {
        color: #ffffff !important;
        text-align: left !important;
    }

    [data-testid="stChatMessage"]:has([data-testid*="User"]) code {
        background-color: rgba(255, 255, 255, 0.22) !important;
        color: #ffffff !important;
        border-radius: 4px !important;
    }

    /* ========================================================
       ASSISTANT MESSAGE (Incoming)
       Sleek, warm dark charcoal card - clean, comfortable contrast
       ======================================================== */
    [data-testid="stChatMessage"]:has([data-testid*="Assistant"]),
    [data-testid="stChatMessage"]:has([data-testid*="assistant"]) {
        flex-direction: row !important;
        justify-content: flex-start !important;
        align-self: flex-start !important;
        margin-right: auto !important;
        margin-left: 0 !important;
        max-width: 80% !important;
        background-color: #2b2d31 !important;
        color: #f2f3f5 !important;
        border-radius: 18px 18px 18px 4px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15) !important;
    }

    [data-testid="stChatMessage"]:has([data-testid*="Assistant"]) p,
    [data-testid="stChatMessage"]:has([data-testid*="assistant"]) p {
        color: #f2f3f5 !important;
    }

    /* Circular profile avatars */
    [data-testid^="stChatMessageAvatar"],
    .stChatMessageAvatar {
        border-radius: 50% !important;
        overflow: hidden !important;
        width: 28px !important;
        height: 28px !important;
        min-width: 28px !important;
        min-height: 28px !important;
        margin: 0 !important;
        flex-shrink: 0 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
    }

    /* Rounded images in messages */
    [data-testid="stChatMessage"] img {
        border-radius: 12px !important;
        max-width: 100% !important;
    }

    /* ========================================================
       FACEBOOK MESSENGER CHAT BOX (INPUT BAR)
       ======================================================== */
    [data-testid="stChatInput"] {
        padding-bottom: 1rem !important;
    }

    /* Capsule pill in dark neutral */
    [data-testid="stChatInput"] > div {
        background-color: #2b2d31 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 22px !important;
        padding: 4px 10px !important;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }

    /* Focus border */
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #0084ff !important;
        box-shadow: 0 0 0 2px rgba(0, 132, 255, 0.25), 0 4px 16px rgba(0, 0, 0, 0.3) !important;
    }

    /* Input text & placeholder */
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #f2f3f5 !important;
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 4px 8px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #949ba4 !important;
        font-size: 0.95rem !important;
    }

    /* Blue Send Button */
    [data-testid="stChatInput"] button {
        background: transparent !important;
        border: none !important;
        transition: transform 0.15s ease !important;
    }

    [data-testid="stChatInput"] button:hover {
        transform: scale(1.15) !important;
    }

    [data-testid="stChatInput"] button svg {
        fill: #0084ff !important;
        color: #0084ff !important;
    }

    /* Mobile screens */
    @media (max-width: 640px) {
        [data-testid="stChatMessage"]:has([data-testid*="User"]) {
            max-width: 86% !important;
        }
        [data-testid="stChatMessage"]:has([data-testid*="Assistant"]) {
            max-width: 88% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Clean, authentic Messenger Header (No AI sparkles or gradients)
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1.25rem; padding-bottom: 0.75rem; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
        <div style="position: relative; width: 42px; height: 42px;">
            <div style="width: 42px; height: 42px; border-radius: 50%; background-color: #0084ff; display: flex; align-items: center; justify-content: center; font-size: 17px; font-weight: 700; color: white;">P</div>
            <span style="position: absolute; bottom: 0; right: 0; width: 11px; height: 11px; background-color: #31a24c; border: 2px solid #18191a; border-radius: 50%;"></span>
        </div>
        <div>
            <h2 style="margin: 0; font-size: 1.2rem; font-weight: 700; line-height: 1.2; color: #f2f3f5;">PHILLZ Chat</h2>
            <div style="font-size: 0.8rem; color: #31a24c; font-weight: 500;">● Active now</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key = get_api_key()

# Load persisted conversation history on start
if "messages" not in st.session_state:
    st.session_state.messages = load_history()

if "client" not in st.session_state:
    st.session_state.client = None
    st.session_state.chat = None
elif "chat" not in st.session_state:
    st.session_state.chat = None

with st.sidebar:
    st.subheader("Conversation")
    st.caption(f"💾 **Memory:** {len(st.session_state.messages)} messages remembered")
    if st.button("Start a new chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat = None
        save_history([])  # Clear history from disk
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.messages:
    st.markdown("### Start a conversation")
    st.caption("Ask a question, explore an idea, or share something with your AI.")

if not api_key:
    st.info(
        "Gemini is not configured yet. Set `GEMINI_API_KEY` in your environment "
        "or add `GEMINI_API_KEY = \"...\"` to `.streamlit/secrets.toml`, then "
        "reload the app."
    )
    st.chat_input("Add a GEMINI_API_KEY to begin", disabled=True)
    st.stop()

# Initialize Gemini Chat with full conversation memory
if st.session_state.chat is None:
    if st.session_state.client is None:
        st.session_state.client = genai.Client(api_key=api_key)
    st.session_state.chat = init_chat_session(
        st.session_state.client,
        st.session_state.messages,
    )

# Messenger-styled Chat Input
prompt = st.chat_input("Aa")

if prompt:
    # 1. Store & save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_history(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in st.session_state.chat.send_message_stream(prompt):
                chunk_text = getattr(chunk, "text", None)
                if chunk_text:
                    full_response += chunk_text
                    response_placeholder.markdown(full_response + "▌")

            if full_response:
                response_placeholder.markdown(full_response)
            else:
                full_response = "Gemini returned no text for that prompt."
                response_placeholder.markdown(full_response)
        except APIError as error:
            response_placeholder.empty()
            st.error(f"Gemini request failed: {error}")
        except RuntimeError:
            response_placeholder.empty()
            # Recreate session without losing previous messages
            st.session_state.client = genai.Client(api_key=api_key)
            st.session_state.chat = init_chat_session(
                st.session_state.client,
                st.session_state.messages,
            )
            st.warning("Session reconnected with full memory intact. Please send your message again.")
        else:
            # 2. Store & save assistant response
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
            save_history(st.session_state.messages)
