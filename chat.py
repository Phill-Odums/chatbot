import html
import json
import os
import re
import threading
import zlib
from datetime import datetime

import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
from streamlit.errors import StreamlitSecretNotFoundError

MODEL_NAME = "gemini-3.1-flash-lite"
AI_NAME = "PHILLZ AI"
AI_COLOR = "#f4c494"
DEFAULT_ROOM = "lounge"
key = ""
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")

USER_PALETTE = [
    ("#ff6b9d", "🌹"),
    ("#f472b6", "💖"),
    ("#fb7185", "🥂"),
    ("#e879f9", "✨"),
    ("#fda4af", "💌"),
    ("#f59e0b", "🕯️"),
    ("#c084fc", "🦋"),
    ("#fb923c", "🍓"),
    ("#f43f5e", "❤️‍🔥"),
    ("#a78bfa", "🌙"),
]


def get_user_style(name: str) -> tuple[str, str]:
    """Return a consistent (hex_color, emoji_avatar) pair for a given username."""
    normalized = name.strip().lower() or "guest"
    idx = zlib.crc32(normalized.encode("utf-8")) % len(USER_PALETTE)
    return USER_PALETTE[idx]


def normalize_room_name(room: str) -> str:
    """Sanitize a room name so it is clean and URL-safe."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", room.strip().lower()).strip("-")
    return cleaned or DEFAULT_ROOM


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


def _normalize_message(msg: dict) -> dict:
    """Backfill sender/color/timestamp fields for older single-user history entries."""
    role = msg.get("role", "user")
    if role == "assistant":
        sender = msg.get("sender") or AI_NAME
        color = msg.get("color") or AI_COLOR
    else:
        sender = msg.get("sender") or "Member"
        color = msg.get("color") or get_user_style(sender)[0]

    return {
        "role": role,
        "sender": sender,
        "color": color,
        "timestamp": msg.get("timestamp", ""),
        "content": msg.get("content", ""),
    }


class GroupChatStore:
    """Thread-safe shared chat store across all active Streamlit user sessions."""

    def __init__(self, history_file: str):
        self.history_file = history_file
        self.lock = threading.Lock()
        self.rooms: dict[str, list[dict]] = self._load_from_disk()

    def _load_from_disk(self) -> dict[str, list[dict]]:
        if not os.path.exists(self.history_file):
            return {DEFAULT_ROOM: []}

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {DEFAULT_ROOM: []}

        # Migrate flat list from single-user format into the default room
        if isinstance(data, list):
            return {DEFAULT_ROOM: [_normalize_message(m) for m in data if isinstance(m, dict)]}

        if isinstance(data, dict):
            rooms: dict[str, list[dict]] = {}
            for room_name, msgs in data.items():
                if isinstance(msgs, list):
                    rooms[normalize_room_name(str(room_name))] = [
                        _normalize_message(m) for m in msgs if isinstance(m, dict)
                    ]
            return rooms or {DEFAULT_ROOM: []}

        return {DEFAULT_ROOM: []}

    def _save_to_disk(self) -> None:
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.rooms, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get_messages(self, room: str) -> list[dict]:
        with self.lock:
            return [dict(m) for m in self.rooms.get(room, [])]

    def add_message(self, room: str, message: dict) -> None:
        with self.lock:
            self.rooms.setdefault(room, []).append(dict(message))
            self._save_to_disk()

    def clear_room(self, room: str) -> None:
        with self.lock:
            self.rooms[room] = []
            self._save_to_disk()

    def get_participants(self, room: str) -> list[tuple[str, str]]:
        """Return unique (sender, color) pairs for human participants in a room."""
        seen: dict[str, str] = {}
        with self.lock:
            for msg in self.rooms.get(room, []):
                if msg.get("role") == "user":
                    sender = msg.get("sender", "Member")
                    if sender not in seen:
                        seen[sender] = msg.get("color") or get_user_style(sender)[0]
        return list(seen.items())


@st.cache_resource
def get_chat_store() -> GroupChatStore:
    return GroupChatStore(HISTORY_FILE)


def prepare_gemini_turn_payload(messages: list[dict]) -> tuple[list, str]:
    """
    Convert group messages into alternating Gemini turns and extract the latest user prompt.
    Each user message is prefixed with `[Sender]:` so Gemini can tell participants apart.
    """
    merged_turns: list[tuple[str, str]] = []

    for msg in messages:
        role = "user" if msg.get("role") == "user" else "model"
        content = msg.get("content", "")
        if role == "user":
            sender = msg.get("sender", "User")
            formatted = f"[{sender}]: {content}"
        else:
            formatted = content

        if merged_turns and merged_turns[-1][0] == role:
            prev_role, prev_text = merged_turns[-1]
            merged_turns[-1] = (prev_role, f"{prev_text}\n\n{formatted}")
        else:
            merged_turns.append((role, formatted))

    if not merged_turns or merged_turns[-1][0] != "user":
        return [], ""

    latest_prompt = merged_turns[-1][1]
    history_turns = merged_turns[:-1]

    gemini_history = []
    for role, text in history_turns:
        try:
            gemini_history.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=text)],
                )
            )
        except Exception:
            gemini_history.append({"role": role, "parts": [{"text": text}]})

    return gemini_history, latest_prompt


def init_chat_session(client, history: list, room: str):
    """Create a Gemini chat session aware of all group chat participants."""
    config = types.GenerateContentConfig(
        system_instruction=(
            f"You are {AI_NAME}, a helpful, conversational AI participating in a multi-person "
            f"group chat room named '{room}'. User messages are prefixed with `[Username]:` so "
            "you know who is speaking. Address participants by name when relevant, remember "
            "each person's preferences and context separately, and keep replies concise and natural."
        )
    )
    return client.chats.create(
        model=MODEL_NAME,
        config=config,
        history=history,
    )


def should_ai_respond(prompt: str, ai_mode: str) -> bool:
    if ai_mode == "Every message":
        return True
    return bool(re.search(r"(?i)@ai\b", prompt))


def render_message(msg: dict, current_user: str) -> None:
    """Render a single message bubble visually differentiated by sender."""
    role = msg.get("role", "user")
    sender = msg.get("sender", "Member")
    color = msg.get("color") or get_user_style(sender)[0]
    timestamp = msg.get("timestamp", "")
    content = msg.get("content", "")

    escaped_sender = html.escape(sender)
    escaped_time = html.escape(timestamp)
    time_html = f'<span class="msg-time">{escaped_time}</span>' if timestamp else ""

    if role == "assistant":
        with st.chat_message("assistant", avatar="💘"):
            st.markdown(
                f'<div class="msg-header msg-ai">'
                f'<span class="msg-sender" style="color: {AI_COLOR};">{escaped_sender} · Wingman</span>'
                f"{time_html}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(content)
    elif sender == current_user:
        _, avatar = get_user_style(sender)
        with st.chat_message("user", avatar=avatar):
            st.markdown(
                f'<div class="msg-header msg-self">'
                f'<span class="msg-sender">You ({escaped_sender})</span>'
                f"{time_html}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(content)
    else:
        _, avatar = get_user_style(sender)
        with st.chat_message("user", avatar=avatar):
            st.markdown(
                f'<div class="msg-header msg-peer" data-peer-color="{html.escape(color)}">'
                f'<span class="msg-dot" style="background-color: {html.escape(color)};"></span>'
                f'<span class="msg-sender" style="color: {html.escape(color)};">{escaped_sender}</span>'
                f"{time_html}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(content)


st.set_page_config(
    page_title="PHILLZ Lounge · Romantic Group Chat",
    page_icon="🌹",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700;1,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Romantic Midnight Velvet Background */
    .stApp,
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 85% 10%, rgba(157, 23, 77, 0.34) 0%, transparent 45%),
                    radial-gradient(circle at 12% 80%, rgba(190, 24, 93, 0.22) 0%, transparent 48%),
                    linear-gradient(160deg, #1d0c18 0%, #140812 52%, #0e060e 100%) !important;
        color: #fdf2f8 !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar Velvet Plum */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #210e1b 0%, #150812 100%) !important;
        border-right: 1px solid rgba(255, 143, 171, 0.16) !important;
    }

    [data-testid="stSidebar"] * {
        color: #fce7f3;
    }

    /* Global font & typography */
    html, body, [class*="css"] {
        font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    h1, h2, h3, .romantic-serif {
        font-family: "Cormorant Garamond", Georgia, serif !important;
        letter-spacing: 0.02em;
    }

    /* Glassmorphic Join Form */
    [data-testid="stForm"] {
        background: linear-gradient(145deg, rgba(56, 21, 43, 0.72), rgba(32, 12, 26, 0.85)) !important;
        border: 1px solid rgba(255, 143, 171, 0.28) !important;
        border-radius: 24px !important;
        padding: 1.5rem !important;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(14px) !important;
    }

    /* Romantic Buttons */
    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #ff4d79 0%, #c9184a 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 179, 198, 0.35) !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 16px rgba(201, 24, 74, 0.32) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 22px rgba(255, 77, 121, 0.45) !important;
        border-color: rgba(255, 204, 213, 0.6) !important;
    }

    /* Text Inputs */
    [data-testid="stTextInput"] input {
        background-color: rgba(29, 11, 24, 0.85) !important;
        color: #fff1f2 !important;
        border: 1px solid rgba(255, 143, 171, 0.28) !important;
        border-radius: 14px !important;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: #ff4d79 !important;
        box-shadow: 0 0 0 2px rgba(255, 77, 121, 0.25) !important;
    }

    /* Base Chat Message Container */
    [data-testid="stChatMessage"],
    .stChatMessage {
        display: inline-flex !important;
        width: fit-content !important;
        min-width: 0 !important;
        max-width: 78% !important;
        box-sizing: border-box !important;
        gap: 10px !important;
        align-items: flex-start !important;
        padding: 10px 16px !important;
        margin-bottom: 10px !important;
        border: none !important;
        backdrop-filter: blur(10px) !important;
    }

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
        line-height: 1.5 !important;
        font-size: 0.95rem !important;
    }

    /* Sender label & timestamp inside each bubble */
    .msg-header {
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
        margin-bottom: 4px !important;
        font-size: 0.75rem !important;
        line-height: 1.1 !important;
    }

    .msg-sender {
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
    }

    .msg-dot {
        display: inline-block !important;
        width: 8px !important;
        height: 8px !important;
        border-radius: 50% !important;
        flex-shrink: 0 !important;
        box-shadow: 0 0 6px currentColor !important;
    }

    .msg-time {
        opacity: 0.72 !important;
        font-size: 0.7rem !important;
        font-weight: 400 !important;
    }

    /* ========================================================
       YOUR OWN MESSAGES (Right-aligned Rose Crimson Gradient)
       ======================================================== */
    [data-testid="stChatMessage"]:has(.msg-self) {
        flex-direction: row-reverse !important;
        justify-content: flex-start !important;
        align-self: flex-end !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        background: linear-gradient(135deg, #ff4d79 0%, #c9184a 100%) !important;
        color: #ffffff !important;
        border-radius: 20px 20px 4px 20px !important;
        border: 1px solid rgba(255, 204, 213, 0.28) !important;
        box-shadow: 0 6px 20px rgba(201, 24, 74, 0.3) !important;
    }

    [data-testid="stChatMessage"]:has(.msg-self) p,
    [data-testid="stChatMessage"]:has(.msg-self) .msg-header {
        color: #ffffff !important;
        text-align: left !important;
    }

    [data-testid="stChatMessage"]:has(.msg-self) .msg-sender {
        color: #ffe4e6 !important;
    }

    [data-testid="stChatMessage"]:has(.msg-self) code {
        background-color: rgba(255, 255, 255, 0.22) !important;
        color: #ffffff !important;
        border-radius: 6px !important;
    }

    /* ========================================================
       OTHER PARTICIPANTS' MESSAGES (Left-aligned Rose-Plum Glass)
       ======================================================== */
    [data-testid="stChatMessage"]:has(.msg-peer) {
        flex-direction: row !important;
        justify-content: flex-start !important;
        align-self: flex-start !important;
        margin-right: auto !important;
        margin-left: 0 !important;
        background: rgba(54, 22, 42, 0.86) !important;
        color: #fdf2f8 !important;
        border-radius: 20px 20px 20px 4px !important;
        border: 1px solid rgba(255, 179, 198, 0.22) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
    }

    [data-testid="stChatMessage"]:has(.msg-peer) p {
        color: #fdf2f8 !important;
    }

    /* ========================================================
       AI WINGMAN MESSAGE (Left-aligned Champagne & Velvet Card)
       ======================================================== */
    [data-testid="stChatMessage"]:has(.msg-ai),
    [data-testid="stChatMessage"]:has([data-testid*="Assistant"]),
    [data-testid="stChatMessage"]:has([data-testid*="assistant"]) {
        flex-direction: row !important;
        justify-content: flex-start !important;
        align-self: flex-start !important;
        margin-right: auto !important;
        margin-left: 0 !important;
        max-width: 82% !important;
        background: linear-gradient(145deg, rgba(46, 20, 38, 0.94), rgba(31, 13, 27, 0.94)) !important;
        color: #fff1f2 !important;
        border-radius: 20px 20px 20px 4px !important;
        border: 1px solid rgba(244, 196, 148, 0.34) !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.28) !important;
    }

    [data-testid="stChatMessage"]:has(.msg-ai) p,
    [data-testid="stChatMessage"]:has([data-testid*="Assistant"]) p,
    [data-testid="stChatMessage"]:has([data-testid*="assistant"]) p {
        color: #fff1f2 !important;
    }

    /* Circular profile avatars */
    [data-testid^="stChatMessageAvatar"],
    .stChatMessageAvatar {
        border-radius: 50% !important;
        overflow: hidden !important;
        width: 30px !important;
        height: 30px !important;
        min-width: 30px !important;
        min-height: 30px !important;
        margin: 2px 0 0 0 !important;
        flex-shrink: 0 !important;
        background: rgba(255, 77, 121, 0.16) !important;
        border: 1px solid rgba(255, 179, 198, 0.3) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.28) !important;
    }

    /* Rounded images in messages */
    [data-testid="stChatMessage"] img {
        border-radius: 14px !important;
        max-width: 100% !important;
    }

    /* ========================================================
       ROMANTIC CHAT INPUT BAR
       ======================================================== */
    [data-testid="stChatInput"] {
        padding-bottom: 1rem !important;
    }

    [data-testid="stChatInput"] > div {
        background: rgba(44, 18, 36, 0.92) !important;
        border: 1px solid rgba(255, 143, 171, 0.28) !important;
        border-radius: 26px !important;
        padding: 5px 12px !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35) !important;
        backdrop-filter: blur(12px) !important;
        transition: all 0.2s ease-in-out !important;
    }

    [data-testid="stChatInput"] > div:focus-within {
        border-color: #ff4d79 !important;
        box-shadow: 0 0 0 2px rgba(255, 77, 121, 0.3), 0 8px 26px rgba(255, 77, 121, 0.22) !important;
    }

    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #fff1f2 !important;
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 4px 8px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #d8a7b8 !important;
        font-size: 0.95rem !important;
    }

    [data-testid="stChatInput"] button {
        background: transparent !important;
        border: none !important;
        transition: transform 0.15s ease !important;
    }

    [data-testid="stChatInput"] button:hover {
        transform: scale(1.16) !important;
    }

    [data-testid="stChatInput"] button svg {
        fill: #ff4d79 !important;
        color: #ff4d79 !important;
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(255, 77, 121, 0.55); }
        50% { box-shadow: 0 0 0 6px rgba(255, 77, 121, 0); }
    }

    .romantic-pulse {
        animation: pulseGlow 2.2s infinite;
    }

    /* Mobile screens */
    @media (max-width: 640px) {
        [data-testid="stChatMessage"]:has(.msg-self),
        [data-testid="stChatMessage"]:has(.msg-peer) {
            max-width: 86% !important;
        }
        [data-testid="stChatMessage"]:has(.msg-ai) {
            max-width: 88% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

store = get_chat_store()
api_key = get_api_key()

if "username" not in st.session_state:
    st.session_state.username = ""

if "room" not in st.session_state:
    query_room = st.query_params.get("room", DEFAULT_ROOM)
    st.session_state.room = normalize_room_name(str(query_room))

if "ai_mode" not in st.session_state:
    st.session_state.ai_mode = "Every message"

if "client" not in st.session_state:
    st.session_state.client = None

room = st.session_state.room
participants = store.get_participants(room)

# Header
st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 1.35rem; padding: 0.9rem 1.1rem; border-radius: 20px; background: linear-gradient(135deg, rgba(64, 22, 47, 0.72), rgba(33, 12, 27, 0.85)); border: 1px solid rgba(255, 143, 171, 0.22); box-shadow: 0 10px 28px rgba(0, 0, 0, 0.3);">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div style="position: relative; width: 46px; height: 46px;">
                <div style="width: 46px; height: 46px; border-radius: 50%; background: linear-gradient(135deg, #ff4d79 0%, #c9184a 100%); display: flex; align-items: center; justify-content: center; font-size: 20px; color: white; box-shadow: 0 4px 14px rgba(255, 77, 121, 0.4);">🌹</div>
                <span class="romantic-pulse" style="position: absolute; bottom: 1px; right: 1px; width: 11px; height: 11px; background-color: #ff4d79; border: 2px solid #1d0c18; border-radius: 50%;"></span>
            </div>
            <div>
                <h2 class="romantic-serif" style="margin: 0; font-size: 1.55rem; font-weight: 700; line-height: 1.15; color: #fff1f2;">PHILLZ Lounge</h2>
                <div style="font-size: 0.8rem; color: #ffb3c6; font-weight: 500;">♥ Room #{html.escape(room)} · Intimate Live Chat</div>
            </div>
        </div>
        <div style="font-size: 0.78rem; color: #f4c494; background: rgba(244, 196, 148, 0.12); border: 1px solid rgba(244, 196, 148, 0.28); padding: 5px 12px; border-radius: 999px; font-weight: 600;">
            💘 Cupid Online
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Join screen before entering the group chat
if not st.session_state.username:
    st.markdown("### Step into the Lounge 🌹")
    st.caption(
        "Choose your display name so everyone in the room—and your AI wingman—knows who's speaking."
    )
    with st.form("join_chat_form", clear_on_submit=False):
        name_input = st.text_input("Your display name", placeholder="e.g. Sophia")
        room_input = st.text_input(
            "Lounge room",
            value=st.session_state.room,
            help="Share the same room name (or ?room=name URL) with someone special or your group.",
        )
        submitted = st.form_submit_button("Enter the Lounge ♥", use_container_width=True)
        if submitted:
            cleaned_name = name_input.strip()
            if not cleaned_name:
                st.warning("Please enter a display name to enter.")
            elif cleaned_name.lower() == AI_NAME.lower():
                st.warning(f'Please choose a name other than "{AI_NAME}".')
            else:
                target_room = normalize_room_name(room_input)
                st.session_state.username = cleaned_name
                st.session_state.room = target_room
                st.query_params["room"] = target_room
                st.rerun()
    st.stop()

username = st.session_state.username
user_color, user_avatar = get_user_style(username)
room_messages = store.get_messages(room)

with st.sidebar:
    st.subheader("Your Vibe")
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;'
        f'border-radius:14px;background:rgba(255,77,121,0.12);border:1px solid rgba(255,143,171,0.24);margin-bottom:10px;">'
        f'<span style="font-size:1.1rem;">{user_avatar}</span>'
        f'<span style="font-weight:700;color:{user_color};">{html.escape(username)}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
    if st.button("Change name / lounge", use_container_width=True):
        st.session_state.username = ""
        st.rerun()

    st.divider()
    st.subheader("Private Lounge")
    new_room = st.text_input("Room ID", value=room, help="Share ?room=<id> in the URL.")
    if st.button("Switch lounge", use_container_width=True):
        target_room = normalize_room_name(new_room)
        st.session_state.room = target_room
        st.query_params["room"] = target_room
        st.rerun()

    st.caption(f"💌 **Messages in #{room}:** {len(room_messages)}")

    st.divider()
    st.subheader("In This Lounge")
    st.markdown(f"- 💘 **{AI_NAME} · Wingman**")
    listed_names = set()
    for participant_name, participant_color in participants:
        listed_names.add(participant_name)
        _, participant_avatar = get_user_style(participant_name)
        you_tag = " *(you)*" if participant_name == username else ""
        st.markdown(
            f'- {participant_avatar} <span style="color:{participant_color};font-weight:600;">'
            f"{html.escape(participant_name)}</span>{you_tag}",
            unsafe_allow_html=True,
        )
    if username not in listed_names:
        st.markdown(
            f'- {user_avatar} <span style="color:{user_color};font-weight:600;">'
            f"{html.escape(username)}</span> *(you)*",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Wingman Mode")
    st.session_state.ai_mode = st.radio(
        "When should Cupid reply?",
        options=["Every message", "Only when @ai is mentioned"],
        index=0 if st.session_state.ai_mode == "Every message" else 1,
    )
    if not hasattr(st, "fragment"):
        if st.button("Refresh messages", use_container_width=True):
            st.rerun()
    if st.button("Clear lounge history", use_container_width=True):
        store.clear_room(room)
        st.rerun()


def _render_chat_feed() -> None:
    current_messages = store.get_messages(st.session_state.room)
    for message in current_messages:
        render_message(message, current_user=st.session_state.username)
    if not current_messages:
        st.markdown("### Break the ice 🥂")
        st.caption(
            f"Invite others to `#{st.session_state.room}` (`?room={st.session_state.room}`) "
            "and let the conversation spark."
        )


if hasattr(st, "fragment"):
    @st.fragment(run_every="2s")
    def render_live_chat_feed() -> None:
        _render_chat_feed()

    render_live_chat_feed()
else:
    _render_chat_feed()

if not api_key and st.session_state.ai_mode == "Every message":
    st.info(
        "Gemini is not configured yet. Set `GEMINI_API_KEY` in your environment "
        "or add `GEMINI_API_KEY = \"...\"` to `.streamlit/secrets.toml` "
        "(you can still switch Wingman Mode to `Only when @ai is mentioned` to chat with others)."
    )

placeholder_text = (
    f"Say something sweet in #{room} as {username}..."
    if st.session_state.ai_mode == "Every message"
    else f"Message #{room} as {username} (mention @ai for Wingman)..."
)
prompt = st.chat_input(placeholder_text)

if prompt:
    now_str = datetime.now().strftime("%H:%M")
    user_msg = {
        "role": "user",
        "sender": username,
        "color": user_color,
        "timestamp": now_str,
        "content": prompt,
    }
    store.add_message(room, user_msg)

    if not should_ai_respond(prompt, st.session_state.ai_mode):
        st.rerun()

    # Show the user's message immediately while streaming the AI reply
    render_message(user_msg, current_user=username)

    if not api_key:
        st.info(
            "Gemini is not configured yet. Add `GEMINI_API_KEY` in Secrets or environment variables."
        )
        st.stop()

    with st.chat_message("assistant", avatar="💘"):
        st.markdown(
            f'<div class="msg-header msg-ai">'
            f'<span class="msg-sender" style="color: {AI_COLOR};">{AI_NAME} · Wingman</span>'
            f'<span class="msg-time">{datetime.now().strftime("%H:%M")}</span></div>',
            unsafe_allow_html=True,
        )
        response_placeholder = st.empty()
        full_response = ""

        try:
            if st.session_state.client is None:
                st.session_state.client = genai.Client(api_key=api_key)

            latest_messages = store.get_messages(room)
            history, latest_user_turn = prepare_gemini_turn_payload(latest_messages)
            chat = init_chat_session(st.session_state.client, history, room)

            for chunk in chat.send_message_stream(latest_user_turn):
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
            st.session_state.client = genai.Client(api_key=api_key)
            st.warning("Connection reset. Please send your message again.")
        else:
            store.add_message(
                room,
                {
                    "role": "assistant",
                    "sender": AI_NAME,
                    "color": AI_COLOR,
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "content": full_response,
                },
            )
            st.rerun()
