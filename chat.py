import html
import inspect
import io
import json
import mimetypes
import os
import re
import threading
import time
import uuid
import wave
import zlib
from datetime import datetime

import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
from streamlit.errors import StreamlitSecretNotFoundError

MODEL_NAME = "gemini-3.1-flash-lite"
VIDEO_MODELS = [
    "gemini-omni-1.2-flash",
    "veo-3.1-fast-generate-001",
    "veo-3.1-generate-001",
]
VIDEO_ASPECT_RATIOS = ["16:9", "9:16"]
AUDIO_MODELS = [
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
]
AUDIO_VOICES = [
    "Kore",
    "Zephyr",
    "Puck",
    "Aoede",
    "Leda",
    "Orus",
    "Fenrir",
    "Charon",
]
AI_NAME = "PHILLZ AI"
AI_COLOR = "#f4c494"
DEFAULT_ROOM = "lounge"
key = ""
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "chat_uploads")
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB per file
ALLOWED_UPLOAD_TYPES = [
    "png",
    "jpg",
    "jpeg",
    "webp",
    "gif",
    "mp4",
    "webm",
    "mp3",
    "wav",
    "ogg",
    "m4a",
    "pdf",
    "txt",
    "md",
    "csv",
    "json",
]
GEMINI_INLINE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "audio/mpeg",
    "audio/wav",
    "audio/ogg",
    "audio/mp4",
    "application/pdf",
}
TEXT_FILE_EXTENSIONS = (".txt", ".md", ".csv", ".json", ".py", ".log")

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
    """Backfill sender/color/timestamp/attachments fields for older history entries."""
    role = msg.get("role", "user")
    if role == "assistant":
        sender = msg.get("sender") or AI_NAME
        color = msg.get("color") or AI_COLOR
    else:
        sender = msg.get("sender") or "Member"
        color = msg.get("color") or get_user_style(sender)[0]

    attachments = msg.get("attachments")
    if not isinstance(attachments, list):
        attachments = []

    return {
        "id": msg.get("id") or uuid.uuid4().hex[:12],
        "role": role,
        "sender": sender,
        "color": color,
        "timestamp": msg.get("timestamp", ""),
        "content": msg.get("content", ""),
        "attachments": attachments,
    }


def _format_file_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


class GroupChatStore:
    """Thread-safe shared chat store across all active Streamlit user sessions."""

    def __init__(self, history_file: str, upload_dir: str):
        self.history_file = history_file
        self.upload_dir = upload_dir
        self.lock = threading.Lock()
        self.file_blobs: dict[str, bytes] = {}
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

    def save_bytes_attachment(
        self, raw_bytes: bytes, file_name: str, mime_type: str | None = None
    ) -> dict:
        """Persist raw bytes in memory and best-effort on disk, returning metadata."""
        clean_name = os.path.basename(file_name or "attachment")
        resolved_mime = (
            mime_type
            or mimetypes.guess_type(clean_name)[0]
            or "application/octet-stream"
        )
        _, ext = os.path.splitext(clean_name)
        safe_ext = re.sub(r"[^a-zA-Z0-9.]", "", ext)[:12]
        file_id = uuid.uuid4().hex[:16]
        stored_name = f"{file_id}{safe_ext}"

        with self.lock:
            self.file_blobs[file_id] = raw_bytes
            try:
                os.makedirs(self.upload_dir, exist_ok=True)
                with open(os.path.join(self.upload_dir, stored_name), "wb") as f:
                    f.write(raw_bytes)
            except Exception:
                pass

        return {
            "id": file_id,
            "name": clean_name,
            "stored_name": stored_name,
            "mime_type": resolved_mime,
            "size": len(raw_bytes),
        }

    def save_attachment(self, uploaded_file) -> dict:
        """Persist an uploaded Streamlit file, returning attachment metadata."""
        return self.save_bytes_attachment(
            raw_bytes=uploaded_file.getvalue(),
            file_name=getattr(uploaded_file, "name", "attachment") or "attachment",
            mime_type=getattr(uploaded_file, "type", None),
        )

    def get_file_bytes(self, file_id: str, stored_name: str | None = None) -> bytes | None:
        with self.lock:
            if file_id in self.file_blobs:
                return self.file_blobs[file_id]
            if stored_name:
                disk_path = os.path.join(self.upload_dir, os.path.basename(stored_name))
                if os.path.exists(disk_path):
                    try:
                        with open(disk_path, "rb") as f:
                            raw_bytes = f.read()
                        self.file_blobs[file_id] = raw_bytes
                        return raw_bytes
                    except Exception:
                        return None
        return None

    def get_messages(self, room: str) -> list[dict]:
        with self.lock:
            return [dict(m) for m in self.rooms.get(room, [])]

    def add_message(self, room: str, message: dict) -> None:
        with self.lock:
            self.rooms.setdefault(room, []).append(dict(message))
            self._save_to_disk()

    def clear_room(self, room: str) -> None:
        with self.lock:
            removed_msgs = self.rooms.get(room, [])
            self.rooms[room] = []
            self._save_to_disk()

            still_referenced = {
                att.get("id")
                for r_msgs in self.rooms.values()
                for m in r_msgs
                for att in m.get("attachments", [])
            }
            for msg in removed_msgs:
                for att in msg.get("attachments", []):
                    att_id = att.get("id")
                    if att_id and att_id not in still_referenced:
                        self.file_blobs.pop(att_id, None)
                        stored_name = att.get("stored_name")
                        if stored_name:
                            try:
                                path = os.path.join(self.upload_dir, os.path.basename(stored_name))
                                if os.path.exists(path):
                                    os.remove(path)
                            except Exception:
                                pass

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
    return GroupChatStore(HISTORY_FILE, UPLOAD_DIR)


def _attachment_to_gemini_parts(att: dict, store: GroupChatStore) -> list:
    """Convert an attachment into Gemini Part(s) when bytes are available."""
    name = att.get("name", "file")
    mime_type = (att.get("mime_type") or "application/octet-stream").lower()
    raw_bytes = store.get_file_bytes(att.get("id", ""), att.get("stored_name"))

    if not raw_bytes:
        return [types.Part.from_text(text=f"[Shared file: {name} (unavailable)]")]

    if mime_type in GEMINI_INLINE_MIME_TYPES:
        try:
            return [types.Part.from_bytes(data=raw_bytes, mime_type=mime_type)]
        except Exception:
            pass

    if (
        mime_type.startswith("text/")
        or mime_type in {"application/json", "application/csv"}
        or name.lower().endswith(TEXT_FILE_EXTENSIONS)
    ):
        decoded = raw_bytes.decode("utf-8", errors="replace")[:20000]
        return [types.Part.from_text(text=f"[Attached file: {name}]\n{decoded}")]

    if mime_type.startswith("video/"):
        return [types.Part.from_text(text=f"[Shared video: {name}]")]

    return [types.Part.from_text(text=f"[Attached file: {name} ({mime_type})]")]


def prepare_gemini_turn_payload(messages: list[dict], store: GroupChatStore) -> tuple[list, list]:
    """
    Convert group messages (including images and files) into alternating Gemini turns
    and extract the latest user parts.
    """
    merged_turns: list[tuple[str, list]] = []

    for msg in messages:
        role = "user" if msg.get("role") == "user" else "model"
        content = (msg.get("content") or "").strip()
        attachments = msg.get("attachments") or []
        parts = []

        if role == "user":
            sender = msg.get("sender", "User")
            if content:
                parts.append(types.Part.from_text(text=f"[{sender}]: {content}"))
            elif attachments:
                parts.append(types.Part.from_text(text=f"[{sender}] shared:"))
            for att in attachments:
                parts.extend(_attachment_to_gemini_parts(att, store))
        elif content:
            parts.append(types.Part.from_text(text=content))

        if not parts:
            continue

        if merged_turns and merged_turns[-1][0] == role:
            merged_turns[-1][1].extend(parts)
        else:
            merged_turns.append((role, parts))

    if not merged_turns or merged_turns[-1][0] != "user":
        return [], []

    latest_prompt_parts = merged_turns[-1][1]
    history_turns = merged_turns[:-1]

    gemini_history = []
    for role, turn_parts in history_turns:
        gemini_history.append(types.Content(role=role, parts=turn_parts))

    return gemini_history, latest_prompt_parts


def init_chat_session(client, history: list, room: str):
    """Create a Gemini chat session aware of all group chat participants."""
    config = types.GenerateContentConfig(
        system_instruction=(
            f"You are {AI_NAME}, a helpful, conversational AI participating in a multi-person "
            f"group chat room named '{room}'. User messages are prefixed with `[Username]:` so "
            "you know who is speaking. Participants can also share images and files. Address "
            "participants by name when relevant, remember each person's preferences and context "
            "separately, and keep replies concise and natural."
        )
    )
    return client.chats.create(
        model=MODEL_NAME,
        config=config,
        history=history,
    )


def should_ai_respond(prompt_text: str, ai_mode: str) -> bool:
    if ai_mode == "Every message":
        return True
    return bool(re.search(r"(?i)@ai\b", prompt_text))


def parse_video_command(prompt_text: str) -> str | None:
    """Return the video prompt if the user invoked `/video`, otherwise None."""
    match = re.match(r"^\s*(?:@ai\s+)?/video\b[:\s]*(.*)$", prompt_text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def parse_audio_command(prompt_text: str) -> str | None:
    """Return the spoken text prompt if the user invoked `/audio`, otherwise None."""
    match = re.match(r"^\s*(?:@ai\s+)?/audio\b[:\s]*(.*)$", prompt_text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _pcm_to_wav_bytes(raw_pcm: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw 16-bit mono PCM bytes into a standard WAV container for browser playback."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_pcm)
    return buf.getvalue()


def generate_audio_attachment(
    client,
    text_prompt: str,
    model_name: str,
    voice_name: str,
    store: GroupChatStore,
) -> dict:
    """Generate speech audio via Gemini TTS and store it in GroupChatStore."""
    response = client.models.generate_content(
        model=model_name,
        contents=text_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    )
                )
            ),
        ),
    )

    audio_data = None
    audio_mime = ""
    for candidate in getattr(response, "candidates", None) or []:
        parts = getattr(getattr(candidate, "content", None), "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                audio_data = inline_data.data
                audio_mime = (getattr(inline_data, "mime_type", None) or "").lower()
                break
        if audio_data:
            break

    if not audio_data:
        raise RuntimeError("No audio was returned by the model.")

    if isinstance(audio_data, str):
        import base64
        audio_data = base64.b64decode(audio_data)

    if audio_data.startswith(b"RIFF") or "wav" in audio_mime:
        final_bytes = audio_data
        ext = "wav"
        final_mime = "audio/wav"
    elif "mp3" in audio_mime or "mpeg" in audio_mime:
        final_bytes = audio_data
        ext = "mp3"
        final_mime = "audio/mpeg"
    else:
        rate_match = re.search(r"rate=(\d+)", audio_mime)
        sample_rate = int(rate_match.group(1)) if rate_match else 24000
        final_bytes = _pcm_to_wav_bytes(audio_data, sample_rate=sample_rate)
        ext = "wav"
        final_mime = "audio/wav"

    return store.save_bytes_attachment(
        raw_bytes=final_bytes,
        file_name=f"voice_{uuid.uuid4().hex[:8]}.{ext}",
        mime_type=final_mime,
    )


def generate_video_attachment(
    client,
    prompt_text: str,
    model_name: str,
    aspect_ratio: str,
    store: GroupChatStore,
    reference_image: tuple[bytes, str] | None = None,
    status_container=None,
) -> dict:
    """Generate a video via Gemini Veo and store it in GroupChatStore."""
    kwargs = {
        "model": model_name,
        "prompt": prompt_text,
        "config": types.GenerateVideosConfig(aspect_ratio=aspect_ratio),
    }
    if reference_image is not None:
        img_bytes, img_mime = reference_image
        kwargs["image"] = types.Image(image_bytes=img_bytes, mime_type=img_mime)

    operation = client.models.generate_videos(**kwargs)

    max_polls = 30
    elapsed = 0
    for _ in range(max_polls):
        if getattr(operation, "done", False):
            break
        if status_container is not None:
            status_container.write(f"Rendering video frames... ({elapsed}s elapsed)")
        time.sleep(8)
        elapsed += 8
        operation = client.operations.get(operation)

    if not getattr(operation, "done", False):
        raise TimeoutError("Video generation timed out. Please try again.")

    if getattr(operation, "error", None):
        raise RuntimeError(str(operation.error))

    generated_videos = getattr(getattr(operation, "response", None), "generated_videos", None) or []
    if not generated_videos:
        raise RuntimeError("No video was returned by the model.")

    video_file = generated_videos[0].video
    downloaded = client.files.download(file=video_file)
    if isinstance(downloaded, (bytes, bytearray)):
        video_bytes = bytes(downloaded)
    else:
        tmp_path = os.path.join(store.upload_dir, f"tmp_{uuid.uuid4().hex}.mp4")
        os.makedirs(store.upload_dir, exist_ok=True)
        video_file.save(tmp_path)
        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return store.save_bytes_attachment(
        raw_bytes=video_bytes,
        file_name=f"veo_{uuid.uuid4().hex[:8]}.mp4",
        mime_type="video/mp4",
    )


def _render_attachments(msg: dict, store: GroupChatStore, key_prefix: str) -> None:
    msg_id = msg.get("id", "msg")
    for idx, att in enumerate(msg.get("attachments", [])):
        att_id = att.get("id", str(idx))
        name = att.get("name", "attachment")
        mime_type = (att.get("mime_type") or "").lower()
        raw_bytes = store.get_file_bytes(att_id, att.get("stored_name"))

        if not raw_bytes:
            st.caption(f"📎 {name} *(expired after restart)*")
            continue

        if mime_type.startswith("image/"):
            st.image(raw_bytes, caption=name)
        elif mime_type.startswith("audio/"):
            st.audio(raw_bytes, format=mime_type or "audio/wav")
            size_label = _format_file_size(att.get("size", len(raw_bytes)))
            st.download_button(
                label=f"⬇️ Save {name} ({size_label})",
                data=raw_bytes,
                file_name=name,
                mime=mime_type or "audio/wav",
                key=f"{key_prefix}_{msg_id}_{att_id}_{idx}_dl",
            )
        elif mime_type.startswith("video/"):
            st.video(raw_bytes)
            size_label = _format_file_size(att.get("size", len(raw_bytes)))
            st.download_button(
                label=f"⬇️ Save {name} ({size_label})",
                data=raw_bytes,
                file_name=name,
                mime=mime_type or "video/mp4",
                key=f"{key_prefix}_{msg_id}_{att_id}_{idx}_dl",
            )
        else:
            size_label = _format_file_size(att.get("size", len(raw_bytes)))
            st.download_button(
                label=f"📄 {name} ({size_label})",
                data=raw_bytes,
                file_name=name,
                mime=mime_type or "application/octet-stream",
                key=f"{key_prefix}_{msg_id}_{att_id}_{idx}",
            )


def render_message(
    msg: dict,
    current_user: str,
    store: GroupChatStore,
    key_prefix: str = "feed",
) -> None:
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
            if content:
                st.markdown(content)
            _render_attachments(msg, store, key_prefix)
    elif sender == current_user:
        _, avatar = get_user_style(sender)
        with st.chat_message("user", avatar=avatar):
            st.markdown(
                f'<div class="msg-header msg-self">'
                f'<span class="msg-sender">You ({escaped_sender})</span>'
                f"{time_html}</div>",
                unsafe_allow_html=True,
            )
            if content:
                st.markdown(content)
            _render_attachments(msg, store, key_prefix)
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
            if content:
                st.markdown(content)
            _render_attachments(msg, store, key_prefix)


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

    /* Rounded images and videos in messages */
    [data-testid="stChatMessage"] img,
    [data-testid="stChatMessage"] [data-testid="stImage"] img {
        border-radius: 14px !important;
        max-width: min(320px, 100%) !important;
        width: auto !important;
        height: auto !important;
        margin-top: 6px !important;
    }

    [data-testid="stChatMessage"] video {
        border-radius: 14px !important;
        max-width: min(360px, 100%) !important;
        margin-top: 6px !important;
        border: 1px solid rgba(255, 179, 198, 0.24) !important;
    }

    [data-testid="stChatMessage"] audio {
        min-width: 240px !important;
        max-width: min(340px, 100%) !important;
        margin-top: 6px !important;
        border-radius: 12px !important;
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
        color: #ff4d79 !important;
        transition: transform 0.15s ease !important;
    }

    [data-testid="stChatInput"] button:hover {
        transform: scale(1.16) !important;
    }

    [data-testid="stChatInputSubmitButton"] svg {
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

if "video_model" not in st.session_state:
    st.session_state.video_model = VIDEO_MODELS[0]

if "video_aspect_ratio" not in st.session_state:
    st.session_state.video_aspect_ratio = VIDEO_ASPECT_RATIOS[0]

if "audio_model" not in st.session_state:
    st.session_state.audio_model = AUDIO_MODELS[0]

if "audio_voice" not in st.session_state:
    st.session_state.audio_voice = AUDIO_VOICES[0]

if "auto_voice_replies" not in st.session_state:
    st.session_state.auto_voice_replies = False

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

    st.divider()
    st.subheader("Video Studio 🎬")
    st.session_state.video_model = st.selectbox(
        "Video model",
        options=VIDEO_MODELS,
        index=VIDEO_MODELS.index(st.session_state.video_model)
        if st.session_state.video_model in VIDEO_MODELS
        else 0,
    )
    st.session_state.video_aspect_ratio = st.radio(
        "Aspect ratio",
        options=VIDEO_ASPECT_RATIOS,
        horizontal=True,
        index=VIDEO_ASPECT_RATIOS.index(st.session_state.video_aspect_ratio)
        if st.session_state.video_aspect_ratio in VIDEO_ASPECT_RATIOS
        else 0,
    )
    st.caption("Type `/video <prompt>` in chat (attach a photo for image-to-video).")

    st.divider()
    st.subheader("Audio Studio 🎙️")
    st.session_state.audio_model = st.selectbox(
        "TTS model",
        options=AUDIO_MODELS,
        index=AUDIO_MODELS.index(st.session_state.audio_model)
        if st.session_state.audio_model in AUDIO_MODELS
        else 0,
    )
    st.session_state.audio_voice = st.selectbox(
        "Voice",
        options=AUDIO_VOICES,
        index=AUDIO_VOICES.index(st.session_state.audio_voice)
        if st.session_state.audio_voice in AUDIO_VOICES
        else 0,
    )
    st.session_state.auto_voice_replies = st.checkbox(
        "Speak AI text replies aloud",
        value=st.session_state.auto_voice_replies,
        help="Also attach a voice note whenever Cupid replies with text.",
    )
    st.caption("Type `/audio <text>` in chat (e.g. `/audio Say warmly: Good night!`).")

    if not hasattr(st, "fragment"):
        if st.button("Refresh messages", use_container_width=True):
            st.rerun()
    if st.button("Clear lounge history", use_container_width=True):
        store.clear_room(room)
        st.rerun()


def _render_chat_feed() -> None:
    current_messages = store.get_messages(st.session_state.room)
    for message in current_messages:
        render_message(
            message,
            current_user=st.session_state.username,
            store=store,
            key_prefix="feed",
        )
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

CHAT_INPUT_SUPPORTS_FILES = "accept_file" in inspect.signature(st.chat_input).parameters
fallback_uploads = []

if not CHAT_INPUT_SUPPORTS_FILES:
    if "uploader_nonce" not in st.session_state:
        st.session_state.uploader_nonce = 0
    uploader_container = (
        st.popover("📎 Share photos or files")
        if hasattr(st, "popover")
        else st.expander("📎 Share photos or files", expanded=False)
    )
    with uploader_container:
        fallback_uploads = st.file_uploader(
            "Attach pictures or files to your next message",
            type=ALLOWED_UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"fallback_uploader_{st.session_state.uploader_nonce}",
        ) or []

placeholder_text = (
    f"Message #{room}, attach 📎, /audio <text>, or /video <scene>..."
    if st.session_state.ai_mode == "Every message"
    else f"Message #{room}, attach 📎, @ai, /audio, or /video..."
)

if CHAT_INPUT_SUPPORTS_FILES:
    prompt = st.chat_input(
        placeholder_text,
        accept_file="multiple",
        file_type=ALLOWED_UPLOAD_TYPES,
    )
else:
    prompt = st.chat_input(placeholder_text)

if prompt:
    if hasattr(prompt, "text") or hasattr(prompt, "files"):
        prompt_text = (getattr(prompt, "text", "") or "").strip()
        uploaded_files = list(getattr(prompt, "files", None) or [])
    else:
        prompt_text = str(prompt).strip()
        uploaded_files = list(fallback_uploads)

    attachments = []
    oversized_names = []
    for uploaded_file in uploaded_files:
        if getattr(uploaded_file, "size", len(uploaded_file.getvalue())) > MAX_UPLOAD_BYTES:
            oversized_names.append(getattr(uploaded_file, "name", "file"))
            continue
        attachments.append(store.save_attachment(uploaded_file))

    if oversized_names:
        st.warning(
            f"Skipped files over 15 MB: {', '.join(oversized_names)}"
        )

    if not prompt_text and not attachments:
        st.stop()

    if not CHAT_INPUT_SUPPORTS_FILES and fallback_uploads:
        st.session_state.uploader_nonce += 1

    video_prompt = parse_video_command(prompt_text)
    audio_prompt = parse_audio_command(prompt_text)

    now_str = datetime.now().strftime("%H:%M")
    user_msg = {
        "id": uuid.uuid4().hex[:12],
        "role": "user",
        "sender": username,
        "color": user_color,
        "timestamp": now_str,
        "content": prompt_text,
        "attachments": attachments,
    }
    store.add_message(room, user_msg)

    if audio_prompt is not None:
        render_message(user_msg, current_user=username, store=store, key_prefix="preview")

        if not audio_prompt:
            st.warning(
                "Add the text you want spoken after `/audio` "
                "(for example: `/audio Say warmly: You make my evening brighter`)."
            )
            st.stop()

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
            with st.spinner(f"🎙️ Generating voice note ({st.session_state.audio_voice})..."):
                try:
                    if st.session_state.client is None:
                        st.session_state.client = genai.Client(api_key=api_key)

                    audio_att = generate_audio_attachment(
                        client=st.session_state.client,
                        text_prompt=audio_prompt,
                        model_name=st.session_state.audio_model,
                        voice_name=st.session_state.audio_voice,
                        store=store,
                    )
                except APIError as error:
                    st.error(f"Audio generation failed: {error}")
                except Exception as error:
                    st.error(f"Could not generate audio: {error}")
                else:
                    store.add_message(
                        room,
                        {
                            "id": uuid.uuid4().hex[:12],
                            "role": "assistant",
                            "sender": AI_NAME,
                            "color": AI_COLOR,
                            "timestamp": datetime.now().strftime("%H:%M"),
                            "content": f"🎙️ Voice note ({st.session_state.audio_voice}): *{audio_prompt}*",
                            "attachments": [audio_att],
                        },
                    )
                    st.rerun()
        st.stop()

    if video_prompt is not None:
        first_image = next(
            (a for a in attachments if (a.get("mime_type") or "").startswith("image/")),
            None,
        )
        effective_video_prompt = video_prompt or (
            "Smooth cinematic motion bringing this image to life." if first_image else ""
        )

        render_message(user_msg, current_user=username, store=store, key_prefix="preview")

        if not effective_video_prompt:
            st.warning(
                "Describe the video after `/video` (for example: `/video sunset boat ride in Venice`), "
                "or attach a photo with `/video`."
            )
            st.stop()

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
            with st.status(
                f"🎬 Creating video with {st.session_state.video_model}...",
                expanded=True,
            ) as status:
                try:
                    if st.session_state.client is None:
                        st.session_state.client = genai.Client(api_key=api_key)

                    ref_image = None
                    if first_image is not None:
                        img_bytes = store.get_file_bytes(
                            first_image["id"], first_image.get("stored_name")
                        )
                        if img_bytes:
                            ref_image = (img_bytes, first_image["mime_type"])

                    video_att = generate_video_attachment(
                        client=st.session_state.client,
                        prompt_text=effective_video_prompt,
                        model_name=st.session_state.video_model,
                        aspect_ratio=st.session_state.video_aspect_ratio,
                        store=store,
                        reference_image=ref_image,
                        status_container=status,
                    )
                except APIError as error:
                    status.update(label="Video generation failed", state="error")
                    st.error(
                        f"Video request failed: {error}. "
                        "Note: Video generation typically requires a billing-enabled Gemini API key."
                    )
                except Exception as error:
                    status.update(label="Video generation failed", state="error")
                    st.error(f"Could not generate video: {error}")
                else:
                    status.update(label="🎬 Video ready!", state="complete", expanded=False)
                    store.add_message(
                        room,
                        {
                            "id": uuid.uuid4().hex[:12],
                            "role": "assistant",
                            "sender": AI_NAME,
                            "color": AI_COLOR,
                            "timestamp": datetime.now().strftime("%H:%M"),
                            "content": f"🎬 Generated video for: *{effective_video_prompt}*",
                            "attachments": [video_att],
                        },
                    )
                    st.rerun()
        st.stop()

    if not should_ai_respond(prompt_text, st.session_state.ai_mode):
        st.rerun()

    # Show the user's message immediately while streaming the AI reply
    render_message(user_msg, current_user=username, store=store, key_prefix="preview")

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
            history, latest_user_parts = prepare_gemini_turn_payload(latest_messages, store)
            chat = init_chat_session(st.session_state.client, history, room)

            for chunk in chat.send_message_stream(latest_user_parts):
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
            reply_attachments = []
            if st.session_state.auto_voice_replies and full_response:
                try:
                    voice_att = generate_audio_attachment(
                        client=st.session_state.client,
                        text_prompt=full_response[:1500],
                        model_name=st.session_state.audio_model,
                        voice_name=st.session_state.audio_voice,
                        store=store,
                    )
                    reply_attachments.append(voice_att)
                except Exception:
                    pass

            store.add_message(
                room,
                {
                    "id": uuid.uuid4().hex[:12],
                    "role": "assistant",
                    "sender": AI_NAME,
                    "color": AI_COLOR,
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "content": full_response,
                    "attachments": reply_attachments,
                },
            )
            st.rerun()
