# PHILLZ Chat

PHILLZ Chat is a Streamlit chat app powered by Google's Gemini API. It streams replies as they are generated and saves conversation history locally so it can be restored when the app restarts.

## Requirements

- Python 3.9 or later
- A Google Gemini API key

## Setup

Install the required packages:

```bash
python -m pip install streamlit google-genai
```

Provide your API key using either an environment variable:

```bash
export GEMINI_API_KEY="your-api-key"
```

Or create `.streamlit/secrets.toml` in the project directory:

```toml
GEMINI_API_KEY = "your-api-key"
```

Keep your API key private. Do not commit it to the repository.

## Run

From the project directory, start the app with:

```bash
streamlit run chat.py
```

Open the local URL printed by Streamlit in your browser.

## Conversation history

Messages are stored in `chat_history.json` next to `chat.py` and loaded the next time the app starts. Use **Start a new chat** in the sidebar to clear the conversation and its saved history. The history file may contain private conversation content; treat it accordingly and avoid committing it.

The app uses the `gemini-3.1-flash-lite` model. A working internet connection and a valid API key are needed to send messages.
