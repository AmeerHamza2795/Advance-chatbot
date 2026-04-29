import os
import json
import uuid
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="Groq Chatbot Pro",
    page_icon="💬",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}
.main {
    background: linear-gradient(135deg, #0b1020 0%, #111827 50%, #0f172a 100%);
}
.block-container {
    max-width: 950px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid rgba(255,255,255,0.08);
}
.app-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f9fafb;
    margin-bottom: 0.2rem;
}
.app-subtitle {
    color: #94a3b8;
    margin-bottom: 1rem;
}
.info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 14px 16px;
    border-radius: 16px;
    margin-bottom: 14px;
}
.copy-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 10px 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Constants ----------
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Respond clearly, accurately, and concisely."
DEFAULT_MODEL = "openai/gpt-oss-120b"
CHAT_FILE = "chat_history.json"
MAX_HISTORY_PAIRS = 20

# ---------- Helpers ----------
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_chat_id():
    return str(uuid.uuid4())[:8]


def make_chat_title(messages):
    for role, content in messages:
        if role == "user" and content.strip():
            title = content.strip().replace("\n", " ")
            return title[:30] + ("..." if len(title) > 30 else "")
    return "New Chat"


def trim_history(messages, max_pairs=20):
    system_msg = messages[0] if messages and messages[0][0] == "system" else None
    non_system = [m for m in messages if m[0] != "system"]

    max_items = max_pairs * 2
    if len(non_system) > max_items:
        non_system = non_system[-max_items:]

    if system_msg:
        return [system_msg] + non_system
    return non_system


def save_chats_to_file():
    data = {
        "current_chat_id": st.session_state.current_chat_id,
        "chats": st.session_state.chats
    }
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_chats_from_file():
    if not os.path.exists(CHAT_FILE):
        return {}, None

    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        chats = data.get("chats", {})
        current_chat_id = data.get("current_chat_id")

        for chat_id in chats:
            chats[chat_id]["messages"] = [
                tuple(item) if isinstance(item, list) else item
                for item in chats[chat_id]["messages"]
            ]

        return chats, current_chat_id
    except Exception:
        return {}, None


def create_new_chat():
    chat_id = generate_chat_id()
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "created_at": now_str(),
        "messages": [("system", DEFAULT_SYSTEM_PROMPT)]
    }
    st.session_state.current_chat_id = chat_id
    save_chats_to_file()


def get_current_chat():
    return st.session_state.chats[st.session_state.current_chat_id]


def get_current_messages():
    return get_current_chat()["messages"]


def set_current_messages(messages):
    st.session_state.chats[st.session_state.current_chat_id]["messages"] = messages
    if get_current_chat()["title"] == "New Chat":
        st.session_state.chats[st.session_state.current_chat_id]["title"] = make_chat_title(messages)
    save_chats_to_file()


def rename_current_chat(new_title):
    new_title = new_title.strip()
    if new_title:
        st.session_state.chats[st.session_state.current_chat_id]["title"] = new_title
        save_chats_to_file()


# def export_current_chat():
#     chat = get_current_chat()
#     data = {
#         "title": chat["title"],
#         "created_at": chat["created_at"],
#         "messages": [
#             {"role": role, "content": content}
#             for role, content in chat["messages"]
#             if role != "system"
#         ]
#     }
#     return json.dumps(data, indent=2, ensure_ascii=False)


def get_last_assistant_reply():
    messages = get_current_messages()
    for role, content in reversed(messages):
        if role == "assistant":
            return content
    return ""


def build_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        return None, "GROQ_API_KEY not found. Add it in your .env file."

    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=DEFAULT_MODEL,
            temperature=0.7,
        )
        return llm, None
    except Exception as e:
        return None, f"Model initialization failed: {e}"


def generate_response(messages):
    llm, error = build_llm()
    if error:
        return None, error

    try:
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({})
        return response, None
    except Exception as e:
        return None, f"Response generation failed: {e}"


def stream_text(text, delay=0.01):
    placeholder = st.empty()
    current = ""

    for char in text:
        current += char
        placeholder.markdown(current)
        time.sleep(delay)

    return placeholder


# ---------- Session State ----------
if "chats" not in st.session_state or "current_chat_id" not in st.session_state:
    loaded_chats, loaded_current_chat_id = load_chats_from_file()
    st.session_state.chats = loaded_chats
    st.session_state.current_chat_id = loaded_current_chat_id

if "rename_value" not in st.session_state:
    st.session_state.rename_value = ""

if not st.session_state.chats:
    create_new_chat()

if st.session_state.current_chat_id not in st.session_state.chats:
    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
    save_chats_to_file()

# ---------- Sidebar ----------
with st.sidebar:
    st.title("💬 Chats")

    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    chat_ids = list(st.session_state.chats.keys())[::-1]

    selected_chat = st.radio(
        "Previous Chat History",
        options=chat_ids,
        index=chat_ids.index(st.session_state.current_chat_id),
        format_func=lambda x: st.session_state.chats[x]["title"],
    )
    st.session_state.current_chat_id = selected_chat
    save_chats_to_file()

    st.markdown("---")

    current_chat = get_current_chat()
    st.session_state.rename_value = st.text_input(
        "Rename Current Chat",
        value=current_chat["title"],
        key="rename_box"
    )

    if st.button("✏️ Save Chat Name", use_container_width=True):
        rename_current_chat(st.session_state.rename_value)
        st.rerun()

    st.markdown("---")

    if st.button("🧹 Clear Current Chat", use_container_width=True):
        st.session_state.chats[st.session_state.current_chat_id]["messages"] = [
            ("system", DEFAULT_SYSTEM_PROMPT)
        ]
        st.session_state.chats[st.session_state.current_chat_id]["title"] = "New Chat"
        save_chats_to_file()
        st.rerun()

    if st.button("🗑 Delete Current Chat", use_container_width=True):
        if len(st.session_state.chats) > 1:
            del st.session_state.chats[st.session_state.current_chat_id]
            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
            save_chats_to_file()
            st.rerun()
        else:
            st.warning("At least one chat must remain.")

    # st.download_button(
    #     "📥 Export Current Chat",
    #     data=export_current_chat(),
    #     file_name="current_chat.json",
    #     mime="application/json",
    #     use_container_width=True
    # )

    # last_reply = get_last_assistant_reply()
    # if last_reply:
    #     st.download_button(
    #         "📋 Copy Last Reply",
    #         data=last_reply,
    #         file_name="last_reply.txt",
    #         mime="text/plain",
    #         use_container_width=True
    #     )

    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY missing in .env")

# ---------- Header ----------
st.markdown("<div class='main'></div>", unsafe_allow_html=True)
st.markdown('<div class="app-title">💬 Groq Chatbot Pro</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Clean UI with saved chat history, rename, and streaming reply</div>',
    unsafe_allow_html=True
)

current_chat = get_current_chat()
messages = current_chat["messages"]

if len([m for m in messages if m[0] != "system"]) == 0:
    st.markdown("""
    <div class="info-card">
        <b>Ready to chat.</b><br>
    </div>
    """, unsafe_allow_html=True)

# ---------- Render Messages ----------
for role, message in messages:
    if role == "system":
        continue

    avatar = "🧑" if role == "user" else "🤖"
    with st.chat_message("user" if role == "user" else "assistant", avatar=avatar):
        st.markdown(message)

# ---------- Chat Input ----------
user_input = st.chat_input("Type your message here...")

if user_input and user_input.strip():
    user_input = user_input.strip()

    messages = get_current_messages()
    messages.append(("user", user_input))
    messages = trim_history(messages, MAX_HISTORY_PAIRS)
    set_current_messages(messages)

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response, error = generate_response(messages)

            if error:
                st.error(error)
                assistant_text = f"Error: {error}"
            else:
                stream_text(response, delay=0.008)
                assistant_text = response

    messages = get_current_messages()
    messages.append(("assistant", assistant_text))
    messages = trim_history(messages, MAX_HISTORY_PAIRS)
    set_current_messages(messages)

    st.rerun()

# ---------- Bottom Copy Area ----------
last_reply = get_last_assistant_reply()
if last_reply:
    st.markdown("---")
    st.markdown("**Last Assistant Reply**")
    st.code(last_reply, language=None)
