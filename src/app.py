import streamlit as st
import random
import time
import requests
import json
import re
from urllib.parse import unquote

from chat import chatHere


def set_header():
    """
    Sets the header and user information for the Philosopher RAG Chatbot app.

    This function sets the title and user information section for the Philosopher RAG Chatbot app.
    It displays an overview of the Philosopher RAG model and provides example queries.

    Args:
        None

    Returns:
        None
    """
    st.title("Philosopher RAG Chatbot")
    with st.expander("User Information", expanded=True):
        st.markdown("""##### Philosopher RAG Overview:""")
        st.markdown(
            """The Philosopher RAG model is a conversational AI model that can answer questions and provide information on a wide range of topics."""
        )
        st.markdown("Example query:")
        st.markdown("> Get me a quote from Plato about the nature of reality.")
        st.markdown("> what did aristotle quoted about ethics?")

        st.markdown(
            "        *Please note: Chat history is not stored; each query is treated as a new session.* "
        )

if "logged_prompt" not in st.session_state:
    st.session_state.logged_prompt = None
if "feedback_key" not in st.session_state:
    st.session_state.feedback_key = 0
if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = False
if "user_id" not in st.session_state:
    st.session_state.user_id = "test"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = [{"Human": "", "AI": ""}]


set_header()
chatHere()
