from rag import rag_chain_with_source_multi as chain
from rag import rag_chain_with_source_basic as chain_basic

import streamlit as st
import random
import time
import requests
import json
import re
from urllib.parse import unquote
# from streamlit_pills import pills
import hmac
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

def start_countdown(n=60):
    st.markdown(""" ## We are facing High Demand at the moment.

    Currently we are at free tier,
    Please wait untill the API is back live.
    If you feel this tool is useful, please contact us: 

    Thank you for your patience Please Wait for 60 seconds.""")
    ph = st.empty()
    N = n
    for secs in range(N,0,-1):
        mm, ss = secs//60, secs%60
        ph.metric("", f"{mm:02d}:{ss:02d}")
        time.sleep(1)


collector = None

widget_id = (id for id in range(1, 10000))


def chatHere():
    """
    Function to handle the chat functionality.

    This function displays chat messages from history on app rerun, accepts user input,
    and generates assistant responses based on the user's input.

    Returns:
        None
    """
    # Display chat messages from history on app rerun
    # print(st.session_state.messages)

    for n, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and n >= 1:
                feedback_key = f"feedback_{int(n/2)}"

                if feedback_key not in st.session_state:
                    st.session_state[feedback_key] = None

            if "source" in message and len(list(message["source"])) > 1:
                selected = None
                with st.expander("Click to view Sources", expanded=False):
                    st.write(message["source"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner(
                    'Fetching the Response for your Query from the SelfQuery, Multi Retreiver...'
            ):
                try:
                    response = chain.invoke(prompt)
                except:
                    st.write("Error in fetching response from the AI Engine, Hence using the basic model for response.")
                    try:
                        response = chain_basic.invoke(prompt)
                    except:
                        response = {
                            "answer":
                            "Sorry, I didn't get that. Please try again."
                        }
                        message_placeholder.markdown(response["answer"])
                        start_countdown()
                        st.stop()
            try:
                assistant_response = response["answer"]
                source = response["context"]
            except Exception as e:
                print(e)
                assistant_response = "Sorry, I didn't get that. Please try again."
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            for chunk in re.split(r'(\s+)', assistant_response):
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(f"{full_response}")

        st.session_state.history.append({
            "Human": prompt,
            "AI": assistant_response
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "prompt": prompt,
            "source": source
        })
        st.rerun()
