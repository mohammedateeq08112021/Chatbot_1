import streamlit as st
import re
import os
import json
from datetime import datetime

# Import the components
from assistant_components import URLAnalyzer, ThreatDetectionModel, AIAssistant, QueryRouter, InteractiveAssistant

# Set page config
st.set_page_config(
    page_title="Interactive AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
.user-avatar {
    background-color: #2e7aff;
    color: white;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    text-align: center;
    line-height: 40px;
    margin-right: 10px;
}
.assistant-avatar {
    background-color: #ff5733;
    color: white;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    text-align: center;
    line-height: 40px;
    margin-right: 10px;
}
.chat-container {
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}
.user-container {
    background-color: #f0f2f6;
}
.assistant-container {
    background-color: #e6f7ff;
}
.risk-low {
    color: green;
    font-weight: bold;
}
.risk-medium {
    color: orange;
    font-weight: bold;
}
.risk-high {
    color: red;
    font-weight: bold;
}
.risk-critical {
    color: darkred;
    font-weight: bold;
}
</style>""", unsafe_allow_html=True)

# App title
st.title("Interactive AI Assistant")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # API key input
    api_key = st.text_input("OpenAI API Key", type="password")
    
    # Model selection
    model = st.selectbox(
        "AI Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    )
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.assistant.clear_history()
        st.experimental_rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "assistant" not in st.session_state or api_key != st.session_state.get("api_key", ""):
    st.session_state.assistant = InteractiveAssistant(openai_api_key=api_key)
    st.session_state.api_key = api_key

# Update model if changed
if "model" not in st.session_state or model != st.session_state.model:
    st.session_state.assistant.set_model(model)
    st.session_state.model = model

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check if this is a URL analysis result
        if message["role"] == "assistant" and message["content"].startswith("URL ANALYSIS RESULT"):
            # Parse the risk level
            risk_level_match = re.search(r"Risk Level: (\w+)", message["content"])
            if risk_level_match:
                risk_level = risk_level_match.group(1).lower()
                # Replace the risk level with styled version
                styled_content = message["content"].replace(
                    f"Risk Level: {risk_level_match.group(1)}",
                    f"Risk Level: <span class='risk-{risk_level}'>{risk_level_match.group(1)}</span>"
                )
                st.markdown(styled_content, unsafe_allow_html=True)
            else:
                st.markdown(message["content"])
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question or enter a URL to analyze..."):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from assistant
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.assistant.process_query(prompt)
                
                # Check if this is a URL analysis result
                if response.startswith("URL ANALYSIS RESULT"):
                    # Parse the risk level
                    risk_level_match = re.search(r"Risk Level: (\w+)", response)
                    if risk_level_match:
                        risk_level = risk_level_match.group(1).lower()
                        # Replace the risk level with styled version
                        styled_response = response.replace(
                            f"Risk Level: {risk_level_match.group(1)}",
                            f"Risk Level: <span class='risk-{risk_level}'>{risk_level_match.group(1)}</span>"
                        )
                        st.markdown(styled_response, unsafe_allow_html=True)
                    else:
                        st.markdown(response)
                else:
                    st.markdown(response)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})

# Display welcome message if no messages
if not st.session_state.messages:
    st.markdown("""
    👋 **Welcome to the Interactive AI Assistant!**
    
    I can help you with:
    - Answering educational questions about cybersecurity and other topics
    - Analyzing URLs for potential security threats
    
    To get started:
    1. Enter your OpenAI API key in the sidebar
    2. Ask me a question or enter a URL to analyze
    
    Example questions:
    - What is phishing and how can I protect myself?
    - How do secure passwords work?
    - https://example.com (to analyze a URL)
    """)
