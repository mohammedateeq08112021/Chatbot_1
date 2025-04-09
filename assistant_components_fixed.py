
"""Components for the Interactive AI Assistant."""

import re
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from openai import OpenAI

class URLFeatureExtractor:
    # ... (unchanged)
    pass

class URLReputationChecker:
    # ... (unchanged)
    pass

class URLAnalyzer:
    # ... (unchanged)
    pass

class ThreatDetectionModel:
    # ... (unchanged)
    pass

class AIAssistant:
    """Handles educational queries using AI models."""
    
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.client = None
        if api_key:
            self.client = OpenAI(api_key=api_key)
    
    def set_api_key(self, api_key):
        """Set the OpenAI API key."""
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
    def set_model(self, model):
        """Set the model to use."""
        self.model = model
    
    def get_response(self, query, conversation_history=None):
        if not self.client:
            return "Error: API key not set. Please set an API key to use the AI assistant."
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful educational assistant specializing in cybersecurity. Provide clear, accurate, and educational responses to user queries."}
            ]
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": query})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_streaming_response(self, query, conversation_history=None):
        if not self.client:
            yield "Error: API key not set. Please set an API key to use the AI assistant."
            return
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful educational assistant specializing in cybersecurity. Provide clear, accurate, and educational responses to user queries."}
            ]
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": query})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error: {str(e)}"

class QueryRouter:
    # ... (unchanged)
    pass

class InteractiveAssistant:
    # ... (unchanged)
    pass
