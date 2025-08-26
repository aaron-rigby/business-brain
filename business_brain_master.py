import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from notion_client import Client
import openai
from pinecone import Pinecone, ServerlessSpec
import os
import traceback
import time
import json

# Add logging
import logging
from datetime import datetime

logging.basicConfig(
    filename='/volume1/Shared/business_brain/logs/rag_update.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

logging.info(f"RAG Update Started: {datetime.now()}")
# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Business Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# DEBUG MODE TOGGLE
# ============================================
DEBUG_MODE = st.sidebar.checkbox("🔧 Debug Mode", value=False)
USE_HARDCODED = st.sidebar.checkbox("🔐 Use Hardcoded Credentials", value=False)

# ============================================
# CREDENTIAL MANAGEMENT
# ============================================
def get_credentials():
    """Get credentials with fallback options"""
    if USE_HARDCODED:
        st.sidebar.info("Using hardcoded credentials")
        return {
            "NOTION_TOKEN": "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",
            "PIPELINE_DB_ID": "24d6e7c238388037b0d1eb52ba9c2b29",
            "MEETING_DB_ID": st.secrets.get("MEETING_DB_ID", ""),
            "OPENAI_API_KEY": st.secrets.get("OPENAI_API_KEY", ""),
            "PINECONE_API_KEY": st.secrets.get("PINECONE_API_KEY", ""),
            "PINECONE_INDEX_NAME": st.secrets.get("PINECONE_INDEX_NAME", "business-brain"),
        }
    else:
        try:
            return {
                "NOTION_TOKEN": st.secrets.get("NOTION_TOKEN"),
                "PIPELINE_DB_ID": st.secrets.get("PIPELINE_DB_ID"),
                "MEETING_DB_ID": st.secrets.get("MEETING_DB_ID", ""),
                "OPENAI_API_KEY": st.secrets.get("OPENAI_API_KEY", ""),
                "PINECONE_API_KEY": st.secrets.get("PINECONE_API_KEY", ""),
                "PINECONE_INDEX_NAME": st.secrets.get("PINECONE_INDEX_NAME", "business-brain"),
            }
        except Exception as e:
            st.error(f"Failed to load secrets: {e}")
            return None

# ============================================
# NOTION CONNECTION - FIXED CACHING
# ============================================
@st.cache_resource
def get_notion_client():
    """Initialize Notion client with caching"""
    creds = get_credentials()
    if not creds or not creds.get("NOTION_TOKEN"):
        st.error("No Notion token available")
        return None
    
    try:
        client = Client(auth=creds["NOTION_TOKEN"])
        if DEBUG_MODE:
            st.sidebar.success("✅ Notion client initialized")
        return client
    except Exception as e:
        st.error(f"Failed to initialize Notion client: {e}")
        if DEBUG_MODE:
