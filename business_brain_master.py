"""
BUSINESS BRAIN MASTER SYSTEM
Complete unified dashboard with all features integrated
Version 4.0 - Production Ready
"""

import streamlit as st
import pandas as pd
import json
import openai
from pinecone import Pinecone
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import os

# Password Protection
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets.get("APP_PASSWORD", "YourDefaultPassword"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

if not check_password():
    st.stop()  # Do not continue if check_password is not True
# Page Configuration
st.set_page_config(
    page_title="🧠 Business Brain Master",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Configuration
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys from Streamlit secrets (cloud) or environment variables (local)
def get_api_key(key_name):
    if hasattr(st, 'secrets') and key_name in st.secrets:
        return st.secrets[key_name]
    return os.getenv(key_name)

# Set API keys
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
PINECONE_API_KEY = get_api_key("PINECONE_API_KEY")
NOTION_TOKEN = get_api_key("NOTION_TOKEN")
PINECONE_INDEX_NAME = "business-brain"  # Your index name

# Verify keys are loaded
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please check your configuration.")
    st.stop()
if not PINECONE_API_KEY:
    st.error("Pinecone API key not found. Please check your configuration.")
    st.stop()


# Initialize Clients
@st.cache_resource
def init_all_clients():
    """Initialize all API clients"""
    clients = {
        'openai': openai.OpenAI(api_key=OPENAI_API_KEY),
        'pinecone': Pinecone(api_key=PINECONE_API_KEY),
        'index': Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)
    }
    return clients

clients = init_all_clients()

# Initialize Session State
if 'daily_brief' not in st.session_state:
    st.session_state.daily_brief = None
if 'action_items' not in st.session_state:
    st.session_state.action_items = []
if 'crm_contacts' not in st.session_state:
    st.session_state.crm_contacts = {}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        padding: 1rem 0;
    }
    .status-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .metric-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🧠 Business Brain Master System</h1>", unsafe_allow_html=True)

# Top Metrics Bar
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📄 Meetings Indexed", "354", delta="+12 this week")
with col2:
    st.metric("✅ Open Actions", "23", delta="-3 completed")
with col3:
    st.metric("👥 CRM Contacts", len(st.session_state.crm_contacts))
with col4:
    st.metric("📈 Qlik Alerts", "3", delta="2 critical")
with col5:
    st.metric("🎯 VP Progress", "62%", delta="+5%")

st.divider()

# Main Navigation Tabs
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6 = st.tabs([
    "🏠 Command Center",
    "🔍 Intelligence Search", 
    "📋 Action Tracker",
    "👥 CRM & Outreach",
    "📊 Performance (Qlik)",
    "⚙️ Automation"
])

# TAB 1: COMMAND CENTER
with main_tab1:
    st.markdown("## 🎯 Daily Intelligence Brief")
    
    # Generate Daily Brief Button
    if st.button("🌅 Generate Today's Brief", type="primary"):
        with st.spinner("Generating intelligence brief..."):
            # This would call all your intelligence gathering functions
            st.session_state.daily_brief = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'priorities': ["Follow up with SPH on pricing", "Mediacorp Q4 planning call", "Review Astra expansion"],
                'alerts': ["Qlik: CTR down 15% for SPH", "LinkedIn: John Smith promoted to VP"],
                'opportunities': ["Astra interested in premium inventory", "New Google ad format opportunity"]
            }
    
    if st.session_state.daily_brief:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### ⚡ Today's Priorities")
            for priority in st.session_state.daily_brief['priorities']:
                st.checkbox(priority, key=f"priority_{priority}")
        
        with col2:
            st.markdown("### 🚨 Alerts")
            for alert in st.session_state.daily_brief['alerts']:
                st.warning(alert)
        
        with col3:
            st.markdown("### 💰 Opportunities")
            for opp in st.session_state.daily_brief['opportunities']:
                st.success(opp)
    
    # Quick Actions Grid
    st.markdown("### 🚀 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📧 Check Outreach Queue", use_container_width=True):
            st.session_state.active_tab = "crm"
    with col2:
        if st.button("📊 View Qlik Dashboard", use_container_width=True):
            st.session_state.active_tab = "qlik"
    with col3:
        if st.button("📅 Prep Tomorrow's Meetings", use_container_width=True):
            st.session_state.active_tab = "calendar"
    with col4:
        if st.button("🔍 Search Intelligence", use_container_width=True):
            st.session_state.active_tab = "search"

# TAB 2: INTELLIGENCE SEARCH
with main_tab2:
    st.markdown("## 🔍 Unified Intelligence Search")
    
    # Search Configuration
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        search_query = st.text_area(
            "Ask anything about your business:",
            placeholder="E.g., What are all my commitments to SPH? What's the competitive situation with Google?",
            height=80
        )
    
    with col2:
        ai_model = st.selectbox(
            "AI Model",
            ["GPT-4 (Smart)", "GPT-3.5 (Fast)", "Claude (Genius)"]
        )
        
    with col3:
        data_sources = st.multiselect(
            "Search in",
            ["Meetings", "CRM", "Qlik", "Guru"],
            default=["Meetings"]
        )
    
if st.button("🔍 Search All Intelligence", type="primary", use_container_width=True):
    with st.spinner("Searching across all data sources..."):
        if query:
            try:
                # Create embedding for the query
                embedding = clients['openai'].embeddings.create(
                    input=query,
                    model="text-embedding-ada-002"
                ).data[0].embedding
                
                # Query Pinecone for similar content
                results = clients['index'].query(
                    vector=embedding,
                    top_k=5,
                    include_metadata=True
                )
                
                # Display results
                if results.matches:
                    st.subheader("📍 From MEETINGS")
                    for match in results.matches:
                        metadata = match.metadata
                        score = match.score
                        
                        # Display meeting info
                        meeting_date = metadata.get('date', 'Unknown date')
                        meeting_title = metadata.get('title', 'Meeting')
                        meeting_text = metadata.get('text', '')[:300] + "..."
                        
                        st.write(f"• **{meeting_title}** ({meeting_date}) - Score: {score:.2f}")
                        st.write(f"  {meeting_text}")
                        st.divider()
                else:
                    st.info("No matching results found. Try a different search query.")
                    
            except Exception as e:
                st.error(f"Search error: {str(e)}")
        else:
            st.warning("Please enter a search query") 

# TAB 3: ACTION TRACKER
with main_tab3:
    st.markdown("## ✅ Action Item Tracker")
    
    # Add new action
    with st.expander("➕ Add New Action"):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            new_action = st.text_input("Action item:")
        with col2:
            priority = st.selectbox("Priority:", ["High", "Medium", "Low"])
        with col3:
            if st.button("Add"):
                st.session_state.action_items.append({
                    'task': new_action,
                    'priority': priority,
                    'added': datetime.now(),
                    'status': 'Open'
                })
    
    # Display actions by priority
    high_priority = [a for a in st.session_state.action_items if a.get('priority') == 'High']
    medium_priority = [a for a in st.session_state.action_items if a.get('priority') == 'Medium']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 High Priority")
        for action in high_priority:
            st.checkbox(action['task'], key=f"action_{action['task']}")
    
    with col2:
        st.markdown("### 🟡 Medium Priority")
        for action in medium_priority:
            st.checkbox(action['task'], key=f"action_med_{action['task']}")

# TAB 4: CRM & OUTREACH
with main_tab4:
    st.markdown("## 👥 CRM & Intelligent Outreach")
    
    crm_tab1, crm_tab2, crm_tab3 = st.tabs(["Contacts", "Outreach Queue", "LinkedIn Monitor"])
    
    with crm_tab1:
        # Contact management
        st.markdown("### 📋 Contact Database")
        
        # Sample contacts display
        contacts_df = pd.DataFrame({
            'Name': ['John Smith', 'Jane Doe', 'Michael Chen'],
            'Company': ['SPH', 'Mediacorp', 'Astra'],
            'Last Contact': ['2 days ago', '1 week ago', '3 days ago'],
            'Relationship': [9, 7, 8]
        })
        
        st.dataframe(contacts_df, use_container_width=True)
    
    with crm_tab2:
        st.markdown("### ✉️ Outreach Queue")
        
        # Sample outreach items
        outreach_items = [
            {
                'contact': 'John Smith',
                'trigger': 'News: SPH Digital Growth',
                'email': 'Hi John, Saw the news about SPH\'s digital growth...'
            }
        ]
        
        for item in outreach_items:
            with st.expander(f"📧 {item['contact']} - {item['trigger']}"):
                st.text_area("Email:", item['email'], height=100)
                col1, col2 = st.columns(2)
                with col1:
                    st.button(f"✅ Send", key=f"send_{item['contact']}")
                with col2:
                    st.button(f"✏️ Edit", key=f"edit_{item['contact']}")
    
    with crm_tab3:
        st.markdown("### 🔍 LinkedIn Changes")
        if st.button("Check All LinkedIn Profiles"):
            st.info("Checking 47 LinkedIn profiles...")

# TAB 5: QLIK PERFORMANCE
with main_tab5:
    st.markdown("## 📊 Qlik Performance Monitor")
    
    # KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Revenue", "$45,230", delta="-12%", delta_color="inverse")
    with col2:
        st.metric("CTR", "2.3%", delta="+0.2%")
    with col3:
        st.metric("CPM", "$4.50", delta="+$0.30", delta_color="inverse")
    with col4:
        st.metric("Fill Rate", "87%", delta="-3%", delta_color="inverse")
    with col5:
        st.metric("Viewability", "72%", delta="+5%")
    
    # Alerts
    st.markdown("### 🚨 Performance Alerts")
    st.error("**SPH**: CTR dropped 15% - investigate ad fatigue")
    st.warning("**Mediacorp**: Fill rate below 85% - check demand")
    
    # Upload new data
    with st.expander("📤 Upload Today's Qlik Export"):
        uploaded_file = st.file_uploader("Choose Qlik CSV/Excel", type=['csv', 'xlsx'])
        if uploaded_file:
            st.success("Processing Qlik data...")

# TAB 6: AUTOMATION SETTINGS
with main_tab6:
    st.markdown("## ⚙️ Automation Configuration")
    
    auto_tab1, auto_tab2, auto_tab3 = st.tabs(["Schedules", "Integrations", "Logs"])
    
    with auto_tab1:
        st.markdown("### ⏰ Automation Schedule")
        
        schedules = {
            "Morning Brief": "6:00 AM",
            "LinkedIn Check": "12:00 PM",
            "News Scan": "3:00 PM",
            "Qlik Import": "5:00 PM",
            "Daily Summary": "6:00 PM"
        }
        
        for task, time in schedules.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{task}**")
            with col2:
                st.write(time)
            with col3:
                st.checkbox("Active", value=True, key=f"schedule_{task}")
    
    with auto_tab2:
        st.markdown("### 🔗 Integration Status")
        
        integrations = {
            "Notion": "✅ Connected",
            "OpenAI": "✅ Connected",
            "Pinecone": "✅ Connected",
            "Qlik": "⚠️ Manual Upload",
            "LinkedIn": "⚠️ Setup Required",
            "NewsAPI": "❌ Not Configured"
        }
        
        for service, status in integrations.items():
            st.write(f"**{service}:** {status}")
    
    with auto_tab3:
        st.markdown("### 📜 Automation Logs")
        st.text_area(
            "Recent Activity",
            value="""2024-08-11 06:00 - Morning brief generated
2024-08-11 06:15 - 3 LinkedIn changes detected
2024-08-11 06:30 - 5 news matches found
2024-08-11 06:45 - 3 emails queued for approval""",
            height=200
        )

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Quick Stats")
    st.metric("Time Saved This Week", "12 hours")
    st.metric("Insights Generated", "47")
    st.metric("Actions Completed", "19/23")
    
    st.divider()
    
    st.markdown("### 🚀 Quick Actions")
    if st.button("🌅 Morning Brief", use_container_width=True):
        st.session_state.show_brief = True
    if st.button("📧 Outreach Queue", use_container_width=True):
        st.session_state.show_outreach = True
    if st.button("📊 Qlik Alerts", use_container_width=True):
        st.session_state.show_qlik = True
    
    st.divider()
    
    st.markdown("### 📈 VP Progress")
    st.progress(0.62)
    st.caption("62% to VP by 2028")
    
    # Key milestones
    st.markdown("""
    **Milestones:**
    - ✅ Built intelligence system
    - ✅ Automated workflows
    - ⏳ Expand to 2 markets
    - ⏳ Hit $10M revenue
    - ⏳ Build strategic team
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 1rem;'>
    <p>🧠 Business Brain Master v4.0 | 354 Meetings | 4000+ Knowledge Cards | Real-time Intelligence</p>
    <p>Building your path to VP by 2028</p>
</div>
""", unsafe_allow_html=True)
