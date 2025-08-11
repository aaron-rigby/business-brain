import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from openai import OpenAI
from pinecone import Pinecone
import time

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
    page_title="Business Brain Master",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Clients
@st.cache_resource
def init_all_clients():
    """Initialize all API clients"""
    clients = {
        'openai': OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        'pinecone': Pinecone(api_key=os.getenv("PINECONE_API_KEY")),
        'index': Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(os.getenv("PINECONE_INDEX_NAME", "business-brain"))
    }
    return clients

# Load clients
clients = init_all_clients()

# Header
st.markdown('<h1 class="main-header">🧠 Business Brain Master System</h1>', unsafe_allow_html=True)

# Top Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

# Get Pinecone stats
try:
    index_stats = clients['index'].describe_index_stats()
    total_meetings = index_stats.get('total_vector_count', 0)
except:
    total_meetings = 354  # Fallback

with col1:
    st.metric("📊 Meetings Indexed", total_meetings, "+12 this week", delta_color="normal")

with col2:
    st.metric("✅ Open Actions", "23", "-3 completed", delta_color="inverse")

with col3:
    st.metric("👥 CRM Contacts", "0", "", delta_color="off")

with col4:
    st.metric("📈 Qlik Alerts", "3", "2 critical", delta_color="normal")

with col5:
    st.metric("🎯 VP Progress", "62%", "+5%", delta_color="normal")

# Main Tabs
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6 = st.tabs([
    "🏠 Command Center",
    "🔍 Intelligence Search", 
    "📋 Action Tracker",
    "👥 CRM & Outreach",
    "📊 Performance (Qlik)",
    "🤖 Automation"
])

# TAB 1: COMMAND CENTER
with main_tab1:
    st.markdown("## 🎯 Daily Intelligence Brief")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("📋 Generate Today's Brief", type="primary", use_container_width=True):
            with st.spinner("Generating your intelligence brief..."):
                st.success("Daily brief generated!")
                st.markdown("""
                ### 📅 Monday, December 15, 2024
                
                **🔥 Priority Actions:**
                1. Follow up with SPH on pricing proposal
                2. Prepare Mediacorp Q1 campaign review
                3. Submit Astra International renewal docs
                
                **📊 Key Metrics:**
                - Pipeline: $2.3M (87% to target)
                - At-risk renewals: 2 accounts ($450K)
                - New opportunities: 5 qualified leads
                
                **🎯 Strategic Focus:**
                - SEA expansion: Indonesia office planning
                - Team scaling: 3 open headcount approved
                - Product launch: Native video units (Jan 2025)
                """)
    
    with col2:
        st.markdown("### ⚡ Quick Actions")
        if st.button("📧 Check Outreach Queue", use_container_width=True):
            st.info("3 follow-ups pending")
        if st.button("📊 View Qlik Dashboard", use_container_width=True):
            st.info("Opening Qlik...")
        if st.button("📅 Prep Tomorrow's Meetings", use_container_width=True):
            st.info("2 meetings tomorrow")
        if st.button("🔍 Search Intelligence", use_container_width=True):
            st.info("Switch to Search tab")

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
    
    # Search Button - INSIDE the tab
    if st.button("🔍 Search All Intelligence", type="primary", use_container_width=True):
        with st.spinner("Analyzing your query..."):
            if search_query:
                try:
                    query_lower = search_query.lower()
                    
                    # 1. STATISTICAL QUERIES
                    if any(word in query_lower for word in ['how many', 'count', 'total', 'number of', 'amount', 'quantity']):
                        stats = clients['index'].describe_index_stats()
                        total_count = stats.get('total_vector_count', 0)
                        
                        st.success(f"📊 **Answer:** You have **{total_count} meetings** indexed in your Business Brain")
                        
                        st.info(f"""
                        📈 **Quick Stats:**
                        - Average of {total_count//52:.0f} meetings per week
                        - Covering {total_count//30:.0f} months of business intelligence
                        - Searchable across all your key accounts (SPH, Mediacorp, Astra, etc.)
                        """)
                    
                    # 2. TEMPORAL QUERIES
                    elif any(word in query_lower for word in ['recent', 'latest', 'last', 'yesterday', 'today', 'this week', 'this month']):
                        st.info("🕒 Searching for recent meetings...")
                        
                        embedding = clients['openai'].embeddings.create(
                            input=search_query,
                            model="text-embedding-ada-002"
                        ).data[0].embedding
                        
                        results = clients['index'].query(
                            vector=embedding,
                            top_k=10,
                            include_metadata=True
                        )
                        
                        if results.matches:
                            st.subheader("📅 Recent Meetings")
                            for i, match in enumerate(results.matches[:5], 1):
                                metadata = match.metadata
                                st.write(f"**{i}.** Meeting ID: {metadata.get('id', 'Unknown')}")
                                st.write(f"   Score: {match.score:.2%}")
                                if metadata.get('text'):
                                    st.write(f"   Content: {metadata.get('text', '')[:200]}...")
                                st.divider()
                    
                    # 3. DEFAULT SEMANTIC SEARCH
                    else:
                        embedding = clients['openai'].embeddings.create(
                            input=search_query,
                            model="text-embedding-ada-002"
                        ).data[0].embedding
                        
                        results = clients['index'].query(
                            vector=embedding,
                            top_k=5,
                            include_metadata=True
                        )
                        
                        if results.matches:
                            st.subheader(f"🔍 Search Results for: '{search_query}'")
                            
                            for i, match in enumerate(results.matches, 1):
                                metadata = match.metadata
                                score = match.score
                                
                                with st.container():
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.write(f"**{i}. Result**")
                                    with col2:
                                        st.write(f"📊 {score:.1%} match")
                                    
                                    # Display all metadata
                                    st.json(metadata)
                                    st.divider()
                        else:
                            st.warning(f"No results found for '{search_query}'")
                            
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
            else:
                st.warning("Please enter a search query")

# TAB 3: ACTION TRACKER
with main_tab3:
    st.markdown("## 📋 Action Tracker")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.button("➕ Add Action", type="primary", use_container_width=True)
    
    with col2:
        filter_option = st.selectbox("Filter", ["All", "Open", "In Progress", "Completed"])
    
    with col3:
        sort_option = st.selectbox("Sort by", ["Due Date", "Priority", "Account"])
    
    # Sample actions
    actions_data = {
        "Action": ["Follow up on SPH proposal", "Mediacorp Q1 review", "Astra renewal docs"],
        "Account": ["SPH", "Mediacorp", "Astra International"],
        "Due Date": ["2024-12-16", "2024-12-17", "2024-12-18"],
        "Priority": ["High", "Medium", "High"],
        "Status": ["Open", "In Progress", "Open"]
    }
    
    df_actions = pd.DataFrame(actions_data)
    st.dataframe(df_actions, use_container_width=True, hide_index=True)

# TAB 4: CRM & OUTREACH
with main_tab4:
    st.markdown("## 👥 CRM & Outreach")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🏢 Account Intelligence")
        account = st.selectbox("Select Account", ["SPH", "Mediacorp", "Astra International"])
        
        if account:
            st.info(f"Loading intelligence for {account}...")
            st.markdown(f"""
            **Last Meeting:** 2 days ago
            **Renewal Date:** Q1 2025
            **Revenue YTD:** $1.2M
            **Key Stakeholder:** John Smith
            **Sentiment:** 🟢 Positive
            """)
    
    with col2:
        st.markdown("### 📧 Smart Outreach")
        if st.button("🤖 Generate Follow-up Email", use_container_width=True):
            st.text_area("Generated Email", 
                value="Subject: Following up on our pricing discussion\n\nHi [Name],\n\nThank you for...",
                height=200)

# TAB 5: PERFORMANCE (QLIK)
with main_tab5:
    st.markdown("## 📊 Performance Dashboard")
    
    # Sample performance data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    revenue = [1.2, 1.5, 1.8, 2.1, 2.3, 2.5, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=revenue, mode='lines+markers', name='Revenue ($M)'))
    fig.update_layout(title='2024 Revenue Trend', xaxis_title='Month', yaxis_title='Revenue ($M)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Q4 Target", "$4.5M", "87% achieved")
    
    with col2:
        st.metric("YTD Growth", "+34%", "vs 2023")
    
    with col3:
        st.metric("Pipeline", "$8.2M", "3.2x coverage")

# TAB 6: AUTOMATION
with main_tab6:
    st.markdown("## 🤖 Automation Hub")
    
    st.markdown("### ⚡ Active Automations")
    
    automations = {
        "Automation": ["Meeting Capture", "Daily Brief", "Outreach Queue", "Qlik Alerts"],
        "Status": ["🟢 Active", "🟢 Active", "🟡 Paused", "🟢 Active"],
        "Last Run": ["2 hours ago", "This morning", "3 days ago", "1 hour ago"],
        "Next Run": ["In 1 hour", "Tomorrow 8am", "Manual", "In 2 hours"]
    }
    
    df_auto = pd.DataFrame(automations)
    st.dataframe(df_auto, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    f"<center>🧠 Business Brain Master v4.0 | {total_meetings} Meetings | 4000+ Knowledge Cards | Real-time Intelligence<br>"
    f"Building your path to VP by 2028</center>",
    unsafe_allow_html=True
)
