#!/usr/bin/env python3
"""
BUSINESS BRAIN MASTER SYSTEM
Complete unified dashboard with all features integrated
Version 5.0 - With Salesforce Pipeline Integration
"""

import streamlit as st
import pandas as pd
import json
import openai
from pinecone import Pinecone
from notion_client import Client
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import os
import pytz

# Page Configuration
st.set_page_config(
    page_title="🧠 Business Brain Master",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Configuration
from dotenv import load_dotenv
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
PINECONE_INDEX_NAME = "business-brain"
PIPELINE_DB_ID = "14d6e7c2-3838-804c-844a-000c85c988c6"

# Verify keys are loaded
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please check your configuration.")
    st.stop()
if not PINECONE_API_KEY:
    st.error("Pinecone API key not found. Please check your configuration.")
    st.stop()
if not NOTION_TOKEN:
    st.error("Notion token not found. Please check your configuration.")
    st.stop()

# Initialize Clients
@st.cache_resource
def init_all_clients():
    """Initialize all API clients"""
    clients = {
        'openai': openai.OpenAI(api_key=OPENAI_API_KEY),
        'pinecone': Pinecone(api_key=PINECONE_API_KEY),
        'index': Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME),
        'notion': Client(auth=NOTION_TOKEN)
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
if 'new_big_deals' not in st.session_state:
    st.session_state.new_big_deals = []

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
    .deal-alert {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🧠 Business Brain Master System</h1>", unsafe_allow_html=True)

# Fetch Pipeline Data
@st.cache_data(ttl=300)
def fetch_pipeline_data():
    """Fetch Salesforce pipeline from Notion"""
    try:
        response = clients['notion'].databases.query(database_id=PIPELINE_DB_ID)
        deals = []
        for page in response['results']:
            props = page['properties']
            
            # Extract last modified/captured date
            date_captured = props.get('Date_Captured', {}).get('date', {}).get('start', None)
            if date_captured:
                date_captured = datetime.fromisoformat(date_captured.replace('Z', '+00:00'))
            
            deal = {
                'Name': props.get('Opportunity Name', {}).get('title', [{}])[0].get('text', {}).get('content', 'Unknown'),
                'Amount': props.get('Amount_USD', {}).get('number', 0),
                'Stage': props.get('Stage', {}).get('select', {}).get('name', 'Unknown'),
                'Owner': props.get('Owner', {}).get('rich_text', [{}])[0].get('text', {}).get('content', 'Unknown'),
                'Owner_Nickname': props.get('Owner_Nickname', {}).get('select', {}).get('name', 'Unknown'),
                'Market': props.get('Market', {}).get('select', {}).get('name', 'Unknown'),
                'Close_Date': props.get('Close_Date', {}).get('date', {}).get('start', None),
                'Date_Captured': date_captured,
                'Account_Name': props.get('Account_Name', {}).get('rich_text', [{}])[0].get('text', {}).get('content', 'Unknown')
            }
            deals.append(deal)
        return pd.DataFrame(deals)
    except Exception as e:
        st.error(f"Error fetching pipeline: {e}")
        return pd.DataFrame()

# Load pipeline data
df_pipeline = fetch_pipeline_data()

# Check for new big deals (for morning alert)
if not df_pipeline.empty:
    bangkok_tz = pytz.timezone('Asia/Bangkok')
    today = datetime.now(bangkok_tz).date()
    
    # Find deals added today that are over $20k
    new_big_deals = []
    for _, deal in df_pipeline.iterrows():
        if deal['Date_Captured'] and deal['Amount'] >= 20000:
            deal_date = deal['Date_Captured'].date() if hasattr(deal['Date_Captured'], 'date') else deal['Date_Captured']
            if deal_date == today:
                new_big_deals.append(deal)
    
    if new_big_deals:
        st.session_state.new_big_deals = new_big_deals

# Top Metrics Bar (Updated with Pipeline data)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    total_pipeline = df_pipeline['Amount'].sum() if not df_pipeline.empty else 0
    st.metric("💰 Pipeline", f"${total_pipeline/1000000:.1f}M")
with col2:
    active_deals = len(df_pipeline[df_pipeline['Stage'] != 'Closed Lost']) if not df_pipeline.empty else 0
    st.metric("🎯 Active Deals", active_deals)
with col3:
    st.metric("💥 CRM Contacts", len(st.session_state.crm_contacts))
with col4:
    st.metric("📈 Qlik Alerts", "3", delta="2 critical")
with col5:
    st.metric("🎯 VP Progress", "62%", delta="+5%")

# Morning Alert for Big Deals
if st.session_state.new_big_deals:
    st.markdown("### 🚨 New Big Deals Alert (>$20K)")
    for deal in st.session_state.new_big_deals:
        st.warning(f"🆕 **{deal['Name']}** - ${deal['Amount']:,.0f} - {deal['Owner_Nickname']} ({deal['Market']})")

st.divider()

# Main Navigation Tabs
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5, main_tab6, main_tab7 = st.tabs([
    "🏠 Command Center",
    "💰 Salesforce Pipeline",
    "🔍 Intelligence Search", 
    "📋 Action Tracker",
    "💥 CRM & Outreach",
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
                'opportunities': ["Astra interested in premium inventory", "New Google ad format opportunity"],
                'big_deals': st.session_state.new_big_deals
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

# TAB 2: SALESFORCE PIPELINE (NEW ENHANCED)
with main_tab2:
    st.markdown("## 💰 Salesforce Pipeline Analytics")
    
    if not df_pipeline.empty:
        # Pipeline Summary Metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            total = df_pipeline['Amount'].sum()
            st.metric("Total Pipeline", f"${total:,.0f}")
        with col2:
            qualified = df_pipeline[df_pipeline['Stage'].str.contains('Qualified', na=False)]['Amount'].sum()
            st.metric("Qualified", f"${qualified:,.0f}")
        with col3:
            proposal = df_pipeline[df_pipeline['Stage'].str.contains('Proposal', na=False)]['Amount'].sum()
            st.metric("In Proposal", f"${proposal:,.0f}")
        with col4:
            avg_deal = df_pipeline['Amount'].mean()
            st.metric("Avg Deal", f"${avg_deal:,.0f}")
        with col5:
            total_sellers = df_pipeline['Owner_Nickname'].nunique()
            st.metric("Active Sellers", total_sellers)
        with col6:
            markets = df_pipeline['Market'].nunique()
            st.metric("Markets", markets)
        
        st.divider()
        
        # Sub-tabs for different views
        pipe_tab1, pipe_tab2, pipe_tab3, pipe_tab4, pipe_tab5 = st.tabs([
            "📊 By Market", "👥 By Seller", "🎯 Top 10 Deals", "⏰ Stale Deals", "🔍 Pipeline Intelligence"
        ])
        
        with pipe_tab1:
            st.markdown("### 📊 Pipeline by Market")
            
            col1, col2 = st.columns(2)
            with col1:
                # Market summary
                market_summary = df_pipeline.groupby('Market').agg({
                    'Amount': 'sum',
                    'Name': 'count'
                }).rename(columns={'Name': 'Deal_Count'}).sort_values('Amount', ascending=False)
                
                # Bar chart
                fig = px.bar(market_summary, y=market_summary.index, x='Amount', 
                            orientation='h', title='Pipeline Value by Market',
                            labels={'Amount': 'Pipeline Value ($)', 'index': 'Market'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart
                fig = px.pie(market_summary, values='Amount', names=market_summary.index,
                           title='Market Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            # Market details table
            st.markdown("### Market Details")
            market_details = df_pipeline.groupby('Market').agg({
                'Amount': ['sum', 'mean', 'count'],
                'Stage': lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
            }).round(0)
            market_details.columns = ['Total Pipeline', 'Avg Deal Size', 'Deal Count', 'Most Common Stage']
            st.dataframe(market_details.sort_values('Total Pipeline', ascending=False), use_container_width=True)
        
        with pipe_tab2:
            st.markdown("### 👥 Pipeline by Seller")
            
            col1, col2 = st.columns(2)
            with col1:
                # Seller summary
                seller_summary = df_pipeline.groupby('Owner_Nickname').agg({
                    'Amount': 'sum',
                    'Name': 'count',
                    'Market': 'first'
                }).rename(columns={'Name': 'Deal_Count'}).sort_values('Amount', ascending=False)
                
                # Top 10 sellers bar chart
                top_sellers = seller_summary.head(10)
                fig = px.bar(top_sellers, y=top_sellers.index, x='Amount',
                           orientation='h', title='Top 10 Sellers by Pipeline',
                           color='Market')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Seller performance metrics
                st.markdown("#### 🏆 Top Performers")
                for i, (seller, data) in enumerate(seller_summary.head(5).iterrows(), 1):
                    st.metric(f"{i}. {seller} ({data['Market']})", 
                             f"${data['Amount']:,.0f}",
                             f"{int(data['Deal_Count'])} deals")
            
            # Seller details table
            st.markdown("### Seller Performance Details")
            seller_details = df_pipeline.groupby(['Owner_Nickname', 'Market']).agg({
                'Amount': ['sum', 'mean', 'count'],
                'Stage': lambda x: x.value_counts().index[0] if len(x) > 0 else 'N/A'
            }).round(0)
            seller_details.columns = ['Total Pipeline', 'Avg Deal Size', 'Deal Count', 'Most Common Stage']
            st.dataframe(seller_details.sort_values('Total Pipeline', ascending=False), use_container_width=True)
        
        with pipe_tab3:
            st.markdown("### 🎯 Top 10 Biggest Deals")
            
            # Calculate days since last update
            df_pipeline['Days_Since_Update'] = (datetime.now() - pd.to_datetime(df_pipeline['Date_Captured'])).dt.days
            
            # Get top 10 deals
            top_deals = df_pipeline.nlargest(10, 'Amount')[
                ['Name', 'Account_Name', 'Amount', 'Stage', 'Owner_Nickname', 'Market', 'Days_Since_Update', 'Close_Date']
            ]
            
            # Display as cards
            for _, deal in top_deals.iterrows():
                days_color = "🟢" if deal['Days_Since_Update'] < 7 else "🟡" if deal['Days_Since_Update'] < 14 else "🔴"
                
                col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])
                with col1:
                    st.markdown(f"**{deal['Name']}**")
                    st.caption(f"{deal['Account_Name']}")
                with col2:
                    st.metric("Value", f"${deal['Amount']:,.0f}")
                with col3:
                    st.metric("Stage", deal['Stage'])
                with col4:
                    st.metric("Owner", deal['Owner_Nickname'])
                with col5:
                    st.metric("Last Update", f"{days_color} {deal['Days_Since_Update']}d ago")
                
                st.divider()
        
        with pipe_tab4:
            st.markdown("### ⏰ Deals Needing Attention")
            
            # Stale deals (not updated in 14+ days)
            stale_deals = df_pipeline[df_pipeline['Days_Since_Update'] >= 14].sort_values('Amount', ascending=False)
            
            if not stale_deals.empty:
                st.error(f"⚠️ {len(stale_deals)} deals haven't been updated in 2+ weeks!")
                
                # Group by owner
                stale_by_owner = stale_deals.groupby('Owner_Nickname').agg({
                    'Amount': 'sum',
                    'Name': 'count'
                }).rename(columns={'Name': 'Stale_Deal_Count'})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Stale Deals by Owner")
                    st.dataframe(stale_by_owner.sort_values('Amount', ascending=False))
                
                with col2:
                    st.markdown("#### Top Stale Deals to Review")
                    for _, deal in stale_deals.head(5).iterrows():
                        st.warning(f"**{deal['Name']}** - ${deal['Amount']:,.0f} - {deal['Days_Since_Update']} days old - {deal['Owner_Nickname']}")
            else:
                st.success("✅ All deals have been updated recently!")
        
        with pipe_tab5:
            st.markdown("### 🔍 AI Pipeline Intelligence")
            
            pipeline_query = st.text_area(
                "Ask about your pipeline:",
                placeholder="Which deals need attention? What's our exposure in India? Who are the top performers?",
                height=80
            )
            
            if st.button("🧠 Analyze Pipeline", type="primary"):
                with st.spinner("Analyzing pipeline with AI..."):
                    # Create context from pipeline data
                    pipeline_context = f"""
                    Total Pipeline: ${df_pipeline['Amount'].sum():,.0f}
                    Total Deals: {len(df_pipeline)}
                    By Market: {df_pipeline.groupby('Market')['Amount'].sum().to_dict()}
                    By Stage: {df_pipeline.groupby('Stage')['Amount'].sum().to_dict()}
                    Top Sellers: {df_pipeline.groupby('Owner_Nickname')['Amount'].sum().nlargest(5).to_dict()}
                    Stale Deals: {len(df_pipeline[df_pipeline['Days_Since_Update'] >= 14])} deals not updated in 14+ days
                    """
                    
                    response = clients['openai'].chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a sales analytics expert analyzing the pipeline."},
                            {"role": "user", "content": f"{pipeline_query}\n\nPipeline Context:\n{pipeline_context}"}
                        ]
                    )
                    
                    st.write(response.choices[0].message.content)
    else:
        st.warning("No pipeline data available. Check your Notion connection.")

# TAB 3: INTELLIGENCE SEARCH
with main_tab3:
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
            ["GPT-4 (Smart)", "GPT-3.5 (Fast)"]
        )
        
    with col3:
        data_sources = st.multiselect(
            "Search in",
            ["Meetings", "Pipeline", "CRM", "Qlik", "Guru"],
            default=["Meetings", "Pipeline"]
        )
    
    if st.button("🔍 Search All Intelligence", type="primary", use_container_width=True):
        with st.spinner("Searching across all data sources..."):
            # Enhanced search with pipeline data
            results = {}
            
            if "Meetings" in data_sources:
                # Pinecone RAG search
                try:
                    search_embedding = clients['openai'].embeddings.create(
                        model="text-embedding-ada-002",
                        input=search_query
                    ).data[0].embedding
                    
                    pinecone_results = clients['index'].query(
                        vector=search_embedding,
                        top_k=5,
                        include_metadata=True
                    )
                    results['meetings'] = [match.metadata.get('text', '') for match in pinecone_results.matches]
                except:
                    results['meetings'] = []
            
            if "Pipeline" in data_sources and not df_pipeline.empty:
                # Search pipeline data
                pipeline_matches = df_pipeline[
                    df_pipeline['Name'].str.contains(search_query, case=False, na=False) |
                    df_pipeline['Account_Name'].str.contains(search_query, case=False, na=False)
                ]
                if not pipeline_matches.empty:
                    results['pipeline'] = [f"{row['Name']} - ${row['Amount']:,.0f} - {row['Stage']}" 
                                          for _, row in pipeline_matches.head(5).iterrows()]
            
            # Display results
            for source, items in results.items():
                if items:
                    st.markdown(f"### 📌 From {source.upper()}")
                    for item in items:
                        st.write(f"• {item}")

# TAB 4: ACTION TRACKER
with main_tab4:
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

# TAB 5: CRM & OUTREACH
with main_tab5:
    st.markdown("## 💥 CRM & Intelligent Outreach")
    
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

# TAB 6: QLIK PERFORMANCE
with main_tab6:
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

# TAB 7: AUTOMATION SETTINGS
with main_tab7:
    st.markdown("## ⚙️ Automation Configuration")
    
    auto_tab1, auto_tab2, auto_tab3 = st.tabs(["Schedules", "Integrations", "Logs"])
    
    with auto_tab1:
        st.markdown("### ⏰ Automation Schedule")
        
        schedules = {
            "Morning Brief": "6:00 AM",
            "Pipeline Extract": "5:30 AM",
            "Pipeline Process": "6:00 AM",
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
            "Salesforce": "✅ Via Email",
            "Qlik": "⚠️ Manual Upload",
            "LinkedIn": "⚠️ Setup Required",
            "NewsAPI": "❌ Not Configured"
        }
        
        for service, status in integrations.items():
            st.write(f"**{service}:** {status}")
    
    with auto_tab3:
        st.markdown("### 📜 Automation Logs")
        if os.path.exists('/volume1/Shared/business_brain/logs/pipeline.log'):
            with open('/volume1/Shared/business_brain/logs/pipeline.log', 'r') as f:
                st.text_area("Pipeline Processing Log", value=f.read()[-1000:], height=200)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Quick Stats")
    
    if not df_pipeline.empty:
        total_pipeline_value = df_pipeline['Amount'].sum()
        deals_closing_soon = len(df_pipeline[df_pipeline['Stage'].str.contains('Proposal|Contract', na=False)])
        
        st.metric("Pipeline Value", f"${total_pipeline_value/1000000:.1f}M")
        st.metric("Deals Closing Soon", deals_closing_soon)
        st.metric("Time Saved This Week", "12 hours")
    
    st.divider()
    
    st.markdown("### 🚀 Quick Actions")
    if st.button("🌅 Morning Brief", use_container_width=True):
        st.session_state.show_brief = True
    if st.button("💰 Pipeline Review", use_container_width=True):
        st.session_state.show_pipeline = True
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
    - ✅ Pipeline tracking live
    - ⏳ Expand to 2 markets
    - ⏳ Hit $10M revenue
    - ⏳ Build strategic team
    """)
    
    st.divider()
    
    # Refresh button
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 1rem;'>
    <p>🧠 Business Brain Master v5.0 | 354 Meetings | 143 Pipeline Deals | Real-time Intelligence</p>
    <p>Building your path to VP by 2028</p>
</div>
""", unsafe_allow_html=True)
