import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from notion_client import Client
import os
import traceback

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Business Brain Master System",
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
            "OPENAI_API_KEY": "your-openai-key",  # Add if needed
            "PINECONE_API_KEY": "your-pinecone-key",  # Add if needed
        }
    else:
        try:
            return {
                "NOTION_TOKEN": st.secrets.get("NOTION_TOKEN"),
                "PIPELINE_DB_ID": st.secrets.get("PIPELINE_DB_ID"),
                "OPENAI_API_KEY": st.secrets.get("OPENAI_API_KEY", ""),
                "PINECONE_API_KEY": st.secrets.get("PINECONE_API_KEY", ""),
            }
        except Exception as e:
            st.error(f"Failed to load secrets: {e}")
            return None

# ============================================
# NOTION CONNECTION
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
        # Test the connection
        if DEBUG_MODE:
            st.sidebar.success("✅ Notion client initialized")
        return client
    except Exception as e:
        st.error(f"Failed to initialize Notion client: {e}")
        if DEBUG_MODE:
            st.sidebar.error(f"Error details: {traceback.format_exc()}")
        return None

# ============================================
# PIPELINE DATA FETCHING
# ============================================
def get_pipeline_data():
    """Fetch pipeline data from Notion with extensive error handling"""
    notion = get_notion_client()
    if not notion:
        return pd.DataFrame()
    
    creds = get_credentials()
    database_id = creds.get("PIPELINE_DB_ID")
    
    if not database_id:
        st.error("No database ID configured")
        return pd.DataFrame()
    
    try:
        # Query the database
        if DEBUG_MODE:
            st.write(f"📊 Querying database: {database_id}")
        
        response = notion.databases.query(
            database_id=database_id,
            page_size=100
        )
        
        if DEBUG_MODE:
            st.write(f"✅ Got response with {len(response.get('results', []))} items")
            if response.get('results'):
                st.write("First item properties:", list(response['results'][0].get('properties', {}).keys()))
        
        if not response or 'results' not in response:
            st.warning("No data returned from Notion")
            return pd.DataFrame()
        
        results = response.get('results', [])
        if not results:
            st.info("Database is connected but contains no deals")
            return pd.DataFrame()
        
        # Process the results
        deals = []
        for idx, item in enumerate(results):
            try:
                props = item.get('properties', {})
                
                # Debug first item structure
                if DEBUG_MODE and idx == 0:
                    st.write("Sample deal structure:")
                    st.json(props)
                
                # Try different possible field names for amount
                amount = 0
                amount_fields = ['Amount_USD', 'Amount', 'Value', 'Deal_Amount', 'Pipeline_Amount']
                for field in amount_fields:
                    if field in props:
                        amount_prop = props[field]
                        if 'number' in amount_prop:
                            amount = amount_prop.get('number', 0)
                        elif 'formula' in amount_prop:
                            amount = amount_prop.get('formula', {}).get('number', 0)
                        if amount > 0:
                            break
                
                # Try different possible field names for account
                account = "Unknown"
                account_fields = ['Account_Name', 'Account', 'Company', 'Customer', 'Client']
                for field in account_fields:
                    if field in props:
                        account_prop = props[field]
                        if 'title' in account_prop:
                            titles = account_prop.get('title', [])
                            if titles and len(titles) > 0:
                                account = titles[0].get('plain_text', 'Unknown')
                                break
                        elif 'rich_text' in account_prop:
                            texts = account_prop.get('rich_text', [])
                            if texts and len(texts) > 0:
                                account = texts[0].get('plain_text', 'Unknown')
                                break
                
                # Try different possible field names for stage
                stage = "Unknown"
                stage_fields = ['Stage', 'Status', 'Pipeline_Stage', 'Deal_Stage']
                for field in stage_fields:
                    if field in props:
                        stage_prop = props[field]
                        if 'select' in stage_prop:
                            stage = stage_prop.get('select', {}).get('name', 'Unknown')
                            break
                        elif 'status' in stage_prop:
                            stage = stage_prop.get('status', {}).get('name', 'Unknown')
                            break
                
                # Try to get close date
                close_date = None
                date_fields = ['Close_Date', 'CloseDate', 'Expected_Close', 'Date']
                for field in date_fields:
                    if field in props:
                        date_prop = props[field]
                        if 'date' in date_prop:
                            date_val = date_prop.get('date', {})
                            if date_val and 'start' in date_val:
                                close_date = date_val['start']
                                break
                
                deals.append({
                    'Account': account,
                    'Amount': amount,
                    'Stage': stage,
                    'Close_Date': close_date,
                })
                
            except Exception as e:
                if DEBUG_MODE:
                    st.write(f"⚠️ Error processing deal {idx}: {e}")
                continue
        
        df = pd.DataFrame(deals)
        
        if DEBUG_MODE:
            st.write(f"✅ Processed {len(df)} deals")
            st.write("DataFrame shape:", df.shape)
            st.write("DataFrame columns:", df.columns.tolist())
            st.write("Total pipeline value:", df['Amount'].sum())
        
        return df
        
    except Exception as e:
        st.error(f"Failed to fetch pipeline data: {e}")
        if DEBUG_MODE:
            st.write("Full error:", traceback.format_exc())
        return pd.DataFrame()

# ============================================
# METRICS CALCULATION
# ============================================
def calculate_metrics(df):
    """Calculate key metrics from pipeline data"""
    if df.empty:
        return {
            'total_pipeline': 0,
            'num_deals': 0,
            'avg_deal_size': 0,
            'by_stage': pd.DataFrame(),
        }
    
    total = df['Amount'].sum()
    num_deals = len(df)
    avg_size = total / num_deals if num_deals > 0 else 0
    
    # Group by stage
    by_stage = df.groupby('Stage')['Amount'].agg(['sum', 'count']).reset_index()
    by_stage.columns = ['Stage', 'Total', 'Count']
    
    return {
        'total_pipeline': total,
        'num_deals': num_deals,
        'avg_deal_size': avg_size,
        'by_stage': by_stage,
    }

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    # Header
    st.title("🧠 Business Brain Master System")
    
    # Error message placeholder
    error_placeholder = st.container()
    
    # Fetch pipeline data
    with st.spinner("Loading pipeline data..."):
        df = get_pipeline_data()
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Pipeline",
            f"${metrics['total_pipeline']/1e6:.1f}M" if metrics['total_pipeline'] > 0 else "$0.0M",
            delta=None
        )
    
    with col2:
        st.metric(
            "🎯 Active Deals",
            metrics['num_deals'],
            delta=None
        )
    
    with col3:
        st.metric(
            "📊 Avg Deal Size",
            f"${metrics['avg_deal_size']/1e3:.0f}K" if metrics['avg_deal_size'] > 0 else "$0K",
            delta=None
        )
    
    with col4:
        st.metric(
            "📈 Qlik Alerts",
            "3",
            delta="2 critical"
        )
    
    with col5:
        st.metric(
            "💜 VP Progress",
            "62%",
            delta="+5%"
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 Command Center",
        "💰 Salesforce Pipeline",
        "🔍 Intelligence Search",
        "📊 Action Tracker",
        "👥 CRM & Outreach",
        "📈 Performance (Qlik)",
        "🤖 Automation"
    ])
    
    with tab1:
        st.header("Daily Intelligence Brief")
        if st.button("🎯 Generate Today's Brief"):
            st.info("Generating intelligence brief...")
        
        # Quick Actions
        st.subheader("🚀 Quick Actions")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button("✅ Check Outreach Queue")
        with col2:
            st.button("📊 View Qlik Dashboard")
        with col3:
            st.button("📅 Prep Tomorrow's Meetings")
        with col4:
            st.button("🔍 Search Intelligence")
    
    with tab2:
        st.header("Salesforce Pipeline Analysis")
        
        if not df.empty:
            # Pipeline by Stage
            if not metrics['by_stage'].empty:
                fig = px.bar(
                    metrics['by_stage'],
                    x='Stage',
                    y='Total',
                    title='Pipeline by Stage',
                    labels={'Total': 'Amount ($)', 'Stage': 'Sales Stage'},
                    color='Total',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Deal details
            st.subheader("Deal Details")
            st.dataframe(
                df.sort_values('Amount', ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No pipeline data available. Check connection settings or use Debug Mode.")
    
    with tab3:
        st.header("Intelligence Search")
        search_query = st.text_input("🔍 Search across all data sources...")
        if search_query:
            st.info(f"Searching for: {search_query}")
    
    with tab4:
        st.header("Action Tracker")
        st.info("Action items will appear here")
    
    with tab5:
        st.header("CRM & Outreach")
        st.info("CRM integration coming soon")
    
    with tab6:
        st.header("Performance (Qlik)")
        st.info("Qlik dashboards will be embedded here")
    
    with tab7:
        st.header("Automation Hub")
        st.info("Workflow automation controls")
    
    # Sidebar - VP Progress
    st.sidebar.header("📈 VP Progress")
    st.sidebar.progress(0.62)
    st.sidebar.caption("62% to VP by 2028")
    
    st.sidebar.subheader("Milestones:")
    st.sidebar.write("✅ Built intelligence system")
    st.sidebar.write("✅ Automated workflows")
    st.sidebar.write("✅ Pipeline tracking live")
    st.sidebar.write("🏃 Expand to 2 markets")
    st.sidebar.write("🏃 Hit $10M revenue")
    
    # Debug Info
    if DEBUG_MODE:
        st.sidebar.header("🔧 Debug Info")
        creds = get_credentials()
        st.sidebar.write("Token present:", bool(creds and creds.get("NOTION_TOKEN")))
        st.sidebar.write("DB ID:", creds.get("PIPELINE_DB_ID", "Not set")[:10] + "..." if creds else "No creds")
        st.sidebar.write("DataFrame shape:", df.shape if not df.empty else "Empty")

if __name__ == "__main__":
    main()
