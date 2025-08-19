import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from notion_client import Client
import openai
from pinecone import Pinecone, ServerlessSpec
import os
import traceback
import time

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
            "OPENAI_API_KEY": st.secrets.get("OPENAI_API_KEY", ""),
            "PINECONE_API_KEY": st.secrets.get("PINECONE_API_KEY", ""),
            "PINECONE_INDEX_NAME": st.secrets.get("PINECONE_INDEX_NAME", "business-brain"),
        }
    else:
        try:
            return {
                "NOTION_TOKEN": st.secrets.get("NOTION_TOKEN"),
                "PIPELINE_DB_ID": st.secrets.get("PIPELINE_DB_ID"),
                "OPENAI_API_KEY": st.secrets.get("OPENAI_API_KEY", ""),
                "PINECONE_API_KEY": st.secrets.get("PINECONE_API_KEY", ""),
                "PINECONE_INDEX_NAME": st.secrets.get("PINECONE_INDEX_NAME", "business-brain"),
            }
        except Exception as e:
            st.error(f"Failed to load secrets: {e}")
            return None

# ============================================
# NOTION CONNECTION
# ============================================
@st.cache_data(ttl=300)  # Cache for 5 minutes
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
            st.sidebar.error(f"Error details: {traceback.format_exc()}")
        return None

# ============================================
# PIPELINE DATA FETCHING - WITH PAGINATION
# ============================================
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_pipeline_data():
    """Fetch ALL pipeline data from Notion with pagination"""
    notion = get_notion_client()
    if not notion:
        return pd.DataFrame()
    
    creds = get_credentials()
    database_id = creds.get("PIPELINE_DB_ID")
    
    if not database_id:
        st.error("No database ID configured")
        return pd.DataFrame()
    
    try:
        # Query the database - GET ALL RECORDS WITH PAGINATION
        if DEBUG_MODE:
            st.write(f"📊 Querying database: {database_id}")
        
        # Fetch ALL pages, not just first 100
        all_results = []
        has_more = True
        next_cursor = None
        page_count = 0
        
        while has_more:
            if DEBUG_MODE:
                st.write(f"Fetching page {page_count + 1}...")
            
            response = notion.databases.query(
                database_id=database_id,
                page_size=100,
                start_cursor=next_cursor
            )
            
            all_results.extend(response.get('results', []))
            has_more = response.get('has_more', False)
            next_cursor = response.get('next_cursor', None)
            page_count += 1
        
        if DEBUG_MODE:
            st.write(f"✅ Got {len(all_results)} total items across {page_count} pages")
            if all_results:
                st.write("First item properties:", list(all_results[0].get('properties', {}).keys()))
        
        if not all_results:
            st.info("Database is connected but contains no deals")
            return pd.DataFrame()
        
        # Process ALL results
        deals = []
        for idx, item in enumerate(all_results):
            try:
                props = item.get('properties', {})
                
                # Debug first item structure
                if DEBUG_MODE and idx == 0:
                    st.write("Sample deal structure:")
                    st.json(props)
                
                # Get amount
                amount = 0
                amount_prop = props.get('Amount_USD', {})
                if 'number' in amount_prop:
                    amount = amount_prop.get('number', 0)
                
                # Get account name
                account = "Unknown"
                account_prop = props.get('Account_Name', {})
                if 'rich_text' in account_prop:
                    texts = account_prop.get('rich_text', [])
                    if texts and len(texts) > 0:
                        account = texts[0].get('plain_text', 'Unknown')
                
                # Get stage
                stage = "Unknown"
                stage_prop = props.get('Stage', {})
                if 'select' in stage_prop and stage_prop['select']:
                    stage = stage_prop.get('select', {}).get('name', 'Unknown')
                
                # Get close date
                close_date = None
                close_date_prop = props.get('Close_Date', {})
                if 'date' in close_date_prop and close_date_prop['date']:
                    date_val = close_date_prop.get('date', {})
                    if date_val and 'start' in date_val:
                        close_date = date_val['start']
                
                # Get opportunity name
                opp_name = "Unknown"
                opp_prop = props.get('Opportunity Name', {})
                if 'title' in opp_prop:
                    titles = opp_prop.get('title', [])
                    if titles and len(titles) > 0:
                        opp_name = titles[0].get('plain_text', 'Unknown')
                
                # Get owner
                owner = "Unknown"
                owner_prop = props.get('Owner', {})
                if 'rich_text' in owner_prop:
                    texts = owner_prop.get('rich_text', [])
                    if texts and len(texts) > 0:
                        owner = texts[0].get('plain_text', 'Unknown')
                
                deals.append({
                    'Opportunity': opp_name,
                    'Account': account,
                    'Amount': amount,
                    'Stage': stage,
                    'Close_Date': close_date,
                    'Owner': owner,
                })
                
            except Exception as e:
                if DEBUG_MODE:
                    st.write(f"⚠️ Error processing deal {idx}: {e}")
                continue
        
        df = pd.DataFrame(deals)
        
        if DEBUG_MODE:
            st.write(f"✅ Processed {len(df)} deals")
            st.write("DataFrame shape:", df.shape)
            st.write("Total pipeline value: ${:,.2f}".format(df['Amount'].sum()))
            st.write("Sample data:")
            st.dataframe(df.head())
        
        return df
        
    except Exception as e:
        st.error(f"Failed to fetch pipeline data: {e}")
        if DEBUG_MODE:
            st.write("Full error:", traceback.format_exc())
        return pd.DataFrame()

# ============================================
# INTELLIGENCE SEARCH (PINECONE + OPENAI)
# ============================================
@st.cache_resource
def init_pinecone():
    """Initialize Pinecone connection"""
    creds = get_credentials()
    if not creds or not creds.get("PINECONE_API_KEY"):
        return None
    
    try:
        pc = Pinecone(api_key=creds["PINECONE_API_KEY"])
        index_name = creds.get("PINECONE_INDEX_NAME", "business-brain")
        
        # Check if index exists
        if index_name in pc.list_indexes().names():
            return pc.Index(index_name)
        else:
            st.warning(f"Pinecone index '{index_name}' not found")
            return None
    except Exception as e:
        if DEBUG_MODE:
            st.error(f"Failed to initialize Pinecone: {e}")
        return None

def search_intelligence(query, k=5):
    """Search across all data sources using Pinecone"""
    creds = get_credentials()
    
    # Check for OpenAI API key
    if not creds or not creds.get("OPENAI_API_KEY"):
        st.warning("OpenAI API key not configured. Add it to Streamlit secrets to enable intelligence search.")
        return []
    
    # Initialize OpenAI
    openai.api_key = creds["OPENAI_API_KEY"]
    
    # Get Pinecone index
    index = init_pinecone()
    if not index:
        st.info("Intelligence search requires Pinecone setup. Using basic search instead.")
        # Fallback to basic DataFrame search
        df = get_pipeline_data()
        if not df.empty and query:
            # Simple text search in pipeline data
            mask = df.apply(lambda row: query.lower() in str(row).lower(), axis=1)
            results = df[mask]
            return results.to_dict('records') if not results.empty else []
        return []
    
    try:
        # Generate embedding for query
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = response['data'][0]['embedding']
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        return results['matches'] if results else []
        
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

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
            'by_owner': pd.DataFrame(),
        }
    
    total = df['Amount'].sum()
    num_deals = len(df)
    avg_size = total / num_deals if num_deals > 0 else 0
    
    # Group by stage
    by_stage = df.groupby('Stage')['Amount'].agg(['sum', 'count']).reset_index()
    by_stage.columns = ['Stage', 'Total', 'Count']
    by_stage = by_stage.sort_values('Total', ascending=False)
    
    # Group by owner
    by_owner = df.groupby('Owner')['Amount'].agg(['sum', 'count']).reset_index()
    by_owner.columns = ['Owner', 'Total', 'Count']
    by_owner = by_owner.sort_values('Total', ascending=False).head(10)
    
    return {
        'total_pipeline': total,
        'num_deals': num_deals,
        'avg_deal_size': avg_size,
        'by_stage': by_stage,
        'by_owner': by_owner,
    }

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    # Header
    st.title("🧠 Business Brain Master System")
    
    # Fetch pipeline data
    with st.spinner("Loading pipeline data..."):
        df = get_pipeline_data()
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Display main metrics (only real data)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "💰 Total Pipeline",
            "${:,.0f}".format(metrics['total_pipeline']) if metrics['total_pipeline'] > 0 else "$0",
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
            "${:,.0f}".format(metrics['avg_deal_size']) if metrics['avg_deal_size'] > 0 else "$0",
            delta=None
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Command Center",
        "💰 Salesforce Pipeline",
        "🔍 Intelligence Search",
        "📊 Pipeline Analytics",
        "🤖 Automation Status"
    ])
    
    with tab1:
        st.header("Daily Intelligence Brief")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Pipeline Summary")
            if not metrics['by_stage'].empty:
                # Show stage breakdown
                for _, row in metrics['by_stage'].iterrows():
                    st.write(f"**{row['Stage']}**: ${row['Total']:,.0f} ({row['Count']} deals)")
        
        with col2:
            st.subheader("👥 Top Performers")
            if not metrics['by_owner'].empty:
                for _, row in metrics['by_owner'].head(5).iterrows():
                    st.write(f"**{row['Owner']}**: ${row['Total']:,.0f} ({row['Count']} deals)")
        
        # Quick Actions
        st.subheader("🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Refresh Pipeline Data"):
                st.cache_data.clear()
                st.rerun()
        with col2:
            st.button("📧 Check Salesforce Email")
        with col3:
            st.button("📅 Today's Meetings")
    
    with tab2:
        st.header("Salesforce Pipeline Analysis")
        
        if not df.empty:
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                stage_filter = st.multiselect(
                    "Filter by Stage",
                    options=df['Stage'].unique(),
                    default=df['Stage'].unique()
                )
            
            with col2:
                owner_filter = st.multiselect(
                    "Filter by Owner",
                    options=df['Owner'].unique(),
                    default=df['Owner'].unique()
                )
            
            with col3:
                min_amount = st.number_input(
                    "Min Deal Size",
                    min_value=0,
                    max_value=int(df['Amount'].max()),
                    value=0
                )
            
            # Apply filters
            filtered_df = df[
                (df['Stage'].isin(stage_filter)) &
                (df['Owner'].isin(owner_filter)) &
                (df['Amount'] >= min_amount)
            ]
            
            # Show filtered metrics
            st.write(f"**Showing {len(filtered_df)} of {len(df)} deals** | Total: ${filtered_df['Amount'].sum():,.0f}")
            
            # Pipeline by Stage Chart
            if not metrics['by_stage'].empty:
                fig = px.bar(
                    metrics['by_stage'],
                    x='Stage',
                    y='Total',
                    title='Pipeline by Stage',
                    labels={'Total': 'Amount ($)', 'Stage': 'Sales Stage'},
                    color='Total',
                    color_continuous_scale='Blues',
                    text='Total'
                )
                fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            # Deal details table
            st.subheader("Deal Details")
            
            # Format the dataframe for display
            display_df = filtered_df.copy()
            display_df['Amount'] = display_df['Amount'].apply(lambda x: f"${x:,.0f}")
            display_df = display_df.sort_values('Opportunity')
            
            st.dataframe(
                display_df[['Opportunity', 'Account', 'Amount', 'Stage', 'Owner', 'Close_Date']],
                use_container_width=True,
                hide_index=True
            )
            
            # Export button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Pipeline Data",
                data=csv,
                file_name=f"pipeline_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
        else:
            st.warning("No pipeline data available. Check connection settings or use Debug Mode.")
    
    with tab3:
        st.header("Intelligence Search")
        
        # Search interface
        search_query = st.text_input("🔍 Search across all data sources...", placeholder="e.g., 'deals closing this month' or 'Madison Communications'")
        
        if search_query:
            with st.spinner("Searching..."):
                # First, search in pipeline data
                st.subheader("Pipeline Results")
                pipeline_results = df[df.apply(lambda row: search_query.lower() in str(row).lower(), axis=1)] if not df.empty else pd.DataFrame()
                
                if not pipeline_results.empty:
                    st.write(f"Found {len(pipeline_results)} deals matching '{search_query}':")
                    display_results = pipeline_results.copy()
                    display_results['Amount'] = display_results['Amount'].apply(lambda x: f"${x:,.0f}")
                    st.dataframe(display_results[['Opportunity', 'Account', 'Amount', 'Stage', 'Owner']], use_container_width=True, hide_index=True)
                else:
                    st.info("No pipeline deals match your search")
                
                # Vector search if configured
                st.subheader("Knowledge Base Results")
                kb_results = search_intelligence(search_query)
                if kb_results:
                    for result in kb_results:
                        if isinstance(result, dict) and 'metadata' in result:
                            with st.expander(f"📄 {result['metadata'].get('title', 'Result')} (Score: {result.get('score', 0):.2f})"):
                                st.write(result['metadata'].get('content', 'No content'))
                else:
                    st.info("Knowledge base search requires OpenAI and Pinecone configuration")
    
    with tab4:
        st.header("Pipeline Analytics")
        
        if not df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Pipeline trend (mock data for now)
                st.subheader("Pipeline Trend")
                st.info("Historical trend analysis coming soon")
                
                # Top deals
                st.subheader("Top 10 Deals")
                top_deals = df.nlargest(10, 'Amount')[['Opportunity', 'Account', 'Amount', 'Stage']]
                top_deals['Amount'] = top_deals['Amount'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(top_deals, use_container_width=True, hide_index=True)
            
            with col2:
                # Owner performance
                st.subheader("Sales Rep Performance")
                if not metrics['by_owner'].empty:
                    fig = px.bar(
                        metrics['by_owner'],
                        x='Owner',
                        y='Total',
                        title='Pipeline by Sales Rep',
                        labels={'Total': 'Pipeline ($)', 'Owner': 'Sales Rep'},
                        text='Count'
                    )
                    fig.update_traces(texttemplate='%{text} deals', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Stage conversion
                st.subheader("Stage Distribution")
                if not metrics['by_stage'].empty:
                    fig = px.pie(
                        metrics['by_stage'],
                        values='Total',
                        names='Stage',
                        title='Pipeline Distribution by Stage'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("Automation Status")
        
        # Synology status
        st.subheader("🖥️ Synology NAS")
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ Pipeline Processor: Active")
            st.info("Last run: Today 6:00 AM")
        with col2:
            st.info(f"Processed: {len(df)} deals")
            st.info("Next run: Tomorrow 6:00 AM")
        
        # Mac Mini status
        st.subheader("💻 Mac Mini")
        col1, col2 = st.columns(2)
        with col1:
            st.warning("⚠️ Email Extraction: Manual")
            st.info("Salesforce email arrives: 4:00 AM")
        with col2:
            if st.button("📧 Setup Automation"):
                st.code("""
# Add to crontab:
30 5 * * * /usr/bin/python3 ~/Desktop/sf_extraction/extract_salesforce_report.py
                """, language="bash")
        
        # Notion status
        st.subheader("📝 Notion Integration")
        st.success(f"✅ Connected: {len(df)} deals synced")
        st.info(f"Database ID: {get_credentials().get('PIPELINE_DB_ID', 'Not set')}")
    
    # Sidebar - System Status
    st.sidebar.header("🔧 System Status")
    
    if df.empty:
        st.sidebar.error("❌ No pipeline data")
    else:
        st.sidebar.success(f"✅ {len(df)} deals loaded")
    
    # Debug Info
    if DEBUG_MODE:
        st.sidebar.header("🔍 Debug Info")
        creds = get_credentials()
        st.sidebar.write("**Credentials Status:**")
        st.sidebar.write("- Notion Token:", "✅" if creds and creds.get("NOTION_TOKEN") else "❌")
        st.sidebar.write("- Pipeline DB:", "✅" if creds and creds.get("PIPELINE_DB_ID") else "❌")
        st.sidebar.write("- OpenAI Key:", "✅" if creds and creds.get("OPENAI_API_KEY") else "❌")
        st.sidebar.write("- Pinecone Key:", "✅" if creds and creds.get("PINECONE_API_KEY") else "❌")
        
        if not df.empty:
            st.sidebar.write("**Data Stats:**")
            st.sidebar.write(f"- Total deals: {len(df)}")
            st.sidebar.write(f"- Total value: ${df['Amount'].sum():,.0f}")
            st.sidebar.write(f"- Unique stages: {df['Stage'].nunique()}")
            st.sidebar.write(f"- Unique owners: {df['Owner'].nunique()}")

if __name__ == "__main__":
    main()
