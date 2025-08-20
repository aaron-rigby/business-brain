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
import json

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
@st.cache_resource  # Use cache_resource for connection objects
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
def get_pipeline_data():  # No caching to avoid serialization issues
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
            st.write(f"📊 Querying pipeline database: {database_id}")
        
        # Fetch ALL pages, not just first 100
        all_results = []
        has_more = True
        next_cursor = None
        page_count = 0
        
        while has_more:
            if DEBUG_MODE and page_count > 0:
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
            st.write(f"✅ Got {len(all_results)} total deals across {page_count} pages")
        
        if not all_results:
            st.info("Database is connected but contains no deals")
            return pd.DataFrame()
        
        # Process ALL results
        deals = []
        for idx, item in enumerate(all_results):
            try:
                props = item.get('properties', {})
                
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
            st.write("Total pipeline value: ${:,.2f}".format(df['Amount'].sum()))
        
        return df
        
    except Exception as e:
        st.error(f"Failed to fetch pipeline data: {e}")
        if DEBUG_MODE:
            st.write("Full error:", traceback.format_exc())
        return pd.DataFrame()

# ============================================
# MEETING INTELLIGENCE DATA FETCHING
# ============================================
def get_meeting_intelligence():
    """Fetch meeting intelligence from Notion"""
    notion = get_notion_client()
    if not notion:
        return pd.DataFrame()
    
    creds = get_credentials()
    database_id = creds.get("MEETING_DB_ID")
    
    if not database_id:
        if DEBUG_MODE:
            st.info("Meeting Intelligence DB not configured in secrets")
        return pd.DataFrame()
    
    try:
        if DEBUG_MODE:
            st.write(f"📅 Querying meeting database: {database_id}")
        
        # Fetch all meeting records with pagination
        all_results = []
        has_more = True
        next_cursor = None
        
        while has_more:
            response = notion.databases.query(
                database_id=database_id,
                page_size=100,
                start_cursor=next_cursor
            )
            
            all_results.extend(response.get('results', []))
            has_more = response.get('has_more', False)
            next_cursor = response.get('next_cursor', None)
        
        if DEBUG_MODE:
            st.write(f"✅ Got {len(all_results)} meeting records")
        
        # Process meeting records
        meetings = []
        for item in all_results:
            try:
                props = item.get('properties', {})
                
                # Try different field name variations for meeting data
                # Date field
                meeting_date = None
                for date_field in ['Meeting_Date', 'Date', 'When', 'Created_Date']:
                    if date_field in props:
                        date_prop = props[date_field]
                        if 'date' in date_prop and date_prop['date']:
                            meeting_date = date_prop['date'].get('start', '')
                            break
                
                # Participants/Company field
                participants = ""
                for part_field in ['Participants', 'Attendees', 'Company', 'Who', 'Contact']:
                    if part_field in props:
                        part_prop = props[part_field]
                        if 'rich_text' in part_prop:
                            texts = part_prop.get('rich_text', [])
                            if texts:
                                participants = texts[0].get('plain_text', '')
                                break
                        elif 'title' in part_prop:
                            titles = part_prop.get('title', [])
                            if titles:
                                participants = titles[0].get('plain_text', '')
                                break
                
                # Notes/Content field
                notes = ""
                for notes_field in ['Notes', 'Content', 'Summary', 'Description', 'Details']:
                    if notes_field in props:
                        notes_prop = props[notes_field]
                        if 'rich_text' in notes_prop:
                            texts = notes_prop.get('rich_text', [])
                            if texts:
                                notes = texts[0].get('plain_text', '')
                                break
                
                # Key themes/topics
                themes = ""
                for theme_field in ['Key_Themes', 'Topics', 'Themes', 'Subject']:
                    if theme_field in props:
                        theme_prop = props[theme_field]
                        if 'rich_text' in theme_prop:
                            texts = theme_prop.get('rich_text', [])
                            if texts:
                                themes = texts[0].get('plain_text', '')
                                break
                
                # Action items
                actions = ""
                for action_field in ['Action_Items', 'Actions', 'Next_Steps', 'Follow_Up']:
                    if action_field in props:
                        action_prop = props[action_field]
                        if 'rich_text' in action_prop:
                            texts = action_prop.get('rich_text', [])
                            if texts:
                                actions = texts[0].get('plain_text', '')
                                break
                
                meetings.append({
                    'Date': meeting_date,
                    'Participants': participants,
                    'Notes': notes,
                    'Key_Themes': themes,
                    'Action_Items': actions,
                })
                
            except Exception as e:
                if DEBUG_MODE:
                    st.write(f"Error processing meeting record: {e}")
                continue
        
        df = pd.DataFrame(meetings)
        
        if DEBUG_MODE and not df.empty:
            st.write(f"✅ Processed {len(df)} meetings")
        
        return df
        
    except Exception as e:
        if DEBUG_MODE:
            st.error(f"Failed to fetch meetings: {e}")
            st.write("Full error:", traceback.format_exc())
        return pd.DataFrame()

# ============================================
# AI-POWERED SEARCH WITH OPENAI
# ============================================
def search_with_ai(query):
    """Use OpenAI to search and summarize across all data sources"""
    creds = get_credentials()
    
    if not creds.get("OPENAI_API_KEY"):
        # Fallback to basic search without AI
        return basic_search(query)
    
    try:
        # Initialize OpenAI
        openai.api_key = creds["OPENAI_API_KEY"]
        
        # Get all available data
        pipeline_df = get_pipeline_data()
        meetings_df = get_meeting_intelligence()
        
        # Build context from available data
        context_parts = []
        
        # Add pipeline context if query seems pipeline-related
        pipeline_keywords = ['deal', 'pipeline', 'opportunity', 'sales', 'revenue', 'close', 'stage']
        if any(keyword in query.lower() for keyword in pipeline_keywords) and not pipeline_df.empty:
            relevant_deals = pipeline_df[
                pipeline_df.apply(lambda row: any(
                    term in str(row).lower() 
                    for term in query.lower().split()
                ), axis=1)
            ].head(10)
            
            if not relevant_deals.empty:
                context_parts.append("RELEVANT DEALS:\n" + relevant_deals.to_string())
        
        # Add meeting context if query seems meeting-related or mentions specific people/companies
        if not meetings_df.empty:
            # Check for "today", "yesterday", "this week" etc.
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            week_ago = today - timedelta(days=7)
            
            relevant_meetings = meetings_df[
                meetings_df.apply(lambda row: any(
                    term in str(row).lower() 
                    for term in query.lower().split()
                ), axis=1)
            ].head(5)
            
            if not relevant_meetings.empty:
                meeting_context = []
                for _, meeting in relevant_meetings.iterrows():
                    meeting_text = f"""
Meeting Date: {meeting.get('Date', 'Unknown')}
Participants: {meeting.get('Participants', 'Unknown')}
Notes: {meeting.get('Notes', 'No notes')}
Key Themes: {meeting.get('Key_Themes', 'No themes')}
Action Items: {meeting.get('Action_Items', 'No actions')}
---"""
                    meeting_context.append(meeting_text)
                
                context_parts.append("RELEVANT MEETINGS:\n" + "\n".join(meeting_context))
        
        if not context_parts:
            return "No relevant data found for your query."
        
        # Combine all context
        full_context = "\n\n".join(context_parts)
        
        # Use OpenAI to answer the query
        messages = [
            {
                "role": "system", 
                "content": """You are an intelligent business assistant analyzing sales pipeline and meeting data. 
                Provide clear, concise answers based on the data provided. If the data doesn't contain 
                the answer, say so clearly. Format your responses with bullet points where appropriate."""
            },
            {
                "role": "user", 
                "content": f"Based on the following data, answer this question: {query}\n\nDATA:\n{full_context}"
            }
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        if DEBUG_MODE:
            st.error(f"AI search error: {e}")
        return basic_search(query)

def basic_search(query):
    """Fallback basic search without AI"""
    results = []
    
    # Search in pipeline data
    pipeline_df = get_pipeline_data()
    if not pipeline_df.empty:
        pipeline_matches = pipeline_df[
            pipeline_df.apply(lambda row: query.lower() in str(row).lower(), axis=1)
        ]
        if not pipeline_matches.empty:
            results.append(f"**Pipeline Matches ({len(pipeline_matches)} deals):**")
            for _, deal in pipeline_matches.head(5).iterrows():
                results.append(f"• {deal['Opportunity']} - {deal['Account']} (${deal['Amount']:,.0f})")
    
    # Search in meetings
    meetings_df = get_meeting_intelligence()
    if not meetings_df.empty:
        meeting_matches = meetings_df[
            meetings_df.apply(lambda row: query.lower() in str(row).lower(), axis=1)
        ]
        if not meeting_matches.empty:
            results.append(f"\n**Meeting Matches ({len(meeting_matches)} meetings):**")
            for _, meeting in meeting_matches.head(5).iterrows():
                results.append(f"• {meeting.get('Date', 'Unknown date')} - {meeting.get('Participants', 'Unknown participants')}")
    
    return "\n".join(results) if results else "No matches found for your search query."

# ============================================
# PINECONE VECTOR SEARCH (OPTIONAL)
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
        
        if index_name in pc.list_indexes().names():
            return pc.Index(index_name)
        else:
            if DEBUG_MODE:
                st.warning(f"Pinecone index '{index_name}' not found")
            return None
    except Exception as e:
        if DEBUG_MODE:
            st.error(f"Failed to initialize Pinecone: {e}")
        return None

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
    
    # Fetch data
    with st.spinner("Loading data..."):
        df = get_pipeline_data()
        meetings_df = get_meeting_intelligence()
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Display main metrics
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    with col4:
        st.metric(
            "📅 Meetings Tracked",
            len(meetings_df) if not meetings_df.empty else 0,
            delta=None
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Command Center",
        "💰 Salesforce Pipeline",
        "🔍 Intelligence Search",
        "📅 Meeting Intelligence",
        "📊 Analytics",
        "🤖 Automation"
    ])
    
    with tab1:
        st.header("Daily Intelligence Brief")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Pipeline Summary")
            if not metrics['by_stage'].empty:
                for _, row in metrics['by_stage'].head(5).iterrows():
                    st.write(f"**{row['Stage']}**: ${row['Total']:,.0f} ({row['Count']} deals)")
        
        with col2:
            st.subheader("📅 Recent Meetings")
            if not meetings_df.empty:
                recent_meetings = meetings_df.head(5)
                for _, meeting in recent_meetings.iterrows():
                    if meeting.get('Date') and meeting.get('Participants'):
                        st.write(f"• {meeting['Date'][:10]}: {meeting['Participants'][:50]}...")
            else:
                st.info("No meeting data available")
        
        # Quick Actions
        st.subheader("🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Refresh All Data"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        with col2:
            st.button("📧 Process Salesforce Email")
        with col3:
            st.button("📝 Generate Daily Report")
    
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
                    max_value=int(df['Amount'].max()) if df['Amount'].max() > 0 else 1000000,
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
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                if not metrics['by_stage'].empty:
                    fig = px.bar(
                        metrics['by_stage'],
                        x='Stage',
                        y='Total',
                        title='Pipeline by Stage',
                        labels={'Total': 'Amount ($)', 'Stage': 'Sales Stage'},
                        text='Total'
                    )
                    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if not metrics['by_stage'].empty:
                    fig = px.pie(
                        metrics['by_stage'],
                        values='Total',
                        names='Stage',
                        title='Pipeline Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Deal details table
            st.subheader("Deal Details")
            display_df = filtered_df.copy()
            display_df['Amount'] = display_df['Amount'].apply(lambda x: f"${x:,.0f}")
            
            st.dataframe(
                display_df[['Opportunity', 'Account', 'Amount', 'Stage', 'Owner', 'Close_Date']],
                use_container_width=True,
                hide_index=True
            )
            
            # Export
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Export Pipeline Data",
                data=csv,
                file_name=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime='text/csv'
            )
        else:
            st.warning("No pipeline data available")
    
    with tab3:
        st.header("🔍 Intelligence Search")
        st.info("Search across pipeline deals and meeting intelligence using natural language")
        
        # Search interface
        search_query = st.text_input(
            "Ask anything about your business data...",
            placeholder="e.g., 'What were the main themes from my meeting with Sanook?' or 'Show me all deals closing this month'"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_type = st.radio(
                "Search mode:",
                ["AI-Powered (Smart)", "Basic (Keyword)"],
                horizontal=True
            )
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
        if search_query and search_button:
            with st.spinner("Searching..."):
                if search_type == "AI-Powered (Smart)":
                    results = search_with_ai(search_query)
                else:
                    results = basic_search(search_query)
                
                # Display results
                st.subheader("Search Results")
                if results:
                    st.markdown(results)
                else:
                    st.info("No results found for your query")
        
        # Example queries
        with st.expander("📝 Example Queries"):
            st.markdown("""
            **Pipeline Queries:**
            - "Show me all deals closing this month"
            - "What's the total pipeline for Qualified stage?"
            - "Which deals are owned by Pragya?"
            
            **Meeting Queries:**
            - "What were the main themes from today's meetings?"
            - "Show me action items from meetings with Madison Communications"
            - "Summarize yesterday's meeting notes"
            
            **Cross-functional:**
            - "What's the status of the Sanook deal and our last meeting?"
            - "Show me everything related to APAC region"
            """)
    
    with tab4:
        st.header("📅 Meeting Intelligence")
        
        if not meetings_df.empty:
            st.write(f"**Total Meetings Tracked:** {len(meetings_df)}")
            
            # Recent meetings
            st.subheader("Recent Meetings")
            for _, meeting in meetings_df.head(10).iterrows():
                with st.expander(f"{meeting.get('Date', 'Unknown Date')} - {meeting.get('Participants', 'Unknown')[:50]}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Notes:**")
                        st.write(meeting.get('Notes', 'No notes')[:500])
                    with col2:
                        st.write("**Key Themes:**")
                        st.write(meeting.get('Key_Themes', 'No themes'))
                        st.write("**Action Items:**")
                        st.write(meeting.get('Action_Items', 'No action items'))
        else:
            st.info("No meeting data available. Configure MEETING_DB_ID in secrets.")
            st.code("""
# Add to Streamlit Secrets:
MEETING_DB_ID = "your-meeting-database-id"
            """)
    
    with tab5:
        st.header("📊 Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top deals
            st.subheader("🏆 Top 10 Deals")
            if not df.empty:
                top_deals = df.nlargest(10, 'Amount')[['Opportunity', 'Account', 'Amount', 'Stage']]
                top_deals['Amount'] = top_deals['Amount'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(top_deals, use_container_width=True, hide_index=True)
        
        with col2:
            # Owner leaderboard
            st.subheader("👥 Sales Leaderboard")
            if not metrics['by_owner'].empty:
                fig = px.bar(
                    metrics['by_owner'],
                    x='Owner',
                    y='Total',
                    title='Pipeline by Sales Rep',
                    text='Count'
                )
                fig.update_traces(texttemplate='%{text} deals', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.header("🤖 Automation Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖥️ Synology NAS")
            st.success("✅ Pipeline Processor: Active")
            st.info(f"Last sync: {len(df)} deals")
            st.info("Schedule: Daily at 6:00 AM")
        
        with col2:
            st.subheader("💻 Mac Mini")
            st.warning("⚠️ Email Extraction: Manual")
            with st.expander("Setup Automation"):
                st.code("""
# Add to crontab:
30 5 * * * /usr/bin/python3 ~/Desktop/sf_extraction/extract_salesforce_report.py
                """, language="bash")
        
        st.subheader("📊 Data Sources")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Notion Databases", "2 connected")
            st.caption(f"Pipeline: {len(df)} records")
            st.caption(f"Meetings: {len(meetings_df)} records")
        
        with col2:
            creds = get_credentials()
            openai_status = "✅" if creds and creds.get("OPENAI_API_KEY") else "❌"
            st.metric("OpenAI", openai_status)
            st.caption("For AI-powered search")
        
        with col3:
            pinecone_status = "✅" if init_pinecone() else "❌"
            st.metric("Pinecone", pinecone_status)
            st.caption("For vector search")
    
    # Sidebar
    st.sidebar.header("📊 System Overview")
    
    if df.empty:
        st.sidebar.error("❌ No pipeline data")
    else:
        st.sidebar.success(f"✅ {len(df)} deals loaded")
        st.sidebar.info(f"💰 ${metrics['total_pipeline']:,.0f} total pipeline")
    
    if not meetings_df.empty:
        st.sidebar.info(f"📅 {len(meetings_df)} meetings tracked")
    
    # Debug info
    if DEBUG_MODE:
        st.sidebar.header("🔧 Debug Info")
        creds = get_credentials()
        st.sidebar.write("**Credentials:**")
        st.sidebar.write("- Notion:", "✅" if creds and creds.get("NOTION_TOKEN") else "❌")
        st.sidebar.write("- Pipeline DB:", "✅" if creds and creds.get("PIPELINE_DB_ID") else "❌")
        st.sidebar.write("- Meeting DB:", "✅" if creds and creds.get("MEETING_DB_ID") else "❌")
        st.sidebar.write("- OpenAI:", "✅" if creds and creds.get("OPENAI_API_KEY") else "❌")
        st.sidebar.write("- Pinecone:", "✅" if creds and creds.get("PINECONE_API_KEY") else "❌")

if __name__ == "__main__":
    main()
