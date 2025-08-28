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

# ============================================
# CREDENTIAL MANAGEMENT
# ============================================
def get_credentials():
    """Get credentials from Streamlit secrets"""
    try:
        return {
            "NOTION_TOKEN": st.secrets.get("NOTION_TOKEN", ""),
            "PIPELINE_DB_ID": st.secrets.get("PIPELINE_DB_ID", ""),
            "MEETING_DB_ID": st.secrets.get("MEETING_DB_ID", ""),
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
            st.sidebar.error(f"Error details: {traceback.format_exc()}")
        return None

# ============================================
# GET ALL HISTORICAL PIPELINE DATA
# ============================================
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_pipeline_history():
    """Fetch ALL historical pipeline data for temporal analysis"""
    notion = get_notion_client()
    if not notion:
        return pd.DataFrame()
    
    creds = get_credentials()
    database_id = creds.get("PIPELINE_DB_ID")
    
    if not database_id:
        st.error("Pipeline database ID not found in secrets")
        return pd.DataFrame()
    
    try:
        # Fetch ALL pages with pagination
        all_results = []
        has_more = True
        next_cursor = None
        page_count = 0
        
        while has_more:
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
            st.write(f"📚 Fetched {len(all_results)} total historical records across {page_count} pages")
        
        # Process ALL historical results
        deals = []
        for idx, item in enumerate(all_results):
            try:
                props = item.get('properties', {})
                
                # Extract all fields with safe defaults
                amount = props.get('Amount_USD', {}).get('number', 0) or 0
                
                account = "Unknown"
                if props.get('Account_Name', {}).get('rich_text'):
                    account = props.get('Account_Name', {}).get('rich_text', [{}])[0].get('plain_text', 'Unknown')
                
                stage = "Unknown"
                if props.get('Stage', {}).get('select'):
                    stage = props.get('Stage', {}).get('select', {}).get('name', 'Unknown')
                
                date_captured = None
                if props.get('Date_Captured', {}).get('date'):
                    date_captured = props.get('Date_Captured', {}).get('date', {}).get('start', '')
                
                close_date = None
                if props.get('Close_Date', {}).get('date'):
                    close_date = props.get('Close_Date', {}).get('date', {}).get('start', '')
                
                opp_name = "Unknown"
                if props.get('Opportunity Name', {}).get('title'):
                    opp_name = props.get('Opportunity Name', {}).get('title', [{}])[0].get('plain_text', 'Unknown')
                
                owner = "Unknown"
                if props.get('Owner', {}).get('rich_text'):
                    owner = props.get('Owner', {}).get('rich_text', [{}])[0].get('plain_text', 'Unknown')
                
                probability = props.get('Probability', {}).get('number', 0) or 0
                
                market = "Unknown"
                if props.get('Market', {}).get('select'):
                    market = props.get('Market', {}).get('select', {}).get('name', 'Unknown')
                
                deals.append({
                    'Opportunity': opp_name,
                    'Account': account,
                    'Amount': amount,
                    'Stage': stage,
                    'Close_Date': close_date,
                    'Date_Captured': date_captured,
                    'Owner': owner,
                    'Probability': probability,
                    'Market': market,
                })
                
            except Exception as e:
                if DEBUG_MODE and idx < 5:
                    st.write(f"⚠️ Error processing record {idx}: {e}")
                continue
        
        df = pd.DataFrame(deals)
        
        if not df.empty:
            df['Date_Captured'] = pd.to_datetime(df['Date_Captured'], errors='coerce')
            df['Close_Date'] = pd.to_datetime(df['Close_Date'], errors='coerce')
            # Clean up amounts - ensure no NaN values
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        
        return df
        
    except Exception as e:
        st.error(f"Failed to fetch historical data: {e}")
        return pd.DataFrame()

# ============================================
# GET CURRENT PIPELINE DATA (LATEST SNAPSHOT)
# ============================================
def get_pipeline_data():
    """Fetch current pipeline data (latest snapshot only)"""
    df_all = get_pipeline_history()
    
    if df_all.empty:
        return pd.DataFrame()
    
    if 'Date_Captured' not in df_all.columns:
        return pd.DataFrame()
    
    # Ensure Date_Captured is datetime
    df_all['Date_Captured'] = pd.to_datetime(df_all['Date_Captured'], errors='coerce')
    df_all = df_all[df_all['Date_Captured'].notna()]
    
    if df_all.empty:
        return pd.DataFrame()
    
    # Get the latest DATE (not timestamp) to handle multiple snapshots per day
    latest_date = df_all['Date_Captured'].dt.date.max()
    
    # Get ALL records from that DATE (not exact timestamp)
    current_pipeline = df_all[df_all['Date_Captured'].dt.date == latest_date].copy()
    
    # Remove duplicates - keep the one with highest amount per opportunity
    current_pipeline = current_pipeline.sort_values('Amount', ascending=False).drop_duplicates('Opportunity', keep='first')
    
    # Exclude ONLY closed/lost deals - everything else is active pipeline
    excluded_stages = ['Closed Won', 'Closed Lost', 'Closed', 'Lost', 'Disqualified']
    current_pipeline = current_pipeline[~current_pipeline['Stage'].isin(excluded_stages)]
    
    if DEBUG_MODE:
        st.write(f"📅 Latest capture date: {latest_date}")
        st.write(f"📈 Current pipeline: {len(current_pipeline)} deals")
        st.write(f"💰 Total value: ${current_pipeline['Amount'].sum():,.0f}")
    
    return current_pipeline

# ============================================
# TEMPORAL ANALYSIS FUNCTIONS
# ============================================
def calculate_deal_velocity():
    """Calculate average time between stage changes"""
    df_history = get_pipeline_history()
    
    if df_history.empty:
        return {}
    
    # Ensure Date_Captured is datetime
    df_history['Date_Captured'] = pd.to_datetime(df_history['Date_Captured'], errors='coerce')
    df_history = df_history[df_history['Date_Captured'].notna()]
    
    velocity_metrics = {}
    
    for opp in df_history['Opportunity'].unique():
        opp_history = df_history[df_history['Opportunity'] == opp].sort_values('Date_Captured')
        
        if len(opp_history) > 1:
            for i in range(len(opp_history) - 1):
                current = opp_history.iloc[i]
                next_record = opp_history.iloc[i + 1]
                
                if current['Stage'] != next_record['Stage']:
                    days_between = (next_record['Date_Captured'] - current['Date_Captured']).days
                    transition = f"{current['Stage']} → {next_record['Stage']}"
                    
                    if transition not in velocity_metrics:
                        velocity_metrics[transition] = []
                    velocity_metrics[transition].append(days_between)
    
    avg_velocity = {}
    for transition, days_list in velocity_metrics.items():
        if days_list:
            avg_velocity[transition] = {
                'avg_days': np.mean(days_list),
                'median_days': np.median(days_list),
                'count': len(days_list),
                'min_days': min(days_list),
                'max_days': max(days_list)
            }
    
    return avg_velocity

def get_pipeline_trends():
    """Calculate pipeline trends over time"""
    df_history = get_pipeline_history()
    
    if df_history.empty:
        return pd.DataFrame(), {}
    
    # Ensure Date_Captured is datetime
    df_history['Date_Captured'] = pd.to_datetime(df_history['Date_Captured'], errors='coerce')
    df_history = df_history[df_history['Date_Captured'].notna()]
    
    if df_history.empty:
        return pd.DataFrame(), {}
    
    # Exclude closed deals for trend analysis
    excluded_stages = ['Closed Won', 'Closed Lost', 'Closed', 'Lost', 'Disqualified']
    active_history = df_history[~df_history['Stage'].isin(excluded_stages)].copy()
    
    if active_history.empty:
        return pd.DataFrame(), {}
    
    # Group by date, remove duplicates per day
    active_history['DateOnly'] = active_history['Date_Captured'].dt.date
    daily_data = active_history.sort_values('Amount', ascending=False).drop_duplicates(['DateOnly', 'Opportunity'], keep='first')
    
    # Group by date to get daily snapshots
    daily_pipeline = daily_data.groupby('DateOnly').agg({
        'Amount': 'sum',
        'Opportunity': 'count'
    }).reset_index()
    daily_pipeline.columns = ['Date', 'Total_Amount', 'Deal_Count']
    
    # Calculate week-over-week metrics
    metrics = {}
    if len(daily_pipeline) >= 14:
        current_week = daily_pipeline.tail(7)['Total_Amount'].mean()
        previous_week = daily_pipeline.tail(14).head(7)['Total_Amount'].mean()
        wow_change = ((current_week - previous_week) / previous_week * 100) if previous_week > 0 else 0
        
        metrics['current_week_avg'] = current_week
        metrics['previous_week_avg'] = previous_week
        metrics['wow_change'] = wow_change
    
    return daily_pipeline, metrics

def get_win_loss_analysis():
    """Analyze closed deals for win/loss rates"""
    df_history = get_pipeline_history()
    
    if df_history.empty:
        return {}
    
    closed_deals = df_history[df_history['Stage'].isin(['Closed Won', 'Closed Lost'])]
    
    if closed_deals.empty:
        return {}
    
    # Ensure Date_Captured is datetime for sorting
    closed_deals['Date_Captured'] = pd.to_datetime(closed_deals['Date_Captured'], errors='coerce')
    latest_status = closed_deals.sort_values('Date_Captured').groupby('Opportunity').last()
    
    won = len(latest_status[latest_status['Stage'] == 'Closed Won'])
    lost = len(latest_status[latest_status['Stage'] == 'Closed Lost'])
    total_closed = won + lost
    
    won_amount = latest_status[latest_status['Stage'] == 'Closed Won']['Amount'].sum()
    lost_amount = latest_status[latest_status['Stage'] == 'Closed Lost']['Amount'].sum()
    
    return {
        'deals_won': won,
        'deals_lost': lost,
        'win_rate': (won / total_closed * 100) if total_closed > 0 else 0,
        'won_amount': won_amount,
        'lost_amount': lost_amount,
        'avg_won_size': won_amount / won if won > 0 else 0,
        'avg_lost_size': lost_amount / lost if lost > 0 else 0
    }

# ============================================
# MEETING INTELLIGENCE
# ============================================
def get_meeting_intelligence():
    """Fetch meeting intelligence from Notion"""
    notion = get_notion_client()
    if not notion:
        return pd.DataFrame()
    
    creds = get_credentials()
    database_id = creds.get("MEETING_DB_ID")
    
    if not database_id:
        return pd.DataFrame()
    
    try:
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
        
        meetings = []
        for item in all_results:
            try:
                props = item.get('properties', {})
                
                meeting_date = None
                for date_field in ['Meeting_Date', 'Date', 'When', 'Created_Date']:
                    if date_field in props and props[date_field].get('date'):
                        meeting_date = props[date_field]['date'].get('start', '')
                        break
                
                participants = ""
                for part_field in ['Participants', 'Attendees', 'Company', 'Who']:
                    if part_field in props:
                        if props[part_field].get('rich_text'):
                            participants = props[part_field]['rich_text'][0].get('plain_text', '')
                            break
                        elif props[part_field].get('title'):
                            participants = props[part_field]['title'][0].get('plain_text', '')
                            break
                
                notes = ""
                for notes_field in ['Notes', 'Content', 'Summary', 'Description']:
                    if notes_field in props and props[notes_field].get('rich_text'):
                        notes = props[notes_field]['rich_text'][0].get('plain_text', '')
                        break
                
                meetings.append({
                    'Date': meeting_date,
                    'Participants': participants,
                    'Notes': notes,
                })
                
            except Exception as e:
                continue
        
        return pd.DataFrame(meetings)
        
    except Exception as e:
        return pd.DataFrame()

# ============================================
# AI-POWERED SEARCH
# ============================================
def search_with_ai(query):
    """Use OpenAI to search and summarize across all data sources"""
    creds = get_credentials()
    
    if not creds.get("OPENAI_API_KEY"):
        return basic_search(query)
    
    try:
        openai.api_key = creds["OPENAI_API_KEY"]
        
        pipeline_df = get_pipeline_data()
        meetings_df = get_meeting_intelligence()
        
        context_parts = []
        
        if not pipeline_df.empty:
            relevant_deals = pipeline_df[
                pipeline_df.apply(lambda row: any(
                    term in str(row).lower() 
                    for term in query.lower().split()
                ), axis=1)
            ].head(10)
            
            if not relevant_deals.empty:
                context_parts.append("RELEVANT DEALS:\n" + relevant_deals.to_string())
        
        if not meetings_df.empty:
            relevant_meetings = meetings_df[
                meetings_df.apply(lambda row: any(
                    term in str(row).lower() 
                    for term in query.lower().split()
                ), axis=1)
            ].head(5)
            
            if not relevant_meetings.empty:
                context_parts.append("RELEVANT MEETINGS:\n" + relevant_meetings.to_string())
        
        if not context_parts:
            return "No relevant data found for your query."
        
        full_context = "\n\n".join(context_parts)
        
        messages = [
            {
                "role": "system", 
                "content": "You are an intelligent business assistant. Provide clear, concise answers based on the data provided."
            },
            {
                "role": "user", 
                "content": f"Based on this data, answer: {query}\n\nDATA:\n{full_context}"
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
        return basic_search(query)

def basic_search(query):
    """Fallback basic search without AI"""
    results = []
    
    pipeline_df = get_pipeline_data()
    if not pipeline_df.empty:
        pipeline_matches = pipeline_df[
            pipeline_df.apply(lambda row: query.lower() in str(row).lower(), axis=1)
        ]
        if not pipeline_matches.empty:
            results.append(f"**Pipeline Matches ({len(pipeline_matches)} deals):**")
            for _, deal in pipeline_matches.head(5).iterrows():
                results.append(f"• {deal['Opportunity']} - {deal['Account']} (${deal['Amount']:,.0f})")
    
    meetings_df = get_meeting_intelligence()
    if not meetings_df.empty:
        meeting_matches = meetings_df[
            meetings_df.apply(lambda row: query.lower() in str(row).lower(), axis=1)
        ]
        if not meeting_matches.empty:
            results.append(f"\n**Meeting Matches ({len(meeting_matches)}):**")
            for _, meeting in meeting_matches.head(5).iterrows():
                results.append(f"• {meeting.get('Date', 'Unknown')} - {meeting.get('Participants', 'Unknown')}")
    
    return "\n".join(results) if results else "No matches found."

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
    
    # Ensure Amount column is numeric
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    
    # Calculate metrics for ALL deals (no filtering)
    total = df['Amount'].sum()
    num_deals = len(df)
    avg_size = total / num_deals if num_deals > 0 else 0
    
    by_stage = df.groupby('Stage')['Amount'].agg(['sum', 'count']).reset_index()
    by_stage.columns = ['Stage', 'Total', 'Count']
    by_stage = by_stage.sort_values('Total', ascending=False)
    
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
    # Fetch data
    with st.spinner("Loading data..."):
        df = get_pipeline_data()
        df_history = get_pipeline_history()
        meetings_df = get_meeting_intelligence()
    
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Display main metrics - showing FULL pipeline
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Current Pipeline",
            "${:,.0f}".format(metrics['total_pipeline']),
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
            "${:,.0f}".format(metrics['avg_deal_size']),
            delta=None
        )
    
    with col4:
        _, trend_metrics = get_pipeline_trends()
        if trend_metrics and 'wow_change' in trend_metrics:
            st.metric(
                "📈 Week-over-Week",
                f"{trend_metrics['wow_change']:+.1f}%",
                delta=None
            )
        else:
            st.metric("📅 Meetings", len(meetings_df) if not meetings_df.empty else 0)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 Command Center",
        "💰 Salesforce Pipeline",
        "🔍 Intelligence Search",
        "📅 Meeting Intelligence",
        "📊 Analytics",
        "📈 Temporal Analysis",
        "🤖 Automation"
    ])
    
    with tab1:
        st.header("Daily Intelligence Brief")
        
        if not df_history.empty and 'Date_Captured' in df_history.columns:
            # Ensure Date_Captured is datetime before using .dt accessor
            df_history['Date_Captured'] = pd.to_datetime(df_history['Date_Captured'], errors='coerce')
            valid_dates_df = df_history[df_history['Date_Captured'].notna()]
            
            if not valid_dates_df.empty:
                valid_dates = valid_dates_df['Date_Captured'].dt.date.unique()
                available_dates = sorted(valid_dates, reverse=True)
                
                selected_date = st.date_input(
                    "View pipeline for date:",
                    value=available_dates[0],
                    min_value=available_dates[-1],
                    max_value=available_dates[0]
                )
                
                selected_df = df_history[
                    (df_history['Date_Captured'].dt.date == selected_date) &
                    (~df_history['Stage'].isin(['Closed Won', 'Closed Lost', 'Closed', 'Lost', 'Disqualified']))
                ]
                # Remove duplicates for selected date display
                selected_df = selected_df.sort_values('Amount', ascending=False).drop_duplicates('Opportunity', keep='first')
                
                if not selected_df.empty:
                    st.info(f"Pipeline on {selected_date}: ${selected_df['Amount'].sum():,.0f} ({len(selected_df)} deals)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Pipeline by Stage")
            if not metrics['by_stage'].empty:
                for _, row in metrics['by_stage'].head(5).iterrows():
                    st.write(f"**{row['Stage']}**: ${row['Total']:,.0f} ({int(row['Count'])} deals)")
        
        with col2:
            st.subheader("🏆 Win/Loss Metrics")
            win_loss = get_win_loss_analysis()
            if win_loss:
                st.metric("Win Rate", f"{win_loss['win_rate']:.1f}%")
                st.write(f"Won: {win_loss['deals_won']} deals (${win_loss['won_amount']:,.0f})")
                st.write(f"Lost: {win_loss['deals_lost']} deals (${win_loss['lost_amount']:,.0f})")
        
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
            st.button("📝 Generate Report")
    
    with tab2:
        st.header("Salesforce Pipeline Analysis")
        
        if not df.empty:
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
            
            filtered_df = df[
                (df['Stage'].isin(stage_filter)) &
                (df['Owner'].isin(owner_filter)) &
                (df['Amount'] >= min_amount)
            ]
            
            st.write(f"**Showing {len(filtered_df)} of {len(df)} deals** | Total: ${filtered_df['Amount'].sum():,.0f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if not metrics['by_stage'].empty:
                    fig = px.bar(
                        metrics['by_stage'],
                        x='Stage',
                        y='Total',
                        title='Pipeline by Stage',
                        labels={'Total': 'Amount ($)'},
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
            
            st.subheader("Deal Details")
            display_df = filtered_df.copy()
            display_df['Amount'] = display_df['Amount'].apply(lambda x: f"${x:,.0f}")
            if 'Close_Date' in display_df.columns:
                display_df['Close_Date'] = pd.to_datetime(display_df['Close_Date'], errors='coerce')
                display_df['Close_Date'] = display_df['Close_Date'].dt.strftime('%Y-%m-%d').fillna('')
            
            st.dataframe(
                display_df[['Opportunity', 'Account', 'Amount', 'Stage', 'Owner', 'Close_Date']],
                use_container_width=True,
                hide_index=True
            )
            
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Export Current Pipeline",
                data=csv,
                file_name=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime='text/csv'
            )
        else:
            st.warning("No pipeline data available")
    
    with tab3:
        st.header("🔍 Intelligence Search")
        
        search_query = st.text_input(
            "Ask anything about your business data...",
            placeholder="e.g., 'What were the main themes from my meeting with Sanook?' or 'Show deals closing this month'"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_type = st.radio(
                "Search mode:",
                ["AI-Powered", "Basic"],
                horizontal=True
            )
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
        if search_query and search_button:
            with st.spinner("Searching..."):
                if search_type == "AI-Powered":
                    results = search_with_ai(search_query)
                else:
                    results = basic_search(search_query)
                
                st.subheader("Search Results")
                if results:
                    st.markdown(results)
                else:
                    st.info("No results found")
    
    with tab4:
        st.header("📅 Meeting Intelligence")
        
        if not meetings_df.empty:
            st.write(f"**Total Meetings:** {len(meetings_df)}")
            
            for _, meeting in meetings_df.head(10).iterrows():
                with st.expander(f"{meeting.get('Date', 'Unknown')} - {meeting.get('Participants', 'Unknown')[:50]}"):
                    st.write(meeting.get('Notes', 'No notes')[:500])
        else:
            st.info("No meeting data available. Add MEETING_DB_ID to secrets.")
    
    with tab5:
        st.header("📊 Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 10 Deals")
            if not df.empty:
                top_deals = df.nlargest(10, 'Amount')[['Opportunity', 'Account', 'Amount', 'Stage']]
                top_deals = top_deals.copy()
                top_deals['Amount'] = top_deals['Amount'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(top_deals, use_container_width=True, hide_index=True)
        
        with col2:
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
        st.header("📈 Temporal Analysis")
        
        if df_history.empty:
            st.warning("No historical data available for temporal analysis")
        else:
            st.subheader("Pipeline Trend")
            daily_pipeline, trend_metrics = get_pipeline_trends()
            
            if not daily_pipeline.empty:
                fig = px.line(
                    daily_pipeline,
                    x='Date',
                    y='Total_Amount',
                    title='Pipeline Value Over Time',
                    labels={'Total_Amount': 'Pipeline ($)', 'Date': 'Date'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if trend_metrics:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Week Avg", f"${trend_metrics['current_week_avg']:,.0f}")
                    with col2:
                        st.metric("Previous Week Avg", f"${trend_metrics['previous_week_avg']:,.0f}")
                    with col3:
                        st.metric("Week-over-Week", f"{trend_metrics['wow_change']:+.1f}%")
            
            st.subheader("Deal Velocity Analysis")
            velocity = calculate_deal_velocity()
            
            if velocity:
                velocity_df = pd.DataFrame([
                    {
                        'Transition': transition,
                        'Avg Days': metrics['avg_days'],
                        'Median Days': metrics['median_days'],
                        'Count': metrics['count']
                    }
                    for transition, metrics in velocity.items()
                ])
                
                if not velocity_df.empty:
                    velocity_df = velocity_df.sort_values('Count', ascending=False)
                    st.write("**Average Time Between Stage Changes:**")
                    for _, row in velocity_df.head(10).iterrows():
                        st.write(f"• **{row['Transition']}**: {row['Avg Days']:.1f} days (n={int(row['Count'])})")
    
    with tab7:
        st.header("🤖 Automation Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖥️ Synology NAS")
            if df_history.empty:
                st.error("❌ No data received")
            else:
                # Ensure Date_Captured is datetime
                df_history['Date_Captured'] = pd.to_datetime(df_history['Date_Captured'], errors='coerce')
                valid_dates = df_history[df_history['Date_Captured'].notna()]['Date_Captured']
                if not valid_dates.empty:
                    latest_capture = valid_dates.max()
                    st.success("✅ Pipeline Processor: Active")
                    st.info(f"Last sync: {latest_capture.strftime('%Y-%m-%d %H:%M')}")
                    st.info(f"Records: {len(df_history)} total, {len(df)} current")
        
        with col2:
            st.subheader("💻 Mac Mini")
            st.success("✅ Email Extraction: Automated")
            st.info("Cron runs at 5:30 AM daily")
        
        st.subheader("📊 Data Quality Check")
        if not df_history.empty and 'Date_Captured' in df_history.columns:
            # Ensure datetime conversion
            df_history['Date_Captured'] = pd.to_datetime(df_history['Date_Captured'], errors='coerce')
            valid_history = df_history[df_history['Date_Captured'].notna()]
            if not valid_history.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    unique_dates = valid_history['Date_Captured'].dt.date.nunique()
                    st.metric("Capture Dates", unique_dates)
                
                with col2:
                    unique_opps = valid_history['Opportunity'].nunique()
                    st.metric("Unique Opportunities", unique_opps)
                
                with col3:
                    avg_daily = len(valid_history) / unique_dates if unique_dates > 0 else 0
                    st.metric("Avg Deals/Day", f"{avg_daily:.0f}")
    
    # Sidebar
    st.sidebar.header("📊 System Overview")
    
    if df.empty:
        st.sidebar.error("❌ No current pipeline data")
    else:
        st.sidebar.success(f"✅ {len(df)} active deals")
        st.sidebar.info(f"💰 ${metrics['total_pipeline']:,.0f} pipeline")
    
    if not df_history.empty:
        st.sidebar.info(f"📚 {len(df_history)} historical records")
        if 'Date_Captured' in df_history.columns:
            # Ensure datetime conversion
            df_history['Date_Captured'] = pd.to_datetime(df_history['Date_Captured'], errors='coerce')
            valid_dates = df_history[df_history['Date_Captured'].notna()]['Date_Captured'].dt.date.nunique()
            st.sidebar.info(f"📅 {valid_dates} days of data")
    
    if DEBUG_MODE:
        st.sidebar.header("🔧 Debug Info")
        creds = get_credentials()
        st.sidebar.write("**Credentials:**")
        st.sidebar.write("- Notion:", "✅" if creds and creds.get("NOTION_TOKEN") else "❌")
        st.sidebar.write("- Pipeline DB:", "✅" if creds and creds.get("PIPELINE_DB_ID") else "❌")
        st.sidebar.write("- Meeting DB:", "✅" if creds and creds.get("MEETING_DB_ID") else "❌")
        st.sidebar.write("- OpenAI:", "✅" if creds and creds.get("OPENAI_API_KEY") else "❌")

if __name__ == "__main__":
    main()
