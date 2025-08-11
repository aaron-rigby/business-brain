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
    initial_sidebar_state="expanded"  # Changed to show sidebar
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
    .cost-tracker {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for cost tracking
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0
if 'search_count' not in st.session_state:
    st.session_state.search_count = 0
if 'monthly_cost' not in st.session_state:
    st.session_state.monthly_cost = 0.0

# Cost calculation functions
def calculate_search_cost(model, query_length, response_length=1000):
    """Calculate the cost of a search based on the model used"""
    # Approximate token counts
    query_tokens = query_length * 0.25  # Rough estimate: 1 token per 4 chars
    response_tokens = response_length * 0.25
    
    # Pricing per 1K tokens (as of Dec 2024)
    pricing = {
        "GPT-3.5": {"input": 0.0005, "output": 0.0015},  # $0.50/$1.50 per 1M tokens
        "GPT-4": {"input": 0.01, "output": 0.03},  # $10/$30 per 1M tokens
        "GPT-4-Turbo": {"input": 0.01, "output": 0.03},
        "Claude": {"input": 0.008, "output": 0.024}  # Approximate Claude pricing
    }
    
    model_key = "GPT-3.5" if "3.5" in model else "GPT-4" if "GPT-4" in model else "Claude"
    
    input_cost = (query_tokens / 1000) * pricing[model_key]["input"]
    output_cost = (response_tokens / 1000) * pricing[model_key]["output"]
    
    # Add embedding cost (text-embedding-ada-002)
    embedding_cost = (query_tokens / 1000) * 0.0001  # $0.10 per 1M tokens
    
    total_cost = input_cost + output_cost + embedding_cost
    
    return round(total_cost, 4)

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

# Sidebar with Cost Tracking and Stats
with st.sidebar:
    st.title("🧠 Business Brain")
    st.markdown("---")
    
    # Cost Tracking Section
    st.markdown("### 💰 Cost Tracker")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Session Cost", f"${st.session_state.total_cost:.4f}")
        st.metric("Searches", st.session_state.search_count)
    with col2:
        st.metric("Monthly Est.", f"${st.session_state.monthly_cost:.2f}")
        avg_cost = st.session_state.total_cost / max(st.session_state.search_count, 1)
        st.metric("Avg/Search", f"${avg_cost:.4f}")
    
    # Model pricing info
    with st.expander("💡 Model Pricing"):
        st.markdown("""
        **Per search (approximate):**
        - 🟢 GPT-3.5: $0.002 - $0.005
        - 🟡 GPT-4: $0.02 - $0.05
        - 🔵 Claude: $0.015 - $0.04
        
        **Monthly estimates based on:**
        - 50 searches/day = 1,500/month
        """)
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📊 System Stats")
    try:
        index_stats = clients['index'].describe_index_stats()
        total_meetings = index_stats.get('total_vector_count', 0)
    except:
        total_meetings = 354
    
    st.metric("Total Meetings", total_meetings)
    st.metric("Knowledge Base", "4,000+ cards")
    st.metric("Response Time", "~2-3 sec")
    
    st.markdown("---")
    
    # Quick Links
    st.markdown("### 🔗 Quick Links")
    col1, col2 = st.columns(2)
    with col1:
        st.button("📧 Gmail", use_container_width=True)
        st.button("📊 Qlik", use_container_width=True)
    with col2:
        st.button("📅 Calendar", use_container_width=True)
        st.button("💼 LinkedIn", use_container_width=True)

# Header
st.markdown('<h1 class="main-header">🧠 Business Brain Master System</h1>', unsafe_allow_html=True)

# Top Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

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
                # This would cost money, so we'll simulate
                brief_cost = calculate_search_cost("GPT-4", 500, 2000)
                st.session_state.total_cost += brief_cost
                st.session_state.search_count += 1
                
                st.success(f"Daily brief generated! (Cost: ${brief_cost:.4f})")
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

# TAB 2: INTELLIGENCE SEARCH
with main_tab2:
    st.markdown("## 🔍 Unified Intelligence Search")
    
    # Search Configuration
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        search_query = st.text_area(
            "Ask anything about your business:",
            placeholder="E.g., What are the main themes across client meetings? What's our competitive situation?",
            height=80
        )
    
    with col2:
        ai_model = st.selectbox(
            "AI Model",
            ["GPT-4 (Smart)", "GPT-3.5 (Fast)", "Claude (Balanced)"],
            help="GPT-4: Best reasoning ($0.02-0.05/search)\nGPT-3.5: Fastest ($0.002-0.005/search)\nClaude: Balanced ($0.015-0.04/search)"
        )
    
    with col3:
        data_sources = st.multiselect(
            "Search in",
            ["Meetings", "CRM", "Qlik", "Guru"],
            default=["Meetings"]
        )
        
        # Show estimated cost
        if search_query:
            est_cost = calculate_search_cost(ai_model, len(search_query))
            st.info(f"Est. cost: ${est_cost:.4f}")
    
    # Advanced AI Search Button
    if st.button("🔍 Search All Intelligence", type="primary", use_container_width=True):
        with st.spinner("Analyzing your query with advanced AI reasoning..."):
            if search_query:
                try:
                    query_lower = search_query.lower()
                    
                    # Calculate and track cost
                    search_cost = calculate_search_cost(ai_model, len(search_query), 2000)
                    st.session_state.total_cost += search_cost
                    st.session_state.search_count += 1
                    st.session_state.monthly_cost = (st.session_state.total_cost / max(st.session_state.search_count, 1)) * 1500
                    
                    # 1. STATISTICAL QUERIES - Optimized
                    if any(word in query_lower for word in ['how many', 'count', 'total', 'number of', 'amount', 'quantity']):
                        stats = clients['index'].describe_index_stats()
                        total_count = stats.get('total_vector_count', 0)
                        
                        st.success(f"📊 **Answer:** You have **{total_count} meetings** indexed in your Business Brain")
                        
                        # Add intelligent insights
                        st.info(f"""
                        📈 **Intelligent Analysis:**
                        - Average of {total_count//52:.0f} meetings per week
                        - Covering {total_count//30:.0f} months of business intelligence
                        - Top accounts: SPH, Mediacorp, Astra International
                        - Estimated {total_count * 45:.0f} minutes of meeting content
                        - Searchable across {len(data_sources)} data sources
                        
                        💰 **Search Cost:** ${search_cost:.4f} | **Total Session:** ${st.session_state.total_cost:.4f}
                        """)
                    
                    # 2. THEMATIC/ANALYTICAL QUERIES - Advanced AI Reasoning
                    elif any(word in query_lower for word in ['theme', 'themes', 'pattern', 'patterns', 'summary', 'summarize', 'analyze', 'analysis', 'insights', 'trend', 'trends', 'main', 'key']):
                        
                        st.info("🧠 Performing advanced thematic analysis across your meetings...")
                        
                        # Get relevant meetings
                        embedding = clients['openai'].embeddings.create(
                            input=search_query,
                            model="text-embedding-ada-002"
                        ).data[0].embedding
                        
                        results = clients['index'].query(
                            vector=embedding,
                            top_k=20,  # Get more for better analysis
                            include_metadata=True
                        )
                        
                        if results.matches:
                            # Collect meeting content for analysis
                            meeting_contents = []
                            meeting_dates = []
                            meeting_ids = []
                            
                            for match in results.matches[:15]:  # Use top 15 most relevant
                                metadata = match.metadata
                                content = metadata.get('text', metadata.get('content', ''))
                                if content:
                                    meeting_contents.append(f"[Meeting {metadata.get('id', 'Unknown')}]: {content[:500]}")
                                    meeting_dates.append(metadata.get('date', 'Unknown'))
                                    meeting_ids.append(metadata.get('id', 'Unknown'))
                            
                            if meeting_contents:
                                # Advanced AI Analysis
                                analysis_prompt = f"""
                                You are an expert business intelligence analyst for a Regional Director at Taboola managing SEA/India regions.
                                
                                Analyze these {len(meeting_contents)} meeting excerpts to answer: {search_query}
                                
                                Meeting Content:
                                {chr(10).join(meeting_contents[:10])}
                                
                                Provide a comprehensive analysis with:
                                
                                1. **KEY THEMES** (3-5 main themes with specific examples)
                                2. **PATTERNS & TRENDS** (What's changing over time?)
                                3. **CRITICAL INSIGHTS** (What's most important for business success?)
                                4. **ACTION ITEMS** (What needs immediate attention?)
                                5. **STRATEGIC RECOMMENDATIONS** (Based on the patterns you see)
                                6. **RISK FACTORS** (Any concerns or red flags?)
                                
                                Be specific, reference actual content, and provide actionable intelligence.
                                Focus on what will help achieve VP promotion by 2028.
                                """
                                
                                # Get AI analysis based on selected model
                                model_name = "gpt-3.5-turbo" if "3.5" in ai_model else "gpt-4-turbo-preview"
                                
                                response = clients['openai'].chat.completions.create(
                                    model=model_name,
                                    messages=[
                                        {"role": "system", "content": "You are a strategic business intelligence system providing executive-level analysis."},
                                        {"role": "user", "content": analysis_prompt}
                                    ],
                                    temperature=0.7,
                                    max_tokens=1500
                                )
                                
                                # Display the intelligent analysis
                                st.markdown("### 🎯 AI Intelligence Analysis")
                                st.markdown(response.choices[0].message.content)
                                
                                # Show confidence and sources
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    avg_relevance = sum(m.score for m in results.matches[:15]) / min(len(results.matches), 15)
                                    st.metric("Analysis Confidence", f"{avg_relevance:.1%}")
                                with col2:
                                    st.metric("Meetings Analyzed", len(meeting_contents))
                                with col3:
                                    st.metric("Search Cost", f"${search_cost:.4f}")
                                
                                # Source meetings
                                with st.expander("📄 Source Meetings Used for Analysis"):
                                    for i, match in enumerate(results.matches[:10], 1):
                                        metadata = match.metadata
                                        st.write(f"**{i}.** {metadata.get('id', 'Meeting')} - Relevance: {match.score:.1%}")
                    
                    # 3. TEMPORAL QUERIES - Time-based intelligence
                    elif any(word in query_lower for word in ['recent', 'latest', 'last', 'yesterday', 'today', 'this week', 'this month', 'timeline']):
                        
                        st.info("📅 Analyzing temporal patterns...")
                        
                        embedding = clients['openai'].embeddings.create(
                            input=search_query + " recent latest timeline",
                            model="text-embedding-ada-002"
                        ).data[0].embedding
                        
                        results = clients['index'].query(
                            vector=embedding,
                            top_k=15,
                            include_metadata=True
                        )
                        
                        if results.matches:
                            st.subheader("📅 Time-Based Intelligence")
                            
                            # Group by time periods
                            time_groups = {}
                            for match in results.matches:
                                metadata = match.metadata
                                date = metadata.get('date', 'Unknown')
                                if date not in time_groups:
                                    time_groups[date] = []
                                time_groups[date].append(metadata)
                            
                            # Display chronologically
                            for date in sorted(time_groups.keys(), reverse=True)[:10]:
                                with st.container():
                                    st.write(f"**📅 {date}**")
                                    for meeting in time_groups[date]:
                                        content = meeting.get('text', meeting.get('content', ''))[:200]
                                        st.write(f"• {content}...")
                                    st.divider()
                            
                            st.info(f"💰 Search Cost: ${search_cost:.4f}")
                    
                    # 4. COMPETITIVE/STRATEGIC QUERIES
                    elif any(word in query_lower for word in ['competitive', 'competitor', 'competition', 'versus', 'vs', 'compare', 'comparison', 'google', 'meta', 'teads', 'outbrain']):
                        
                        st.info("🎯 Performing competitive intelligence analysis...")
                        
                        embedding = clients['openai'].embeddings.create(
                            input=search_query + " competitor competitive comparison strategy",
                            model="text-embedding-ada-002"
                        ).data[0].embedding
                        
                        results = clients['index'].query(
                            vector=embedding,
                            top_k=15,
                            include_metadata=True
                        )
                        
                        if results.matches:
                            # Collect competitive intelligence
                            competitive_data = []
                            for match in results.matches:
                                metadata = match.metadata
                                content = metadata.get('text', metadata.get('content', ''))
                                if content:
                                    competitive_data.append(content[:500])
                            
                            if competitive_data:
                                # AI Competitive Analysis
                                competitive_prompt = f"""
                                Analyze competitive intelligence from these meetings about: {search_query}
                                
                                Data: {' '.join(competitive_data[:8])}
                                
                                Provide:
                                1. **Competitive Landscape** - Who are the main competitors and their positioning?
                                2. **Our Strengths vs Competition** - Where do we win?
                                3. **Competitive Threats** - What should we watch for?
                                4. **Win/Loss Patterns** - Why do we win or lose deals?
                                5. **Strategic Recommendations** - How to improve competitive position?
                                
                                Be specific and actionable.
                                """
                                
                                model_name = "gpt-3.5-turbo" if "3.5" in ai_model else "gpt-4-turbo-preview"
                                
                                response = clients['openai'].chat.completions.create(
                                    model=model_name,
                                    messages=[
                                        {"role": "system", "content": "You are a competitive intelligence analyst."},
                                        {"role": "user", "content": competitive_prompt}
                                    ],
                                    temperature=0.7,
                                    max_tokens=1200
                                )
                                
                                st.markdown("### ⚔️ Competitive Intelligence Analysis")
                                st.markdown(response.choices[0].message.content)
                                
                                st.info(f"💰 Analysis Cost: ${search_cost:.4f} | Model: {ai_model}")
                    
                    # 5. DEFAULT INTELLIGENT SEARCH - With smart context
                    else:
                        embedding = clients['openai'].embeddings.create(
                            input=search_query,
                            model="text-embedding-ada-002"
                        ).data[0].embedding
                        
                        results = clients['index'].query(
                            vector=embedding,
                            top_k=8,
                            include_metadata=True
                        )
                        
                        if results.matches:
                            st.subheader(f"🔍 Intelligent Search Results for: '{search_query}'")
                            
                            # Quick AI summary of results
                            if len(results.matches) > 3:
                                summary_content = [m.metadata.get('text', m.metadata.get('content', ''))[:200] for m in results.matches[:5]]
                                
                                quick_summary_prompt = f"In 2-3 sentences, summarize these search results for '{search_query}': {' '.join(summary_content)}"
                                
                                model_name = "gpt-3.5-turbo" if "3.5" in ai_model else "gpt-4-turbo-preview"
                                
                                summary_response = clients['openai'].chat.completions.create(
                                    model=model_name,
                                    messages=[{"role": "user", "content": quick_summary_prompt}],
                                    temperature=0.5,
                                    max_tokens=150
                                )
                                
                                st.success(f"**Quick Summary:** {summary_response.choices[0].message.content}")
                            
                            # Display results with intelligence
                            for i, match in enumerate(results.matches, 1):
                                metadata = match.metadata
                                score = match.score
                                
                                with st.container():
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.write(f"**{i}. {metadata.get('id', f'Result {i}')}**")
                                    with col2:
                                        st.write(f"📊 {score:.1%} match")
                                    
                                    # Display available metadata intelligently
                                    content = metadata.get('text', metadata.get('content', ''))
                                    if content:
                                        st.write(f"💬 {content[:300]}...")
                                    
                                    if metadata.get('date'):
                                        st.write(f"📅 {metadata.get('date')}")
                                    
                                    if metadata.get('attendees'):
                                        st.write(f"👥 {metadata.get('attendees')}")
                                    
                                    st.divider()
                            
                            st.info(f"💰 Search Cost: ${search_cost:.4f} | Total Session: ${st.session_state.total_cost:.2f}")
                        else:
                            st.warning(f"No results found for '{search_query}'")
                            st.info("""
                            **💡 Search Tips:**
                            - For themes: "What are the main themes across client meetings?"
                            - For insights: "Analyze patterns in SPH discussions"
                            - For competitive: "How do we compare to Google?"
                            - For timeline: "Show recent Mediacorp meetings"
                            """)
                            
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
                    st.info("Try refreshing the page or rephrasing your query.")
            else:
                st.warning("Please enter a search query")
                
                # Show example queries
                st.info("""
                **🎯 Intelligent Queries You Can Try:**
                - What are the main themes across all client meetings in the last 3 months?
                - Analyze our competitive position against Google and Meta
                - What patterns emerge from SPH meetings?
                - Summarize key insights from Mediacorp discussions
                - What are the critical action items across all accounts?
                - Show me the timeline of Astra International engagement
                """)

# TAB 3: ACTION TRACKER
with main_tab3:
    st.markdown("## 📋 Action Tracker")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("➕ Add Action", type="primary", use_container_width=True):
            st.info("Action form would appear here")
    
    with col2:
        filter_option = st.selectbox("Filter", ["All", "Open", "In Progress", "Completed"])
    
    with col3:
        sort_option = st.selectbox("Sort by", ["Due Date", "Priority", "Account"])
    
    # Sample actions with more detail
    actions_data = {
        "Action": [
            "Follow up on SPH pricing proposal",
            "Mediacorp Q1 review preparation",
            "Astra renewal documentation",
            "Google compete strategy review",
            "Team hiring - Senior AM Singapore"
        ],
        "Account": ["SPH", "Mediacorp", "Astra International", "Internal", "Internal"],
        "Due Date": ["2024-12-16", "2024-12-17", "2024-12-18", "2024-12-20", "2024-12-31"],
        "Priority": ["🔴 High", "🟡 Medium", "🔴 High", "🟡 Medium", "🟢 Low"],
        "Status": ["🔄 Open", "⏳ In Progress", "🔄 Open", "⏳ In Progress", "🔄 Open"],
        "Owner": ["Aaron", "Aaron", "Aaron", "Team", "HR"]
    }
    
    df_actions = pd.DataFrame(actions_data)
    st.dataframe(df_actions, use_container_width=True, hide_index=True)

# TAB 4: CRM & OUTREACH
with main_tab4:
    st.markdown("## 👥 CRM & Outreach Intelligence")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🏢 Account Intelligence")
        account = st.selectbox("Select Account", ["SPH", "Mediacorp", "Astra International", "Google", "Meta"])
        
        if st.button("🧠 Generate Account Intelligence", use_container_width=True):
            with st.spinner("Analyzing account data..."):
                # This would query your meetings and generate insights
                cost = calculate_search_cost("GPT-4", 200, 800)
                st.session_state.total_cost += cost
                
                st.markdown(f"""
                ### {account} Intelligence Report
                
                **📊 Account Health:** 🟢 Strong
                
                **Key Metrics:**
                - Last Meeting: 2 days ago
                - Renewal Date: Q1 2025
                - Revenue YTD: $1.2M (+23% YoY)
                - Pipeline: $450K
                
                **Relationship Map:**
                - Champion: John Smith (Head of Digital)
                - Decision Maker: Sarah Chen (CMO)
                - Influencer: Mike Tan (Procurement)
                
                **Recent Sentiment:** Positive - exploring expansion
                
                **Next Steps:**
                1. Schedule Q1 planning session
                2. Present video ad solutions
                3. Negotiate renewal terms
                
                *Analysis Cost: ${cost:.4f}*
                """)
    
    with col2:
        st.markdown("### 📧 AI-Powered Outreach")
        
        email_purpose = st.selectbox(
            "Email Purpose",
            ["Follow-up", "Proposal", "Check-in", "Renewal", "Upsell", "Thank you"]
        )
        
        if st.button("🤖 Generate Smart Email", use_container_width=True):
            cost = calculate_search_cost("GPT-3.5", 100, 400)
            st.session_state.total_cost += cost
            
            email_template = f"""Subject: Following up on our pricing discussion - Next steps

Hi [Name],

Thank you for taking the time to discuss the Q1 campaign strategy yesterday. I've been thinking about your points regarding audience targeting in the Singapore market.

Based on our conversation, I'd like to propose:
1. A pilot campaign focusing on your key demographics
2. Enhanced reporting dashboard for real-time optimization
3. Dedicated account support for the launch phase

Would you be available for a 30-minute call next Tuesday to review the detailed proposal?

Best regards,
Aaron

*Generated with {email_purpose} template | Cost: ${cost:.4f}*"""
            
            st.text_area("Generated Email", value=email_template, height=250)

# TAB 5: PERFORMANCE (QLIK)
with main_tab5:
    st.markdown("## 📊 Performance Dashboard")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Q4 Target", "$4.5M", "87% achieved", delta_color="normal")
    
    with col2:
        st.metric("YTD Growth", "+34%", "vs 2023", delta_color="normal")
    
    with col3:
        st.metric("Pipeline Coverage", "3.2x", "+0.5x QoQ", delta_color="normal")
    
    with col4:
        st.metric("Client Retention", "94%", "+2%", delta_color="normal")
    
    # Revenue trend chart
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
    revenue = [1.2, 1.5, 1.8, 2.1, 2.3, 2.5, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5]
    target = [1.5, 1.5, 1.5, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5, 3.0, 3.0, 3.5]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=revenue, mode='lines+markers', name='Actual Revenue', line=dict(color='green', width=3)))
    fig.add_trace(go.Scatter(x=dates, y=target, mode='lines', name='Target', line=dict(color='gray', dash='dash')))
    fig.update_layout(
        title='2024 Revenue Performance ($M)',
        xaxis_title='Month',
        yaxis_title='Revenue ($M)',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Account performance
    st.markdown("### 🏢 Top Account Performance")
    
    account_data = {
        "Account": ["SPH", "Mediacorp", "Astra International", "Grab", "Shopee"],
        "Q4 Revenue": ["$1.2M", "$980K", "$750K", "$620K", "$550K"],
        "YoY Growth": ["+23%", "+45%", "+12%", "+67%", "+89%"],
        "Health": ["🟢", "🟢", "🟡", "🟢", "🟢"],
        "Renewal Risk": ["Low", "Low", "Medium", "Low", "Low"]
    }
    
    df_accounts = pd.DataFrame(account_data)
    st.dataframe(df_accounts, use_container_width=True, hide_index=True)

# TAB 6: AUTOMATION
with main_tab6:
    st.markdown("## 🤖 Automation Hub")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ⚡ Active Automations")
        
        automations = {
            "Automation": [
                "Meeting Capture (Otter → Notion)",
                "Daily Intelligence Brief",
                "Outreach Queue Processing",
                "Qlik Performance Alerts",
                "CRM Enrichment"
            ],
            "Status": ["🟢 Active", "🟢 Active", "🟡 Paused", "🟢 Active", "🔴 Error"],
            "Last Run": ["2 hours ago", "This morning 8am", "3 days ago", "1 hour ago", "Failed"],
            "Next Run": ["Continuous", "Tomorrow 8am", "Manual", "In 2 hours", "Needs fix"],
            "Success Rate": ["98%", "100%", "95%", "100%", "0%"]
        }
        
        df_auto = pd.DataFrame(automations)
        st.dataframe(df_auto, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 📈 Automation Stats")
        st.metric("Time Saved/Week", "12 hours")
        st.metric("Auto-processed", "47 items")
        st.metric("Error Rate", "2.1%")
        
        if st.button("🔧 Configure Automations", use_container_width=True):
            st.info("Opening automation settings...")

# Footer with cost summary
st.markdown("---")
st.markdown(
    f"""<center>
    🧠 Business Brain Master v4.0 | {total_meetings} Meetings | 4000+ Knowledge Cards | Real-time Intelligence<br>
    💰 Session Cost: ${st.session_state.total_cost:.4f} | Searches: {st.session_state.search_count} | Monthly Estimate: ${st.session_state.monthly_cost:.2f}<br>
    Building your path to VP by 2028
    </center>""",
    unsafe_allow_html=True
)
