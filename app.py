## START OF MODULE 1: IMPORTS & CONFIGURATION ##
import streamlit as st
import psycopg2
from pgvector.psycopg2 import register_vector
import requests
from google.oauth2 import service_account
import google.auth.transport.requests
import json
import datetime

# --- CONFIGURATION ---
DB_HOST = "34.151.78.253" 
# We pull the DB Password and GCP Credentials securely from Streamlit Secrets
DB_PASS = st.secrets["DB_PASS"]
PROJECT_ID = "youtube-analysis-473200"
REGION = "australia-southeast1"

st.set_page_config(page_title="Hansard AI Monitor", page_icon="🏛️", layout="wide")

@st.cache_resource
def init_connections():
    conn = psycopg2.connect(host=DB_HOST, user="postgres", password=DB_PASS, dbname="postgres", port=5432)
    register_vector(conn)
    
    # Load Google Credentials directly from Streamlit's secure secrets dictionary
    gcp_creds = dict(st.secrets["gcp_service_account"])
    credentials = service_account.Credentials.from_service_account_info(
        gcp_creds, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    
    return conn, credentials

conn, credentials = init_connections()
## END OF MODULE 1 ##


## START OF MODULE 2: SIDEBAR & SESSION STATE ##
with st.sidebar:
    st.header("📊 Database Status")
    try:
        cursor = conn.cursor()
        # JOIN the tables to count the actual AI speech chunks per jurisdiction
        cursor.execute("""
            SELECT d.jurisdiction, COUNT(s.chunk_id) 
            FROM debates d
            JOIN speeches_ai s ON d.debate_id = s.debate_id
            GROUP BY d.jurisdiction
            ORDER BY COUNT(s.chunk_id) DESC;
        """)
        stats = cursor.fetchall()
        
        total_speeches = 0
        for stat in stats:
            st.metric(label=f"{stat[0]} Speeches", value=f"{stat[1]:,}")
            total_speeches += stat[1]
            
        st.divider()
        st.metric(label="Total Vectorized Speeches", value=f"{total_speeches:,}")
        
        cursor.close()
    except Exception as e:
        st.error(f"Could not load stats: {e}")

st.title("🏛️ Hansard Media Monitor (MVP)")

if "search_results" not in st.session_state:
    st.session_state.search_results = None
## END OF MODULE 2 ##


## START OF MODULE 3: SEARCH UI & DYNAMIC SQL ENGINE ##
query = st.text_input("🔍 Enter your search topic")

# Advanced Filters Container
with st.expander("⚙️ Advanced Search Filters", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        # [THE FIX] Explicitly added "Victoria" to the filter options
        jurisdiction_filter = st.radio("Filter by Parliament:", ["All", "Federal", "NSW", "Victoria", "Queensland", "ACT", "Tasmania", "South Australia", "Western Australia", "Northern Territory"], horizontal=True)
        # Allows partial text matching (e.g., "Max", "Chandler", "Albanese")
        speaker_filter = st.text_input("👤 Filter by Speaker Name (Optional)")
        
    with col2:
        # Default search range: Jan 1, 2025 to today
        today = datetime.date.today()
        default_start = datetime.date(2025, 1, 1)
        date_range = st.date_input("📅 Date Range", value=(default_start, today))
        
        max_results = st.slider("Maximum results to show", min_value=5, max_value=100, value=25)

if st.button("Search Hansard"):
    if query:
        with st.spinner("Embedding query and searching database..."):
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)
            
            # 1. Embed the user's Search Query via Vertex AI
            embed_url = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/publishers/google/models/text-embedding-004:predict"
            headers = {"Authorization": f"Bearer {credentials.token}", "Content-Type": "application/json"}
            ai_response = requests.post(embed_url, json={"instances":[{"content": query}]}, headers=headers)
            
            if ai_response.status_code == 200:
                query_vector = ai_response.json()['predictions'][0]['embeddings']['values']
                
                # 2. Build the dynamic SQL query securely
                sql_params = [query_vector, query_vector]
                where_clauses = ["(1 - (s.embedding <=> %s::vector)) > 0.35"]
                
                if jurisdiction_filter != "All":
                    where_clauses.append("d.jurisdiction = %s")
                    sql_params.append(jurisdiction_filter)
                    
                if speaker_filter:
                    where_clauses.append("s.speaker_name ILIKE %s")
                    sql_params.append(f"%{speaker_filter}%")
                    
                # Handle Streamlit's tuple date output securely
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    where_clauses.append("d.debate_date >= %s AND d.debate_date <= %s")
                    sql_params.extend([date_range[0], date_range[1]])
                elif isinstance(date_range, tuple) and len(date_range) == 1:
                    where_clauses.append("d.debate_date = %s")
                    sql_params.append(date_range[0])
                elif isinstance(date_range, datetime.date):
                    where_clauses.append("d.debate_date = %s")
                    sql_params.append(date_range)

                where_sql = " AND ".join(where_clauses)
                sql_params.append(max_results)
                
                search_sql = f"""
                    SELECT 
                        s.chunk_id, s.speaker_name, s.speaker_metadata, d.topic_title, d.debate_date,
                        s.original_text, 1 - (s.embedding <=> %s::vector) AS similarity_score,
                        d.jurisdiction, d.house
                    FROM speeches_ai s
                    JOIN debates d ON s.debate_id = d.debate_id
                    WHERE {where_sql}
                    ORDER BY similarity_score DESC
                    LIMIT %s;
                """
                
                cursor = conn.cursor()
                cursor.execute(search_sql, tuple(sql_params))
                st.session_state.search_results = cursor.fetchall()
                cursor.close()
            else:
                st.error(f"Failed to embed query: {ai_response.text}")
## END OF MODULE 3 ##


## START OF MODULE 4: DISPLAY RESULTS & RAG SUMMARIES ##
if st.session_state.search_results is not None:
    results = st.session_state.search_results
    
    if len(results) > 0:
        st.success(f"Found {len(results)} highly relevant mentions.")
        for row in results:
            chunk_id, speaker, metadata, topic, date, text, score, jurisdiction, house = row
            
            # GOAL 2: Standardise the speaker display. If it's "Unknown", call it "General Debate"
            display_speaker = "General Debate" if "Unknown" in speaker else speaker
            
            # Shorten the topic if it's too long for the header bar
            display_topic = (topic[:60] + '...') if len(topic) > 60 else topic
            
            is_procedural = "PROCEDURAL" in speaker.upper() or "CLERK" in speaker.upper()
            party_display = "N/A - Procedural" if is_procedural else metadata.get('party', 'Unknown')
            electorate_display = "N/A - Procedural" if is_procedural else metadata.get('electorate', 'Unknown')
            
            # GOAL 2 & 3: Clean, standardised title with NO match score
            with st.expander(f"🏛️ {jurisdiction} | {display_speaker} | {display_topic} | {date}"):
                
                # We put the detailed metadata inside the drop-down
                st.markdown(f"**Parliament:** {jurisdiction} - {house}")
                st.markdown(f"**Speaker:** {speaker} | **Party:** {party_display} | **Electorate:** {electorate_display}")
                st.markdown(f"**Debate Topic:** {topic}")
                st.divider()
                
                if st.button("✨ Generate AI Executive Summary", key=f"btn_{chunk_id}"):
                    with st.spinner("Gemini is analyzing the speech..."):
                        auth_req = google.auth.transport.requests.Request()
                        credentials.refresh(auth_req)
                        
                        GEMINI_REGION = "us-central1"
                        gemini_url = f"https://{GEMINI_REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{GEMINI_REGION}/publishers/google/models/gemini-2.5-flash:generateContent"
                        prompt = f"You are a political analyst. The client is tracking: '{query}'. Read this speech: '{text}'. Write a highly professional, 2-sentence executive summary explaining exactly what the speaker said regarding the topic."
                        headers = {"Authorization": f"Bearer {credentials.token}", "Content-Type": "application/json"}
                        
                        gemini_resp = requests.post(gemini_url, json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]}, headers=headers)
                        
                        if gemini_resp.status_code == 200:
                            summary_text = gemini_resp.json()['candidates'][0]['content']['parts'][0]['text']
                            st.success(f"**Executive Summary:** {summary_text}")
                        else:
                            st.error(f"Failed to generate summary. Google says: {gemini_resp.text}")
                
                st.info(text)
    else:
        st.warning("No highly relevant mentions found for this topic and filter combination.")
## END OF MODULE 4 ##
##END OF CODE app.py##