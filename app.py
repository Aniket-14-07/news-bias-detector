import streamlit as st
import json
import os
import re
import time
from newspaper import Article
from langchain_groq import ChatGroq
from tavily import TavilyClient
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import httpx
os.environ["GROQ_TIMEOUT"] = "60"

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="News Bias Detector",
    page_icon="🔍",
    layout="wide"
)

# ── CUSTOM CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .report-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2c3e50;
        font-size: 0.95rem;
        line-height: 1.8;
        color: #2c3e50;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──
st.markdown('<div class="main-header">🔍 News Bias Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze political bias, emotional language, and credibility across political and current affairs news articles</div>', unsafe_allow_html=True)
st.info("Designed for political and current affairs articles. Results are AI-generated interpretations and should be used as a starting point for analysis, not as definitive conclusions.")

# ── SIDEBAR: API KEYS ──
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    tavily_key = st.text_input("Tavily API Key", type="password", placeholder="tvly-...")
    st.markdown("---")
    st.markdown("**Get free API keys:**")
    st.markdown("- [Groq Console](https://console.groq.com)")
    st.markdown("- [Tavily](https://tavily.com)")
    st.markdown("---")
    num_sources = st.slider("Max sources to analyze", 3, 8, 5)
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Paste a news article URL")
    st.markdown("2. System scrapes the article")
    st.markdown("3. Searches web for same story")
    st.markdown("4. Analyzes bias across all sources")
    st.markdown("5. Generates comparison report")

# ── INITIALIZE LLM AND TAVILY ──
def init_clients(groq_key, tavily_key):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0,
        http_client=httpx.Client(verify=False)
    )
    tavily = TavilyClient(api_key=tavily_key)
    return llm, tavily

# ── SCRAPE ARTICLE ──
def scrape_article(url: str) -> dict:
    try:
        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        response = requests.get(url, verify=False, headers=headers, timeout=15)

        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code} — could not fetch page"}

        article = Article(url)
        article.download(input_html=response.text)
        article.parse()

        if len(article.text) < 100:
            return {"error": "Article too short or could not be parsed"}

        return {
            "url": url,
            "title": article.title,
            "text": article.text[:5000],
            "authors": article.authors,
            "source": url.split("/")[2].replace("www.", "")
        }

    except Exception as e:
        return {"error": str(e)}
        
# ── ANALYZE ARTICLE ──
def analyze_article(article: dict, llm) -> dict:
    text = article['text']
    title = article['title']
    source = article['source']

    # ── PROMPT 1: POLITICAL BIAS ──
    bias_prompt = f"""You are a political media analyst with expertise in identifying ideological framing in journalism.

Article Title: {title}
Source: {source}
Article Text: {text}

Task: Analyze the political bias of this article in two steps.

Step 1 — Quote 2-3 specific sentences or phrases from the article that reveal its political leaning. Look for: which side gets more favorable framing, whose quotes are included, what language is used to describe each party, what facts are emphasized or omitted.

Step 2 — Based only on the evidence you found, assign a bias score.

Return ONLY valid JSON, nothing else:
{{
    "evidence": ["quote or phrase 1", "quote or phrase 2", "quote or phrase 3"],
    "score": <integer from -10 to +10. -10 = Far Left, 0 = Neutral, +10 = Far Right>,
    "label": <"Far Left" | "Left" | "Center Left" | "Neutral" | "Center Right" | "Right" | "Far Right">,
    "reason": <one precise sentence explaining the score based on the evidence above>
}}

Scoring guide:
-10 to -7: Strongly advocates left positions, dismisses right perspectives entirely
-6 to -4: Clear left bias in framing and source selection
-3 to -1: Slight left lean, mostly balanced
0: Genuinely neutral, both sides represented fairly
+1 to +3: Slight right lean, mostly balanced
+4 to +6: Clear right bias in framing and source selection
+7 to +10: Strongly advocates right positions, dismisses left perspectives entirely

Be precise. Do not default to the center. If the evidence shows bias, reflect it in the score."""

    # ── PROMPT 2: EMOTIONAL LANGUAGE ──
    emotion_prompt = f"""You are a linguistics analyst specializing in media rhetoric and emotional language detection.

Article Title: {title}
Source: {source}
Article Text: {text}

Task: Analyze the emotional language used in this article in two steps.

Step 1 — Extract 3-4 specific words or phrases from the article that are emotionally charged, sensationalized, or designed to provoke a reaction. Look for: loaded adjectives, dramatic verbs, hyperbole, fear-inducing language, hero/villain framing.

Step 2 — Score the overall emotional intensity.

Return ONLY valid JSON, nothing else:
{{
    "examples": ["word or phrase 1", "word or phrase 2", "word or phrase 3"],
    "score": <integer from 0 to 10. 0 = purely factual, 10 = highly sensationalized>,
    "label": <"Factual" | "Slightly Emotional" | "Moderately Emotional" | "Highly Emotional">,
    "reason": <one sentence explaining the score with reference to the examples found>
}}

Scoring guide:
0-2: Clinical, factual reporting. Dates, numbers, direct quotes only.
3-4: Occasional descriptive language but mostly factual.
5-6: Noticeable emotional framing. Adjectives used to color perception.
7-8: Frequent emotional language. Clear attempt to provoke reader reaction.
9-10: Sensationalized throughout. More rhetoric than reporting.

Be strict. If you found emotionally charged phrases, the score should reflect that."""

    # ── PROMPT 3: MISSING CONTEXT ──
    context_prompt = f"""You are an investigative journalism editor who evaluates whether news articles give readers complete context.

Article Title: {title}
Source: {source}
Article Text: {text}

Task: Evaluate what important context, perspectives, or facts are missing from this article.

Consider: Are opposing viewpoints represented? Are relevant historical facts included? Are affected parties given a voice? Are statistics sourced? Is the broader political/social context explained?

Return ONLY valid JSON, nothing else:
{{
    "missing_elements": ["missing element 1", "missing element 2", "missing element 3"],
    "score": <integer from 0 to 10. 0 = complete context, 10 = major context missing>,
    "label": <"Complete" | "Minor Gaps" | "Moderate Gaps" | "Major Gaps">,
    "what_is_missing": <one precise sentence describing the most significant gap>
}}

Scoring guide:
0-2: All major perspectives represented. Sources cited. Historical context provided.
3-4: Minor omissions that don't significantly affect understanding.
5-6: Notable gaps. One major perspective or relevant fact is missing.
7-8: Significant gaps. Reader gets a one-sided picture without realizing it.
9-10: Severely incomplete. Critical context deliberately or carelessly omitted."""

    # ── PROMPT 4: CREDIBILITY ──
    credibility_prompt = f"""You are a fact-checking editor evaluating the journalistic standards of a news article.

Article Title: {title}
Source: {source}
Article Text: {text}

Task: Evaluate the credibility and journalistic standards of this article.

Consider: Are claims attributed to named sources? Are statistics cited? Is anonymous sourcing overused? Are both sides contacted for comment? Does the headline match the content? Is speculation clearly labeled as such?

Return ONLY valid JSON, nothing else:
{{
    "credibility_indicators": ["indicator 1", "indicator 2", "indicator 3"],
    "score": <integer from 0 to 10. 10 = highest credibility>,
    "label": <"Very Low" | "Low" | "Medium" | "High" | "Very High">,
    "reason": <one sentence explaining the score based on specific journalistic standards observed>
}}

Scoring guide:
0-2: No sources cited, speculation presented as fact, headline misleading.
3-4: Minimal sourcing, significant factual claims unverified.
5-6: Some sourcing but gaps in verification or balance.
7-8: Generally well sourced with minor issues.
9-10: All claims attributed, multiple sources, balanced representation."""

    # ── PROMPT 5: SUMMARY ──
    summary_prompt = f"""Summarize the following news article in exactly 2 sentences. 
Be completely neutral. State only what happened — no interpretation, no adjectives, no opinion.

Article: {text[:2000]}

Return ONLY the 2-sentence summary, nothing else."""

    # ── INVOKE ALL PROMPTS ──
    def safe_invoke(prompt, retries=2):
        for attempt in range(retries):
            try:
                response = llm.invoke(prompt)
                clean = response.content.strip().replace("```json", "").replace("```", "").strip()
                return json.loads(clean)
            except Exception as e:
                if attempt == retries - 1:
                    return None
                time.sleep(2)
        return None

    bias_result      = safe_invoke(bias_prompt)
    emotion_result   = safe_invoke(emotion_prompt)
    context_result   = safe_invoke(context_prompt)
    credibility_result = safe_invoke(credibility_prompt)

    summary_response = llm.invoke(summary_prompt)
    summary = summary_response.content.strip()

    # ── FALLBACK IF ANY PROMPT FAILS ──
    if not bias_result:
        bias_result = {"score": 0, "label": "Neutral", "reason": "Analysis unavailable", "evidence": []}
    if not emotion_result:
        emotion_result = {"score": 5, "label": "Moderately Emotional", "reason": "Analysis unavailable", "examples": []}
    if not context_result:
        context_result = {"score": 5, "label": "Moderate Gaps", "what_is_missing": "Analysis unavailable", "missing_elements": []}
    if not credibility_result:
        credibility_result = {"score": 7, "label": "High", "reason": "Analysis unavailable", "credibility_indicators": []}

    return {
        "title": title,
        "source": source,
        "url": article["url"],
        "summary": summary,
        "political_bias": bias_result,
        "emotional_language": emotion_result,
        "missing_context": context_result,
        "credibility": credibility_result
    }

# ── CLEAN SOURCE NAMES ──
def clean_source_name(source):
    name_map = {
        "bbc": "BBC", "economictimes": "Economic Times",
        "newsd": "Newsd", "ndtv": "NDTV",
        "thequint": "The Quint", "timesofindia": "Times of India",
        "thehindu": "The Hindu", "hindustantimes": "Hindustan Times",
        "indiatoday": "India Today", "scroll": "Scroll",
        "firstpost": "Firstpost", "reuters": "Reuters",
        "cnn": "CNN", "bbc": "BBC", "foxnews": "Fox News",
        "guardian": "The Guardian", "nytimes": "NY Times",
        "washingtonpost": "Washington Post"
    }
    clean = source.replace(".com","").replace(".in","").replace(".net","").replace(".org","").replace("www.","")
    clean = clean.replace("indiatimes","").strip(".")
    for key, value in name_map.items():
        if key in clean.lower():
            return value
    return clean.split(".")[0].title()

# ── SEARCH AND SCRAPE ──
def search_and_scrape(original_article: dict, llm, tavily, num_sources: int) -> tuple[list, str]:
    query_response = llm.invoke(
        f"Extract the core news topic as a short search query (max 8 words). Headline: {original_article['title']}. Reply with only the query."
    )
    search_query = query_response.content.strip()

    search_results = tavily.search(query=search_query, max_results=num_sources+3, search_depth="advanced")
    original_domain = original_article['source']
    filtered = [r for r in search_results['results'] if original_domain not in r['url']]

    all_analyses = []

    original_analysis = analyze_article(original_article, llm)
    if "error" not in original_analysis:
        all_analyses.append(original_analysis)

    successful = 0
    for r in filtered:
        if successful >= num_sources:
            break
        scraped = scrape_article(r['url'])
        if "error" in scraped:
            continue
        analysis = analyze_article(scraped, llm)
        if "error" not in analysis:
            all_analyses.append(analysis)
            successful += 1
        time.sleep(1)

    return all_analyses, search_query

# ── GENERATE REPORT ──
def generate_report(all_analyses: list, llm) -> str:
    sources_summary = ""
    for i, a in enumerate(all_analyses):
        sources_summary += f"""
Source {i+1}: {a['source']}
- Political Bias: {a['political_bias']['label']} (score: {a['political_bias']['score']})
- Bias Reason: {a['political_bias']['reason']}
- Emotional Language: {a['emotional_language']['label']} (score: {a['emotional_language']['score']})
- Examples: {a['emotional_language']['examples']}
- Missing Context: {a['missing_context']['label']} — {a['missing_context']['what_is_missing']}
- Credibility: {a['credibility']['label']} (score: {a['credibility']['score']})
- Summary: {a['summary']}
---"""

    prompt = f"""You are a senior media analyst writing for an academic journal on press freedom and media literacy.
You have analyzed the same news event across {len(all_analyses)} different outlets.

{sources_summary}

Write a professional media analysis report. Rules:
- Write like a human analyst, not an AI
- Use varied sentence lengths
- Avoid phrases like "it is important to note" or "in conclusion"
- Be direct and specific, name the sources
- Use dry precise language like The Economist
- Pure flowing prose only, no bullet points, no bold, no markdown

Sections (plain text headings):

STORY OVERVIEW
2-3 sentences of pure facts only.

BIAS ACROSS SOURCES
Compare how each outlet framed the story. Be specific — name sources, reference actual scores, interpret what the bias means in context.

EMOTIONAL LANGUAGE PATTERNS
Which outlets used charged language? Pull specific examples. Explain the effect on reader perception.

CONTEXT GAPS
What did ALL sources collectively fail to tell the reader?

MOST RELIABLE COVERAGE
Name one source and explain specifically why.

FINAL VERDICT
3-4 sentences. What should an informed reader take away?"""

    response = llm.invoke(prompt)
    return response.content

# ── BUILD DASHBOARD ──
def build_dashboard(all_analyses: list):
    sources = [clean_source_name(a["source"]) for a in all_analyses]
    bias_scores = [a["political_bias"]["score"] for a in all_analyses]
    bias_labels = [a["political_bias"]["label"] for a in all_analyses]
    emotion_scores = [a["emotional_language"]["score"] for a in all_analyses]
    credibility_scores = [a["credibility"]["score"] for a in all_analyses]
    missing_scores = [a["missing_context"]["score"] for a in all_analyses]

    def bias_color(score):
        if score <= -6:   return "#1a53ff"
        elif score <= -3: return "#4d94ff"
        elif score <= -1: return "#99c2ff"
        elif score == 0:  return "#95a5a6"
        elif score <= 2:  return "#ffb3b3"
        elif score <= 5:  return "#ff4d4d"
        else:             return "#cc0000"

    bias_colors = [bias_color(s) for s in bias_scores]

    overall_scores = [
        round(credibility_scores[i]*0.5 + (10-emotion_scores[i])*0.25 + (10-missing_scores[i])*0.25, 1)
        for i in range(len(sources))
    ]
    sorted_idx = sorted(range(len(overall_scores)), key=lambda i: overall_scores[i], reverse=True)

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Political Bias Score", "Credibility Score",
            "Emotional Language Score", "Missing Context Score",
            "Source Comparison Radar", "Overall Score Summary"
        ),
        specs=[
            [{"type": "bar"},   {"type": "bar"}],
            [{"type": "bar"},   {"type": "bar"}],
            [{"type": "polar"}, {"type": "table"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.14,
        row_heights=[0.25, 0.25, 0.50]
    )

    # Bias
    fig.add_trace(go.Bar(
        x=sources, y=bias_scores, marker_color=bias_colors,
        marker_line_color="rgba(0,0,0,0.3)", marker_line_width=1,
        text=[f"{s:+d}" for s in bias_scores], textposition="outside",
        name="Bias", hovertemplate="<b>%{x}</b><br>Bias: %{y:+d}<extra></extra>"
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#2c3e50", line_width=1.5, row=1, col=1)

    # Credibility
    fig.add_trace(go.Bar(
        x=sources, y=credibility_scores,
        marker_color=["#27ae60" if s>=8 else "#f39c12" if s>=6 else "#e74c3c" for s in credibility_scores],
        marker_line_color="rgba(0,0,0,0.3)", marker_line_width=1,
        text=[f"{s}/10" for s in credibility_scores], textposition="outside",
        name="Credibility", hovertemplate="<b>%{x}</b><br>Credibility: %{y}/10<extra></extra>"
    ), row=1, col=2)

    # Emotion
    fig.add_trace(go.Bar(
        x=sources, y=emotion_scores,
        marker_color=["#e74c3c" if s>=7 else "#f39c12" if s>=4 else "#27ae60" for s in emotion_scores],
        marker_line_color="rgba(0,0,0,0.3)", marker_line_width=1,
        text=[f"{s}/10" for s in emotion_scores], textposition="outside",
        name="Emotion", hovertemplate="<b>%{x}</b><br>Emotion: %{y}/10<extra></extra>"
    ), row=2, col=1)

    # Missing Context
    fig.add_trace(go.Bar(
        x=sources, y=missing_scores,
        marker_color=["#e74c3c" if s>=7 else "#f39c12" if s>=4 else "#27ae60" for s in missing_scores],
        marker_line_color="rgba(0,0,0,0.3)", marker_line_width=1,
        text=[f"{s}/10" for s in missing_scores], textposition="outside",
        name="Context", hovertemplate="<b>%{x}</b><br>Context: %{y}/10<extra></extra>"
    ), row=2, col=2)

    # Radar
    categories = ["Credibility", "Emotional Language", "Missing Context", "Bias Intensity", "Credibility"]
    radar_colors = px.colors.qualitative.Set2
    for i, source in enumerate(sources):
        fig.add_trace(go.Scatterpolar(
            r=[credibility_scores[i], emotion_scores[i], missing_scores[i], abs(bias_scores[i]), credibility_scores[i]],
            theta=categories, fill="toself", name=source,
            line_color=radar_colors[i % len(radar_colors)],
            fillcolor=radar_colors[i % len(radar_colors)],
            opacity=0.35,
            hovertemplate=f"<b>{source}</b><br>%{{theta}}: %{{r}}<extra></extra>"
        ), row=3, col=1)

    # Table
    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Source</b>", "<b>Bias</b>", "<b>Credibility</b>", "<b>Emotion</b>", "<b>Overall</b>"],
            fill_color="#2c3e50", font=dict(color="white", size=12),
            align="center", height=32
        ),
        cells=dict(
            values=[
                [sources[i] for i in sorted_idx],
                [bias_labels[i] for i in sorted_idx],
                [f"{credibility_scores[i]}/10" for i in sorted_idx],
                [f"{emotion_scores[i]}/10" for i in sorted_idx],
                [f"{overall_scores[i]}/10" for i in sorted_idx]
            ],
            fill_color=[
                ["#f8f9fa"]*len(sources),
                ["#f8f9fa"]*len(sources),
                ["#f8f9fa"]*len(sources),
                ["#f8f9fa"]*len(sources),
                ["#d5f5e3" if overall_scores[i]>=7.5 else "#fef9e7" if overall_scores[i]>=6 else "#fadbd8" for i in sorted_idx]
            ],
            font=dict(color="#2c3e50", size=11),
            align="center", height=30
        )
    ), row=3, col=2)

    fig.update_layout(
        title={"text": "News Bias Analysis Dashboard", "x": 0.5, "xanchor": "center",
               "font": {"size": 24, "family": "Georgia", "color": "#2c3e50"}},
        height=1300, showlegend=False,
        paper_bgcolor="#f0f3f4", plot_bgcolor="#ffffff",
        font={"family": "Arial", "size": 11, "color": "#2c3e50"},
        margin={"t": 80, "b": 50, "l": 60, "r": 60}
    )
    fig.update_xaxes(tickangle=25, tickfont={"size": 11}, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#ecf0f1", gridwidth=1)
    fig.update_yaxes(range=[-12, 12], row=1, col=1)
    fig.update_yaxes(range=[0, 13], row=1, col=2)
    fig.update_yaxes(range=[0, 13], row=2, col=1)
    fig.update_yaxes(range=[0, 13], row=2, col=2)
    fig.update_polars(
        radialaxis=dict(visible=True, range=[0, 10], tickfont={"size": 9}, tickvals=[2,4,6,8,10]),
        angularaxis=dict(tickfont={"size": 11}, rotation=90),
        bgcolor="#ffffff",
        domain={"x": [0.0, 0.46], "y": [0.0, 0.42]}
    )
    fig.add_annotation(
        text="Bias: −10 (Far Left) to +10 (Far Right) | All other scores: 0–10 | Overall = 50% Credibility + 25% Low Emotion + 25% Low Context Gaps",
        xref="paper", yref="paper", x=0.5, y=-0.04,
        showarrow=False, font={"size": 9, "color": "#7f8c8d"}, xanchor="center"
    )

    return fig

# ── MAIN APP ──
st.markdown("### Paste a news article URL to begin")

url_input = st.text_input(
    label="News Article URL",
    placeholder="https://www.bbc.com/news/articles/...",
    label_visibility="collapsed"
)

analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

if analyze_btn:
    # Validate inputs
    if not groq_key or not tavily_key:
        st.error("Please enter your Groq and Tavily API keys in the sidebar first.")
        st.stop()

    if not url_input or not url_input.startswith("http"):
        st.error("Please enter a valid URL starting with http:// or https://")
        st.stop()

    # Initialize clients
    try:
        llm, tavily = init_clients(groq_key, tavily_key)
    except Exception as e:
        st.error(f"Could not connect to API: {str(e)}")
        st.stop()

    # Step 1: Scrape
    with st.status("🔍 Scraping article...", expanded=True) as status:
        st.write("Fetching article from URL...")
        article = scrape_article(url_input)

        if "error" in article:
            st.error(f"Could not scrape article: {article['error']}")
            st.stop()

        st.write(f"✅ Found: **{article['title']}**")
        st.write(f"📰 Source: **{article['source']}**")

        # Step 2: Search
        st.write("🌐 Searching web for related articles...")
        all_analyses, search_query = search_and_scrape(article, llm, tavily, num_sources)
        st.write(f"✅ Analyzed **{len(all_analyses)}** sources for: *{search_query}*")

        # Step 3: Report
        st.write("🧠 Generating media analysis report...")
        report = generate_report(all_analyses, llm)
        st.write("✅ Report generated!")

        status.update(label="✅ Analysis complete!", state="complete")

    st.markdown("---")

    # ── RESULTS: METRICS ROW ──
    st.markdown("### 📊 Quick Summary")
    original = all_analyses[0]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Political Bias",
            value=original['political_bias']['label'],
            delta=f"Score: {original['political_bias']['score']:+d}"
        )
    with col2:
        st.metric(
            label="Credibility",
            value=f"{original['credibility']['score']}/10",
            delta=original['credibility']['label']
        )
    with col3:
        st.metric(
            label="Emotional Language",
            value=f"{original['emotional_language']['score']}/10",
            delta=original['emotional_language']['label']
        )
    with col4:
        st.metric(
            label="Missing Context",
            value=f"{original['missing_context']['score']}/10",
            delta=original['missing_context']['label']
        )

    st.markdown("---")

    # ── RESULTS: TABS ──
    tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "📝 Analysis Report", "🗞️ Sources"])

    with tab1:
        st.plotly_chart(build_dashboard(all_analyses), use_container_width=True)

    with tab2:
        st.markdown("### Media Analysis Report")
        st.markdown(f'<div class="report-box">{report.replace(chr(10), "<br>")}</div>',
                   unsafe_allow_html=True)

    with tab3:
        st.markdown("### Sources Analyzed")
        for i, a in enumerate(all_analyses):
            with st.expander(f"{'🔵' if i==0 else '⚪'} {clean_source_name(a['source'])} — {a['title'][:70]}..."):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Political Bias:** {a['political_bias']['label']} ({a['political_bias']['score']:+d})")
                    st.markdown(f"**Evidence:** {', '.join(a['political_bias'].get('evidence', []))}")
                    st.markdown(f"**Credibility:** {a['credibility']['label']} ({a['credibility']['score']}/10)")
                    st.markdown(f"**Credibility Reason:** {a['credibility']['reason']}")
                with col2:
                    st.markdown(f"**Emotional Language:** {a['emotional_language']['label']} ({a['emotional_language']['score']}/10)")
                    st.markdown(f"**Emotional Examples:** {', '.join(a['emotional_language'].get('examples', []))}")
                    st.markdown(f"**Missing Context:** {a['missing_context']['label']}")
                    st.markdown(f"**What is Missing:** {a['missing_context']['what_is_missing']}")
            st.markdown(f"**Summary:** {a['summary']}")
            st.markdown(f"[Read Original Article]({a['url']})")

# ── FOOTER ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#7f8c8d; font-size:0.85rem;'>"
    "News Bias Detector | Built with LangChain, Groq, Tavily & Streamlit"
    "</div>",
    unsafe_allow_html=True
)