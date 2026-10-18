"""
Streamlit Frontend for AI Job Description–Driven Assessment Blueprint Builder
Connects to FastAPI Backend Service (http://127.0.0.1:8000) with automatic local fallback.
Conforms to TCS Capstone Project Guidelines:
- Frontend: Streamlit
- Backend APIs: FastAPI
"""

import os
import re
import json
import time
import requests
import pandas as pd
import streamlit as st

import core

APP_TITLE = "AI Job Description–Driven Assessment Blueprint Builder"
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


# ---------------------------------------------------------
# FastAPI Client Integration Wrappers
# ---------------------------------------------------------
def get_api_health():
    try:
        r = requests.get(f"{API_BASE_URL}/api/health", timeout=1.5)
        return r.ok, r.json() if r.ok else {}
    except Exception:
        return False, {}


def client_extract_jd(jd_text, use_ai, api_key, model):
    is_up, _ = get_api_health()
    if is_up:
        try:
            r = requests.post(f"{API_BASE_URL}/api/extract-jd", json={
                "jd_text": jd_text, "use_ai": use_ai, "api_key": api_key, "model": model
            }, timeout=90)
            if r.ok:
                return r.json(), "FastAPI Backend"
        except Exception:
            pass
    res = core.ai_based_extraction(jd_text, api_key, model) if (use_ai and api_key) else core.rule_based_extraction(jd_text)
    return res, "Local Core Engine"


def client_build_blueprint(competencies, seniority, total_questions, duration, easy_pct, medium_pct, hard_pct):
    is_up, _ = get_api_health()
    if is_up:
        try:
            r = requests.post(f"{API_BASE_URL}/api/build-blueprint", json={
                "competencies": competencies, "seniority": seniority, "total_questions": total_questions,
                "duration": duration, "easy_pct": easy_pct, "medium_pct": medium_pct, "hard_pct": hard_pct
            }, timeout=30)
            if r.ok:
                data = r.json()
                return data["blueprint"], data["warnings"], "FastAPI Backend"
        except Exception:
            pass
    b, w = core.build_adaptive_blueprint(competencies, seniority, total_questions, duration, easy_pct, medium_pct, hard_pct)
    return b, w, "Local Core Engine"


def client_generate_question(competency, jd_text, difficulty, api_key, model):
    is_up, _ = get_api_health()
    if is_up:
        try:
            r = requests.post(f"{API_BASE_URL}/api/generate-question", json={
                "competency": competency, "difficulty": difficulty, "jd_text": jd_text,
                "api_key": api_key, "model": model
            }, timeout=90)
            if r.ok:
                return r.json()
        except Exception:
            pass
    return core.generate_gap_question(competency, jd_text, difficulty, api_key, model)


def client_simulate_candidates(blueprint, duration):
    is_up, _ = get_api_health()
    if is_up:
        try:
            r = requests.post(f"{API_BASE_URL}/api/simulate-candidates", json={
                "blueprint": blueprint, "duration": duration
            }, timeout=10)
            if r.ok:
                return r.json()
        except Exception:
            pass
    return core.run_candidate_simulation(blueprint, duration)


def client_highlight_evidence(jd_text, competencies):
    is_up, _ = get_api_health()
    if is_up:
        try:
            r = requests.post(f"{API_BASE_URL}/api/highlight-evidence", json={
                "jd_text": jd_text, "competencies": competencies
            }, timeout=10)
            if r.ok:
                return r.json()["highlighted_html"]
        except Exception:
            pass
    return core.highlight_jd_text(jd_text, competencies)


# ---------------------------------------------------------
# Streamlit Application Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.1rem; font-weight: 800; color: #1e293b; margin-bottom: 0.2rem; }
    .sub-caption { font-size: 1.0rem; color: #64748b; margin-bottom: 0.8rem; }
    .api-badge-ok { background-color: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; border: 1px solid #86efac; display: inline-block; margin-bottom: 12px; }
    .api-badge-warn { background-color: #fef9c3; color: #854d0e; padding: 4px 12px; border-radius: 12px; font-weight: 600; font-size: 0.85rem; border: 1px solid #fde047; display: inline-block; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="main-header">🎯 {APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">Enterprise Multi-Agent Assessment Pipeline • Streamlit UI + FastAPI Backend Architecture</div>', unsafe_allow_html=True)

# Backend status indicator
backend_online, backend_meta = get_api_health()
if backend_online:
    st.markdown(
        f'<div class="api-badge-ok">🟢 FastAPI Backend Connected: <a href="{API_BASE_URL}/docs" target="_blank" style="color: #166534; font-weight:700; text-decoration: underline;">{API_BASE_URL} (OpenAPI Swagger Docs)</a></div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="api-badge-warn">🟡 Running in Standalone Core Mode (FastAPI backend offline)</div>',
        unsafe_allow_html=True
    )

# Session State Initialization
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "competencies" not in st.session_state:
    st.session_state.competencies = []
if "blueprint" not in st.session_state:
    st.session_state.blueprint = []
if "warnings" not in st.session_state:
    st.session_state.warnings = []
if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = []
if "audit_logs" not in st.session_state:
    st.session_state.audit_logs = []

def log_audit(event):
    st.session_state.audit_logs.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event
    })


# ---------------------------------------------------------
# Sidebar Configuration & Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings & Agent Controls")
    
    env_key = os.getenv("OPENAI_API_KEY", "")
    has_env_key = bool(env_key and env_key.strip() and env_key != "your_openai_api_key_here")
    
    ai_toggle = st.toggle("Enable AI Agent Mode", value=has_env_key, help="Uses OpenAI for deep extraction and gap question synthesis.")
    api_key_input = ""
    model_choice = "gpt-4o-mini"
    
    if ai_toggle:
        if has_env_key:
            st.caption("🔒 Key auto-loaded from `.env`")
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=env_key if env_key != "your_openai_api_key_here" else "",
            type="password",
            help="Your key is kept only in session memory."
        )
        model_choice = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
        
        # Test API Key Button
        if st.button("🔑 Test API Key Connection", use_container_width=True):
            with st.spinner("Testing OpenAI connection..."):
                is_valid, msg = core.test_openai_api_key(api_key_input, model_choice)
                if is_valid:
                    st.success(f"✅ {msg}")
                    log_audit(f"OpenAI API Key test succeeded with model {model_choice}.")
                else:
                    st.error(f"❌ {msg}")
                    log_audit(f"OpenAI API Key test failed: {msg}")

    st.divider()
    st.markdown("### 🏛️ System Architecture")
    st.markdown(f"""
    - **Frontend**: Streamlit (`port 8501`)
    - **Backend**: FastAPI (`{API_BASE_URL}`)
    - **API Docs**: [Swagger UI]({API_BASE_URL}/docs)
    """)
    st.markdown("### 🤖 5 Orchestrated Agent Roles")
    st.markdown("""
    1. **JD Analysis Agent**: Parses & extracts skills, tools, seniority
    2. **Competency Mapping Agent**: Taxonomy normalization & ambiguity resolution
    3. **Blueprint Design Agent**: Adaptive weight & difficulty allocation
    4. **Question Retrieval Agent**: Multi-tag lookup & gap question synthesis
    5. **Quality Review Agent**: Candidate simulation & bias/overlap audit
    """)
    st.divider()
    st.caption("TCS Capstone Assessment Blueprint Architecture.")


# ---------------------------------------------------------
# Main Tabs (5 Stages)
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. 🤖 JD Analysis & Evidence Map",
    "2. 🧭 Competency Mapping & Review",
    "3. 📐 Adaptive Blueprint",
    "4. 🔍 Questions & Gap Generator",
    "5. 🛡️ Candidate Simulation & Audit"
])


# ---------------------------------------------------------
# TAB 1: Job Description Ingestion & Parsing
# ---------------------------------------------------------
with tab1:
    st.subheader("Stage 1: Job Description Ingestion & Analysis Agent")
    
    c_up1, c_up2 = st.columns([1, 1])
    with c_up1:
        uploaded_doc = st.file_uploader("Upload Job Description (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
        if uploaded_doc:
            try:
                parsed_text = core.extract_text_from_bytes(uploaded_doc.name, uploaded_doc.getvalue())
                if parsed_text.strip():
                    st.session_state.jd_text = parsed_text
                    st.success(f"Successfully extracted {len(parsed_text)} characters from {uploaded_doc.name}")
                    log_audit(f"Parsed uploaded file: {uploaded_doc.name}")
            except Exception as err:
                st.error(f"Failed to read file: {err}")

    with c_up2:
        default_sample = (
            "Senior Backend Engineer\n\n"
            "Responsibilities:\n"
            "- Design and implement scalable REST APIs and microservices in Python (FastAPI/Django).\n"
            "- Optimize PostgreSQL databases, queries, and complex joins.\n"
            "- Deploy containerized workloads using Docker and Kubernetes on AWS.\n"
            "- Lead system design discussions for high-throughput distributed systems.\n"
            "- Mentor junior engineers, conduct code reviews, and champion unit testing with Pytest.\n\n"
            "Requirements:\n"
            "- 5+ years experience in Python and SQL.\n"
            "- Must have strong problem solving and data structures background.\n"
            "- Mandatory experience in Git collaboration, CI/CD, and secure OAuth2 authentication."
        )
        if st.button("Load Sample Senior Backend JD"):
            st.session_state.jd_text = default_sample

    jd_editor = st.text_area(
        "Job Description Content:",
        value=st.session_state.jd_text,
        height=220,
        placeholder="Paste JD text here..."
    )

    if st.button("🚀 Run Analysis Agent", type="primary", use_container_width=True):
        st.session_state.jd_text = jd_editor
        if not jd_editor.strip():
            st.error("Please provide a job description first.")
        else:
            with st.spinner("Analyzing Job Description via FastAPI Agent..."):
                try:
                    analysis, engine_name = client_extract_jd(jd_editor, ai_toggle, api_key_input, model_choice)
                    log_audit(f"Executed JD extraction via {engine_name}.")
                    st.session_state.analysis = analysis
                    st.session_state.competencies = analysis.get("competencies", [])
                    st.session_state.blueprint = []
                    st.success(f"Extracted {len(st.session_state.competencies)} competencies across role: '{analysis.get('role_title', 'Engineer')}' (Processed via {engine_name})")
                except Exception as ex:
                    st.error(f"Analysis failed: {ex}")

    if st.session_state.analysis:
        st.divider()
        st.markdown("### 📊 Extracted Role Summary")
        a = st.session_state.analysis
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Identified Title", a.get("role_title", "Unknown"))
        m2.metric("Detected Seniority", a.get("seniority", "Mid"))
        m3.metric("Competencies Found", len(st.session_state.competencies))

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("#### 🎯 Key Responsibilities")
            for r in a.get("responsibilities", []):
                st.markdown(f"- {r}")
            st.markdown("#### 🛠️ Tools & Technologies")
            st.write(", ".join(a.get("tools", [])) if a.get("tools") else "None explicitly isolated.")

        with col_right:
            st.markdown("#### 🤝 Observable Behaviors")
            for b in a.get("observable_behaviors", []):
                st.markdown(f"- {b}")
            if a.get("ambiguities"):
                st.warning("⚠️ **Ambiguities / Questions for Reviewer:**\n" + "\n".join(f"- {amb}" for amb in a["ambiguities"]))

        # Innovative Workflow #1: Interactive JD Evidence Map
        st.divider()
        st.markdown("### 🗺️ Innovative Workflow 1: JD Evidence Map")
        st.caption("Visual phrase highlighting mapping detected skills to raw text context:")
        highlighted_html = client_highlight_evidence(st.session_state.jd_text, st.session_state.competencies)
        st.markdown(
            f'<div style="background-color: #f1f5f9; padding: 16px; border-radius: 8px; border: 1px solid #cbd5e1; line-height: 1.8; max-height: 300px; overflow-y: auto;">{highlighted_html}</div>',
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# TAB 2: Competency Mapping & Reviewer Confirmation
# ---------------------------------------------------------
with tab2:
    st.subheader("Stage 2: Competency Normalization & Human-in-the-Loop Review Agent")
    st.caption("Review extracted competencies, confirm mandatory flags, and adjust proficiency requirements.")

    if not st.session_state.competencies:
        st.info("Run the JD Analysis Agent in Tab 1 to populate competencies.")
    else:
        edited_comps = st.data_editor(
            st.session_state.competencies,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "mandatory": st.column_config.CheckboxColumn("Mandatory?", default=False),
                "proficiency": st.column_config.SelectboxColumn("Proficiency", options=["Beginner", "Intermediate", "Advanced"]),
                "confidence": st.column_config.ProgressColumn("Confidence Score", min_value=0.0, max_value=1.0, format="%.2f"),
                "importance": st.column_config.NumberColumn("Importance (1-5)", min_value=1, max_value=5),
            },
            key="competency_data_editor"
        )
        st.session_state.competencies = edited_comps

        if st.button("💾 Confirm Competency Mapping"):
            log_audit("Reviewer confirmed competency taxonomy mapping.")
            st.success("Competency mapping confirmed. Proceed to Tab 3 to configure the blueprint.")


# ---------------------------------------------------------
# TAB 3: Adaptive Blueprint Design Agent
# ---------------------------------------------------------
with tab3:
    st.subheader("Stage 3: Adaptive Blueprint Design Agent")
    st.caption("Configures assessment weight, duration, and dynamically adapts difficulty mix to role seniority.")

    if not st.session_state.competencies:
        st.info("Analyze a JD first to configure its blueprint.")
    else:
        detected_seniority = st.session_state.analysis.get("seniority", "Mid") if st.session_state.analysis else "Mid"
        
        c_cfg1, c_cfg2, c_cfg3 = st.columns(3)
        with c_cfg1:
            selected_seniority = st.selectbox(
                "Role Seniority",
                ["Entry", "Junior", "Mid", "Senior", "Lead"],
                index=["Entry", "Junior", "Mid", "Senior", "Lead"].index(detected_seniority)
            )
        with c_cfg2:
            target_q_count = st.number_input("Total Questions", min_value=2, max_value=50, value=10)
        with c_cfg3:
            target_duration = st.number_input("Target Duration (minutes)", min_value=10, max_value=180, value=60)

        # Innovative Workflow #2: Adaptive Blueprint Difficulty Mix
        st.markdown("#### 🎚️ Seniority-Adaptive Difficulty Allocation")
        recommended_mix = core.get_adaptive_difficulty_mix(selected_seniority)
        
        c_d1, c_d2, c_d3, c_d4 = st.columns([1, 1, 1, 1])
        with c_d1:
            easy_val = st.number_input("Easy %", min_value=0, max_value=100, value=recommended_mix["Easy"])
        with c_d2:
            med_val = st.number_input("Medium %", min_value=0, max_value=100, value=recommended_mix["Medium"])
        with c_d3:
            hard_val = st.number_input("Hard %", min_value=0, max_value=100, value=recommended_mix["Hard"])
        with c_d4:
            st.write("")
            st.write("")
            if st.button("Reset to Adaptive Seniority Mix"):
                st.info(f"Recommended for {selected_seniority}: Easy {recommended_mix['Easy']}%, Med {recommended_mix['Medium']}%, Hard {recommended_mix['Hard']}%")

        if st.button("⚡ Generate Assessment Blueprint", type="primary", use_container_width=True):
            blueprint, warnings, engine_used = client_build_blueprint(
                st.session_state.competencies,
                selected_seniority,
                int(target_q_count),
                int(target_duration),
                int(easy_val),
                int(med_val),
                int(hard_val)
            )
            st.session_state.blueprint = blueprint
            st.session_state.warnings = warnings
            log_audit(f"Generated blueprint via {engine_used} ({target_q_count} questions for {selected_seniority} role).")

        if st.session_state.blueprint:
            st.divider()
            st.markdown("### 📋 Generated Assessment Blueprint Preview")
            
            preview_rows = []
            for b in st.session_state.blueprint:
                preview_rows.append({
                    "Competency": b["competency"],
                    "Category": b["category"],
                    "Mandatory": b["mandatory"],
                    "Weight (%)": b["weight_pct"],
                    "Allocated Questions": b["allocated_questions"],
                    "Questions Retrieved": len(b["selected_questions"]),
                    "Coverage Status": "Full" if len(b["selected_questions"]) >= b["allocated_questions"] else "Gap Detected"
                })
            
            st.dataframe(preview_rows, use_container_width=True, hide_index=True)

            if st.session_state.warnings:
                for w in st.session_state.warnings:
                    st.warning(f"⚠️ {w}")

            export_data = {
                "role": selected_seniority,
                "duration_minutes": target_duration,
                "blueprint": st.session_state.blueprint,
                "warnings": st.session_state.warnings,
            }
            st.download_button(
                "📥 Download Blueprint Specification (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name="assessment_blueprint.json",
                mime="application/json"
            )


# ---------------------------------------------------------
# TAB 4: Questions & AI Gap Generator
# ---------------------------------------------------------
with tab4:
    st.subheader("Stage 4: Question Retrieval, Tagging & Gap Synthesis Agent")
    st.caption("Inspect retrieved questions with full multi-tagging (Subskill, Cognitive Level, Job Relevance). Fill gaps with AI scenario questions.")

    if not st.session_state.blueprint:
        st.info("Generate a blueprint in Tab 3 to view questions.")
    else:
        st.markdown("### 📚 Assigned Assessment Questions")
        total_time = 0
        for b in st.session_state.blueprint:
            with st.expander(f"🧩 {b['competency']} ({b['weight_pct']}% Weight • {len(b['selected_questions'])}/{b['allocated_questions']} Questions)"):
                if not b["selected_questions"]:
                    st.error("No suitable question found in local bank! Use the AI Gap Generator below.")
                for i, q in enumerate(b["selected_questions"], 1):
                    total_time += q.get("minutes", 5)
                    st.markdown(f"**{i}. {q['question']}**")
                    st.caption(
                        f"🏷️ **Subskill:** {q.get('subskill', 'General')} | "
                        f"📊 **Difficulty:** {q.get('difficulty')} | "
                        f"🧠 **Cognitive Level:** {q.get('cognitive_level', 'Apply')} | "
                        f"⏱️ **Time:** {q.get('minutes')} mins | "
                        f"🎯 **Relevance:** {q.get('job_relevance', 'Direct')}"
                    )
                    st.markdown(f"*Expected Answer:* {q.get('expected', '')}")
                    st.markdown(f"*Scoring Rubric:* {q.get('rubric', 'Standard 10-point scale')}")
                    st.divider()

        st.markdown(f"**Total Estimated Assessment Time:** `{total_time}` minutes")

        # Gap Synthesis Module
        st.divider()
        st.markdown("### 💡 Fill Coverage Gaps: Scenario Question Generator")
        c_gap1, c_gap2 = st.columns(2)
        with c_gap1:
            all_comp_names = [b["competency"] for b in st.session_state.blueprint]
            gap_comp = st.selectbox("Select Competency to Generate Question For", all_comp_names)
        with c_gap2:
            gap_diff = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"], index=1)

        can_generate = bool(ai_toggle and api_key_input)
        if st.button("✨ Synthesize Scenario Question with AI", disabled=not can_generate):
            with st.spinner("Generating work-sample scenario question via FastAPI..."):
                try:
                    new_q = client_generate_question(gap_comp, st.session_state.jd_text, gap_diff, api_key_input, model_choice)
                    st.session_state.generated_questions.append(new_q)
                    log_audit(f"Generated new draft question for {gap_comp} ({gap_diff}).")
                    st.success("Draft question created. Human review is required before publication.")
                except Exception as ex:
                    st.error(f"Generation failed: {ex}")

        if not can_generate:
            st.caption("ℹ️ Enable AI Agent Mode and provide an OpenAI API Key in the sidebar to generate new questions.")

        if st.session_state.generated_questions:
            st.markdown("#### 📝 Pending Draft Questions for Human Review")
            for idx, gq in enumerate(st.session_state.generated_questions):
                with st.expander(f"Draft #{idx+1}: {gq.get('competency')} ({gq.get('difficulty')}) — {gq.get('review_status')}"):
                    st.write(gq.get("question"))
                    st.write("**Subskill:**", gq.get("subskill"))
                    st.write("**Expected Answer:**", gq.get("expected"))
                    st.write("**Rubric:**", gq.get("rubric"))
                    col_app1, col_app2 = st.columns(2)
                    with col_app1:
                        if st.button(f"✅ Approve & Add to Blueprint #{idx+1}", key=f"app_{idx}"):
                            for b in st.session_state.blueprint:
                                if b["competency"] == gq["competency"]:
                                    b["selected_questions"].append(gq)
                                    break
                            gq["review_status"] = "Approved by Reviewer"
                            log_audit(f"Approved draft question #{idx+1} for {gq['competency']}.")
                            st.rerun()


# ---------------------------------------------------------
# TAB 5: Candidate Simulation & Quality Review Agent
# ---------------------------------------------------------
with tab5:
    st.subheader("Stage 5: Quality Review, Coverage Heatmap & Candidate Simulation Agent")
    st.caption("Validate duration constraints, coverage heatmaps, and run synthetic candidate profiles to verify proficiency separation.")

    if not st.session_state.blueprint:
        st.info("Generate a blueprint in Tab 3 to review assessment quality.")
    else:
        # Innovative Workflow #3: Visual Coverage Heatmap
        st.markdown("### 🌡️ Innovative Workflow 3: Competency Coverage Heatmap")
        
        heatmap_cols = st.columns(len(st.session_state.blueprint))
        for col, b in zip(heatmap_cols, st.session_state.blueprint):
            found_count = len(b["selected_questions"])
            target_count = b["allocated_questions"]
            
            if found_count == 0:
                status_color = "#fecaca"  # Red (Missing)
                status_label = "Missing"
            elif found_count < target_count:
                status_color = "#fef08a"  # Amber (Under-assessed)
                status_label = "Under-assessed"
            else:
                status_color = "#bbf7d0"  # Green (Assessed)
                status_label = "Assessed"

            with col:
                st.markdown(
                    f"""
                    <div style="background-color: {status_color}; border-radius: 8px; padding: 10px; text-align: center; border: 1px solid #cbd5e1; min-height: 110px;">
                        <div style="font-weight: 700; font-size: 0.85rem;">{b['competency']}</div>
                        <div style="font-size: 1.2rem; font-weight: 800; margin: 5px 0;">{found_count}/{target_count}</div>
                        <span style="font-size: 0.75rem; font-weight: 600;">{status_label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Innovative Workflow #5: Synthetic Candidate Simulation
        st.divider()
        st.markdown("### 🧪 Innovative Workflow 5: Synthetic Candidate Simulation Engine")
        st.caption("Simulates real-world testing dynamics across distinct proficiency tiers to verify duration and scoring spread.")

        sim_button = st.button("🎲 Run Candidate Simulation", type="primary")
        if sim_button or "sim_data" in st.session_state:
            if sim_button:
                sim_res = client_simulate_candidates(st.session_state.blueprint, duration=60)
                st.session_state.sim_data = sim_res
                log_audit("Ran synthetic candidate simulation across 3 candidate profiles.")
            
            sim_data = st.session_state.get("sim_data")
            if sim_data:
                res_df = pd.DataFrame(sim_data["results"])
                st.dataframe(res_df, use_container_width=True, hide_index=True)

                c_idx1, c_idx2 = st.columns(2)
                with c_idx1:
                    st.metric("Proficiency Separation Index", f"{sim_data['separation_index']}%")
                with c_idx2:
                    if sim_data["is_discriminating"]:
                        st.success("✅ Assessment Quality Passed: Effective score discrimination between Senior and Junior tiers (>35% spread).")
                    else:
                        st.warning("⚠️ Insufficient Separation: The difficulty mix does not adequately differentiate senior and junior candidates.")

        # Explainability & Audit Log
        st.divider()
        st.markdown("### 🔍 Explainability Report & Audit Log")
        
        with st.expander("View Job Description Evidence Traces"):
            for b in st.session_state.blueprint:
                st.markdown(f"**{b['competency']} (Weight: {b['weight_pct']}%)**")
                if b["evidence"]:
                    for ev in b["evidence"]:
                        st.markdown(f"- *\"{ev}\"*")
                else:
                    st.caption("Inferred from role title and related technology stack.")
        
        with st.expander("View Agent Orchestration Audit Logs"):
            if st.session_state.audit_logs:
                st.dataframe(pd.DataFrame(st.session_state.audit_logs), use_container_width=True, hide_index=True)
            else:
                st.write("No audit events recorded yet.")

        # Final Assessment Export
        full_assessment_export = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "app": APP_TITLE,
                "backend": API_BASE_URL,
            },
            "blueprint": st.session_state.blueprint,
            "audit_logs": st.session_state.audit_logs,
        }
        st.download_button(
            "📦 Export Complete Assessment Package (JSON)",
            data=json.dumps(full_assessment_export, indent=2),
            file_name="complete_assessment_package.json",
            mime="application/json"
        )

st.divider()
st.caption("AI Job Description–Driven Assessment Blueprint Builder • TCS Capstone Architecture")
