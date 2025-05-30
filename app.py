import streamlit as st
import pdfminer.high_level as pdfminer
import spacy
import re
import json
import plotly.graph_objects as go
import language_tool_python
from sentence_transformers import SentenceTransformer, util
from pdfminer.high_level import extract_text


#  Page Config
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

#  Load Resources
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_grammar_tool():
    return language_tool_python.LanguageTool("en-US")

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

nlp = load_nlp()
tool = load_grammar_tool()
sbert = load_semantic_model()

# Load External Data
@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

ATS_KEYWORDS = load_json("ats_keywords.json")
CERTIFICATIONS = load_json("certifications.json")

# PDF Text Extraction
@st.cache_data
def extract_text_from_pdf(pdf):
    return extract_text(pdf)

# Info Extraction
def extract_email(text):
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group() if match else "Not Found"

def extract_phone(text):
    match = re.search(r"\b\d{10}\b", text)
    return match.group() if match else "Not Found"

def extract_name(text):
    return text.split("\n")[0].strip()

# Extract Skills
@st.cache_data
def extract_skills(text):
    text_lower = text.lower()
    return list({kw for kws in ATS_KEYWORDS.values() for kw in kws if kw.lower() in text_lower})

def extract_skills_from_jd(text):
    text_lower = text.lower()
    return list({kw for kws in ATS_KEYWORDS.values() for kw in kws if kw.lower() in text_lower})

# Grammar Check
def check_grammar(text):
    doc = nlp(text)
    ignore = {ent.text for ent in doc.ents}
    ignore.update(["Python", "JavaScript", "SQL", "TensorFlow", "AWS", "Excel"])
    ignore_patterns = [r"\b\d{10}\b", r"[\w\.-]+@[\w\.-]+", r"\b\d+(\.\d+)?\b"]

    issues = []
    for match in tool.check(text):
        word = match.context.strip()
        if word in ignore or any(re.search(p, word) for p in ignore_patterns):
            continue
        if match.ruleId in ["EN_A_VS_AN", "SENTENCE_WHITESPACE", "UPPERCASE_SENTENCE_START"]:
            continue
        suggestion = match.replacements[0] if match.replacements else "No suggestion"
        issues.append(f"✗ {word} → ✅ {suggestion}")

    return len(issues), issues[:10]

# ATS Score Calculations
def rule_based_score(resume_text, jd_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills_from_jd(jd_text)
    if not job_skills:
        return 0, []
    match = len(set(resume_skills) & set(job_skills)) / len(job_skills)
    missing = list(set(job_skills) - set(resume_skills))
    return round(match * 100, 2), missing

def semantic_score(resume_text, jd_text):
    e1 = sbert.encode(resume_text, convert_to_tensor=True)
    e2 = sbert.encode(jd_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(e1, e2).item()
    return round(score * 100, 2)

def combined_score(rule_score, grammar_issues, semantic_score):
    grammar_penalty = len(grammar_issues) * 1.5
    return round((0.6 * rule_score + 0.4 * semantic_score - grammar_penalty), 2)

#  UI
st.title("🤖 AI Resume Analyzer - Hybrid ATS Optimizer")
st.subheader("📂 Upload Resume & Paste Job Description")
jd_input = st.text_area("📌 Paste Job Description")
resume_pdf = st.file_uploader("📂 Upload Resume (PDF)", type=["pdf"])

if resume_pdf and jd_input:
    with st.spinner("🔎 Analyzing Resume..."):
        resume_text = extract_text_from_pdf(resume_pdf)
        name = extract_name(resume_text)
        email = extract_email(resume_text)
        phone = extract_phone(resume_text)
        skills = extract_skills(resume_text)
        grammar_count, grammar_issues = check_grammar(resume_text)

        rule_score, missing_skills = rule_based_score(resume_text, jd_input)
        sem_score = semantic_score(resume_text, jd_input)
        final_score = combined_score(rule_score, grammar_issues, sem_score)

    # ATS Score
    st.plotly_chart(go.Figure(go.Indicator(
        mode="gauge+number",
        value=final_score,
        title={"text": "📊 Combined ATS Score"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "green"}},
    )))

    # Resume Details
    st.subheader("📄 Resume Overview")
    st.write(f"**Name:** {name}")
    st.write(f"**Email:** {email}")
    st.write(f"**Phone:** {phone}")
    st.write(f"**Skills:** {', '.join(skills)}")

    #  Key Metrics
    st.subheader("📌 Evaluation Summary")
    st.write(f"🔍 **Rule-Based Match Score:** {rule_score}%")
    st.write(f"🔬 **Semantic Match Score:** {sem_score}%")
    st.write(f"✒ **Grammar Errors:** {grammar_count}")

    #  Missing Skills
    if missing_skills:
        st.subheader("❌ Missing Skills")
        st.write(", ".join(missing_skills))

    #  Grammar Issues
    if grammar_issues:
        st.subheader("⚠ Grammar Issues")
        for issue in grammar_issues:
            st.write(issue)

    # Suggested Certifications
  
    st.subheader("🎓 Suggested Certifications")
    jd_skills = extract_skills_from_jd(jd_input)
    suggested = []

    for job_role, keywords in ATS_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in jd_input.lower():
                certs = CERTIFICATIONS.get(job_role, [])
                if certs:
                    suggested.extend(certs)
                    break  # avoid duplicates

    # Pick only 1 or 2 unique certifications
    suggested = list(set(suggested))
    if suggested:
        limited = suggested[:2]  # You can change 2 to 1 if you want only one suggestion
        for cert in limited:
            st.write(f"- {cert}")
    else:
        st.write("No specific certifications found for this role.")
    st.success("✅ Analysis Completed!")
