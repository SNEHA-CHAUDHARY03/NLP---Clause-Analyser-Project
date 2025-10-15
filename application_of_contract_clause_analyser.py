
# pip install streamlit
# pip install pyngrok
# pip install python-docx
# pip install pdfplumber
# pip install joblib

import streamlit as st

import joblib
import re
import pdfplumber
from docx import Document

# Load saved stacking classifier, TF-IDF vectorizer, and label encoder
model = joblib.load('legal_risk_stacking_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Read DOCX
def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Read PDF
def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def split_clauses(text):
    # Simple split by newline
    clauses = [c.strip() for c in text.split("\n") if c.strip()]
    return clauses

# Streamlit app
st.set_page_config(page_title="Legal Clause Risk Analyzer", page_icon="⚖️", layout="wide")

# Hero section
st.title("⚖️ Legal Clause Risk Analyzer")
st.subheader("Let's check your contract clauses and make your life easier!")

st.markdown("""
This app allows you to **paste text**, **upload a PDF**, or **upload a DOCX** file containing your contract.
It will analyze each clause and highlight potential risks:
- 🔴 High Risk
- 🟠 Medium Risk
- 🟢 Low Risk
""")

# Choose input method
input_type = st.radio("Choose input method:", ("Paste Text", "Upload File"))

if input_type == "Paste Text":
    user_text = st.text_area("Paste your contract text here:", height=250)
elif input_type == "Upload File":
    uploaded_file = st.file_uploader("Upload PDF or DOCX file", type=["pdf", "docx"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            user_text = read_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            user_text = read_docx(uploaded_file)

if st.button("Analyze Contract") and (user_text is not None and user_text.strip() != ""):

    # Split into clauses
    clauses = split_clauses(user_text)

    if len(clauses) == 0:
        st.warning("No clauses detected. Please check your input.")
    else:
        # Clean clauses
        clauses_clean = [clean_text(c) for c in clauses]

        # Transform using TF-IDF
        X_input = tfidf.transform(clauses_clean)

        # Predict risk levels
        preds = model.predict(X_input)
        preds_labels = le.inverse_transform(preds)

        # Overall summary
        from collections import Counter
        counts = Counter(preds_labels)
        st.markdown("### Overall Risk Summary")
        st.write(counts)

        # Display clause-level results with colors
        st.markdown("### Clause-Level Risk")
        for clause, risk in zip(clauses, preds_labels):
            if risk.lower() == "high":
                st.markdown(f"<p style='color:red'>{clause} ⚠️ ({risk})</p>", unsafe_allow_html=True)
            elif risk.lower() == "medium":
                st.markdown(f"<p style='color:orange'>{clause} ⚠️ ({risk})</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color:green'>{clause} ✅ ({risk})</p>", unsafe_allow_html=True)

