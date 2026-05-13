import streamlit as st
import joblib, re

# Page config
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('best_model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    return model, tfidf

model, tfidf = load_model()

STOPWORDS = {'i','me','my','we','our','you','your','he','him','his','she',
             'her','it','its','they','them','their','what','this','that',
             'am','is','are','was','were','be','been','have','has','had',
             'do','does','did','a','an','the','and','but','if','or','as',
             'of','at','by','for','with','to','from','in','on','so','not',
             'no','just','will','can','now','its','also'}

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', 'url', text)
    text = re.sub(r'\d+', 'num', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join([w for w in text.split() if w not in STOPWORDS and len(w) > 2])

# ── HEADER ──────────────────────────────────────────────
st.title("📧 Email Spam Classifier")
st.markdown("*Comparing 8 Machine Learning Algorithms — ML Project*")
st.divider()

# ── TABS ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Classify Email", "📊 Model Comparison", "📈 Charts"])

# ── TAB 1: CLASSIFY ─────────────────────────────────────
with tab1:
    st.subheader("Paste an email to classify it")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_area("Email content:", height=200,
            placeholder="Paste subject + body of any email here...")

    with col2:
        st.markdown("**Try these examples:**")
        if st.button("🔴 Spam Example 1"):
            st.session_state['example'] = "WINNER!! You have been selected to receive a £900 prize! Call 09061701461 to claim. Limited time offer!"
        if st.button("🔴 Spam Example 2"):
            st.session_state['example'] = "URGENT: Your account has been suspended. Click here immediately to verify: http://fake-bank.com/login"
        if st.button("🟢 Ham Example 1"):
            st.session_state['example'] = "Hey, are you coming to the team meeting tomorrow at 3pm? Let me know if you need the Zoom link."
        if st.button("🟢 Ham Example 2"):
            st.session_state['example'] = "Reminder: Your dentist appointment is on Tuesday at 10:30am. Please arrive 10 minutes early."

    if 'example' in st.session_state:
        user_input = st.session_state['example']
        del st.session_state['example']
        st.rerun()

    if st.button("🔎 Classify", type="primary", use_container_width=True):
        if user_input.strip():
            cleaned = preprocess(user_input)
            vec = tfidf.transform([cleaned])
            prediction = model.predict(vec.toarray())[0]
            probability = model.predict_proba(vec.toarray())[0]
            confidence = max(probability) * 100

            st.divider()
            if prediction == 1:
                st.error(f"## 🚨 SPAM DETECTED")
                st.markdown(f"**Confidence: {confidence:.1f}%**")
                st.progress(confidence / 100)
                st.warning("This email shows characteristics of spam. Do not click any links or share personal information.")
            else:
                st.success(f"## ✅ LEGITIMATE EMAIL (HAM)")
                st.markdown(f"**Confidence: {confidence:.1f}%**")
                st.progress(confidence / 100)
                st.info("This email appears to be legitimate.")

            col_a, col_b = st.columns(2)
            col_a.metric("Spam Probability", f"{probability[1]*100:.1f}%")
            col_b.metric("Ham Probability",  f"{probability[0]*100:.1f}%")
        else:
            st.warning("Please paste some email text first.")

# ── TAB 2: MODEL COMPARISON ─────────────────────────────
with tab2:
    st.subheader("Algorithm Performance Comparison")
    st.markdown("All models trained on **5,572 real SMS/email messages** · **5-fold cross-validation**")

    import pandas as pd
    try:
        df = pd.read_csv('results.csv', index_col=0)
        df = df.sort_values('F1', ascending=False)

        # Highlight best row
        def highlight_best(row):
            if row.name == df.index[0]:
                return ['background-color: #1e3a5f; color: white'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df.style.highlight_max(axis=0, color='#1e3a5f')
                    .format("{:.2f}%"),
            use_container_width=True
        )
        st.success(f"🏆 **Best Model: {df.index[0]}** with F1-Score of {df.loc[df.index[0],'F1']:.2f}%")
    except:
        st.error("Run spam_classifier.py first to generate results.csv")

    st.divider()
    st.markdown("""
    **Why Naive Bayes wins for text classification:**
    - Handles high-dimensional text data perfectly
    - Assumes word independence (works great for spam)
    - Extremely fast — classifies in milliseconds
    - Strong probabilistic foundation
    """)

# ── TAB 3: CHARTS ───────────────────────────────────────
with tab3:
    st.subheader("Visualizations")
    import os
    charts = [
        ('chart1_metrics.png', 'Accuracy, Precision, Recall & F1 — All Models'),
        ('chart2_roc.png',     'ROC Curves — All Models'),
        ('chart3_confusion.png','Confusion Matrix — Best Model'),
        ('chart4_cv.png',      'Cross-Validation Scores'),
    ]
    for fname, title in charts:
        if os.path.exists(fname):
            st.markdown(f"**{title}**")
            st.image(fname, use_column_width=True)
            st.divider()
        else:
            st.warning(f"Run spam_classifier.py to generate {fname}")

# ── FOOTER ──────────────────────────────────────────────
st.divider()
st.markdown("*Built with Python · Scikit-learn · Streamlit · TF-IDF Feature Extraction*")
