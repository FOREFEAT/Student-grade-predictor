import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Student Grade Predictor", page_icon="📋", layout="centered")

st.markdown("""
<style>
.main {
    background: #fffef9;
}
.title-box {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 18px;
    border: 1px solid #f1d9a6;
    border-radius: 14px;
    background: #fff8e8;
    margin-bottom: 18px;
}
.title-text {
    font-size: 28px;
    font-weight: 700;
    color: #2b2b2b;
}
.sub-box {
    padding: 18px;
    border: 1px solid #f3e2b8;
    border-radius: 14px;
    background: #fffdf7;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
    margin-bottom: 18px;
}
.result-box {
    text-align: center;
    padding: 24px 18px;
    border: 1px solid #f3e2b8;
    border-radius: 14px;
    background: #fffaf0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-top: 18px;
}
.result-box.fail {
    border: 1px solid #f3c2b8;
    background: #fff4f2;
}
.result-label {
    font-size: 18px;
    color: #9a6a00;
    font-weight: 600;
}
.result-grade {
    font-size: 36px;
    font-weight: 800;
    color: #c58a00;
    margin-top: 8px;
}
.result-grade.fail {
    color: #c53d00;
}
.placeholder-text {
    font-size: 18px;
    color: #9a8a6a;
    font-style: italic;
}
.small-text {
    color: #555;
    font-size: 15px;
}
.reason-text {
    color: #a04a00;
    font-size: 14px;
    margin-top: 6px;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #e0a100, #d89200);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 16px;
    font-weight: 700;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #c58f00, #b67f00);
    color: white;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "model.pkl"

MIN_PASSING_INTERNAL_MARKS = 14
MIN_PASSING_ATTENDANCE = 50


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Could not find `{MODEL_PATH}`. Run `python train_model.py` first "
            "to generate the dataset and train the model."
        )
        st.stop()
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_names"], bundle["test_accuracy"]


model, feature_names, test_accuracy = load_model()

st.markdown("""
<div class="title-box">
    <div style="font-size:40px;">📋</div>
    <div class="title-text">Student Grade Predictor</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sub-box">', unsafe_allow_html=True)

internal_marks = st.number_input(
    "Enter Internal Marks (out of 40)",
    min_value=0,
    max_value=40,
    value=34,
    step=1
)

attendance = st.number_input(
    "Enter Attendance (%)",
    min_value=0,
    max_value=100,
    value=85,
    step=1
)

predict = st.button("Predict Grade")

st.markdown('</div>', unsafe_allow_html=True)

if predict:
    user_data = pd.DataFrame([[internal_marks, attendance]], columns=feature_names)
    grade = model.predict(user_data)[0]
    probabilities = model.predict_proba(user_data)[0]
    confidence = max(probabilities) * 100

    fail_reasons = []
    if grade == "Fail":
        if internal_marks < MIN_PASSING_INTERNAL_MARKS:
            fail_reasons.append(f"Internal marks below minimum ({MIN_PASSING_INTERNAL_MARKS})")
        if attendance < MIN_PASSING_ATTENDANCE:
            fail_reasons.append(f"Attendance below minimum ({MIN_PASSING_ATTENDANCE}%)")

    box_class = "result-box fail" if grade == "Fail" else "result-box"
    grade_class = "result-grade fail" if grade == "Fail" else "result-grade"
    icon = "❌" if grade == "Fail" else "🏆"

    reasons_html = ""
    if fail_reasons:
        reasons_html = "<p class='reason-text'>" + " · ".join(fail_reasons) + "</p>"

    st.markdown(f"""
    <div class="{box_class}">
        <div class="result-label">Prediction Result</div>
        <div class="{grade_class}">{grade} {icon}</div>
        <p class="small-text">Model confidence: {confidence:.1f}%</p>
        {reasons_html}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="result-box">
        <div class="result-label">Prediction Result</div>
        <p class="placeholder-text">Enter details above and click "Predict Grade" to see a result.</p>
    </div>
    """, unsafe_allow_html=True)
