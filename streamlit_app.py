import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import base64


# Load your model and encoders
model = load_model("src/my_model.h5")
le = pickle.load(open("src/label_encoder.pkl", "rb"))
symptom_index = pickle.load(open("src/symptom_index.pkl", "rb"))
all_symptoms = list(symptom_index.keys())

# Set up background image using HTML/CSS
def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg(path):
    img_base64 = get_base64(path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .title-text {{
            color: white;
            font-size: 36px;
            font-weight: bold;
        }}
        .subtitle-text {{
            color: #dddddd;
            font-size: 18px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("src/bg.png")

# ---- HEADER ----
st.markdown("""
    <div style="
        background-color: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    ">
        <h1 class="title-text">🩺 Disease Predictor AI</h1>
        <p class="subtitle-text">Select your symptoms below to get AI-powered disease predictions</p>
    </div>
""", unsafe_allow_html=True)

# ---- SYMPTOM INPUT ----
# ---- SYMPTOM INPUT ----
with st.expander("🔍 Select Your Symptoms", expanded=True):
    selected_symptoms = st.multiselect(
        "Select Symptoms",
        sorted(all_symptoms),
        help="Select any symptoms you are experiencing"
    )

predict_col, clear_col = st.columns([3, 1])
with clear_col:
    if st.button("🧹 Clear", use_container_width=True):
        selected_symptoms = []
        st.rerun()


# ---- PREDICT ----

with predict_col:
    if st.button("🔮 Predict Disease", type="primary", use_container_width=True):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            input_vector = [0] * len(symptom_index)
            for symptom in selected_symptoms:
                input_vector[symptom_index[symptom]] = 1

            input_array = np.array(input_vector).reshape(1, -1)
            probabilities = model.predict(input_array, verbose=0)[0]
            top3_indices = np.argsort(probabilities)[-3:][::-1]

            filtered_results = [(le.inverse_transform([i])[0], round(probabilities[i] * 100, 2))
                            for i in top3_indices if probabilities[i] >= 0.20]

            if not filtered_results:
                st.error("⚠️ The model is not confident about any prediction based on these symptoms. Please try selecting more symptoms.")
            else:
                with st.container():
                    
                    st.success("## Top Predictions")
                    for i, (disease, confidence) in enumerate(filtered_results, 1):
                        with st.expander(f"{i}. {disease} ({confidence}%)", expanded=True if i == 1 else False):
                            st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
                            st.progress(int(confidence), text=f"Confidence: {confidence}%")
                            st.markdown(f"**Recommendation:** Consult a healthcare professional for proper diagnosis and treatment.")
                            st.markdown(f"**Next Steps:** Consider tracking these symptoms and their progression.")
                            st.markdown('</div>', unsafe_allow_html=True)


st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <a href="https://ecf96222277bcf0e36.gradio.live" target="_blank" style="
            display: inline-block;
            padding: 10px 20px;
            font-size: 16px;
            color: white;
            border: 2px solid white;
            border-radius: 10px;
            text-decoration: none;
            background-color: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(4px);
            transition: all 0.3s ease;
        " onmouseover="this.style.backgroundColor='rgba(255,255,255,0.2)';"
          onmouseout="this.style.backgroundColor='rgba(255,255,255,0.1)';">
            💬 Chat with AI Agent
        </a>
    </div>
""", unsafe_allow_html=True)
