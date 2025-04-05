import streamlit as st
import torch
from DeepQA.config.configuration import ConfigurationManager
from DeepQA.Components.Stage_06_Model_Prediction import Predictor

# ----------------------------------------
# 🔧 App config
# ----------------------------------------
st.set_page_config(page_title="DeepQA", page_icon="📘", layout="centered")

# ----------------------------------------
# 🌐 Page Navigation Sidebar
# ----------------------------------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🤖 Predict"])


# ================================================================
# 🏠 HOME PAGE
# ================================================================
if page == "🏠 Home":
    st.title("📘 DeepQA - Project Overview")
    st.markdown("---")

    st.header("📌 Project Goal")
    st.write("""
    DeepQA is an end-to-end **Question Answering (QA)** system built using PyTorch. It processes natural language questions 
    and predicts the most likely answer word using Recurrent Neural Networks.

    This project is modular, production-ready, and supports tools like:
    - ✅ DVC for experiment tracking
    - ✅ MLflow (optional)
    - ✅ Dockerized pipeline
    - ✅ Modular configuration with YAML
    """)

    st.header("⚙️ Architecture")
    st.write("""
    The pipeline includes the following stages:

    1. **Data Ingestion**: Loads raw QA pairs  
    2. **Data Validation**: Ensures schema consistency  
    3. **Data Transformation**: Tokenization, vocab building, train-test split  
    4. **Model Training**: Trains RNN on the input sequences  
    5. **Model Evaluation**: Evaluates model on test set  
    6. **Model Prediction**: Loads model and vocab to predict answers for new questions
    """)


    st.header("🚀 How to Use")
    st.markdown("""
    1. Go to the **Predict** page from the sidebar  
    2. Select a model (e.g. RNN)  
    3. Enter your question  
    4. Get an instant answer!
    """)

    st.info("Model currently supports one-word answer predictions. More advanced decoding will be added later.")


# ================================================================
# 🤖 PREDICT PAGE
# ================================================================
elif page == "🤖 Predict":
    st.title("🤖 DeepQA - Predict Answer")
    st.markdown("---")

    # -----------------------------
    # 🔘 Model Dropdown
    # -----------------------------
    model_options = ["RNN"]  # You can add "LSTM", "Transformer", etc. later
    selected_model = st.selectbox("Select a model", model_options)

    # -----------------------------
    # 📦 Load model
    # -----------------------------
    if "predictor" not in st.session_state:
        st.session_state.predictor = None

    if st.button("🔄 Load Model"):
        try:
            config = ConfigurationManager()
            model_config = config.get_model_prediction_config()

            if selected_model == "RNN":
                st.session_state.predictor = Predictor(model_config)

            st.success(f"✅ {selected_model} model loaded successfully!")

        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")

    # -----------------------------
    # ❓ Input & Predict
    # -----------------------------
    st.subheader("Ask a question:")
    question = st.text_input("Type your question here")

    if st.button("🎯 Get Answer"):
        if not st.session_state.predictor:
            st.warning("⚠️ Please load a model first.")
        elif not question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            try:
                answer = st.session_state.predictor.predict(question)
                st.success(f"🧠 Predicted Answer: **{answer}**")
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
