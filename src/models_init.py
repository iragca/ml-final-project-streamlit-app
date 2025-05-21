import shap
import streamlit as st
from transformers import BertModel, BertTokenizer

from src.config import X_TRAIN, X_TEST, MODEL


@st.cache_resource
def load_bert_model():
    """
    Load the pre-trained BERT model and tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    return tokenizer, model


@st.cache_resource
def init_shap():
    background = shap.sample(X_TRAIN, 100, random_state=0)
    explainer = shap.KernelExplainer(MODEL.predict, background)

    # Explain a few predictions
    shap_values = explainer.shap_values(
        X_TEST[:100]
    )  # slice to limit compute time
    return explainer
