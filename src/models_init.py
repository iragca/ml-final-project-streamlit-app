import shap
import streamlit as st
from transformers import BertModel, BertTokenizer

from src.config import MODEL


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
    explainer = shap.TreeExplainer(MODEL)
    return explainer
