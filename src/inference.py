import numpy as np
import shap
import torch


def get_cls_embedding(text, tokenizer, bert_model):
    """
    Generate CLS embedding for a given text using a pre-trained BERT model.

    Parameters:
    - text (str): The input text to encode.
    - tokenizer (transformers.BertTokenizer): The tokenizer for the BERT model.
    - model (transformers.BertModel): The pre-trained BERT model.

    Returns:
    - torch.Tensor: The CLS embedding of the input text.
    """
    # Tokenize and convert to input IDs
    inputs = tokenizer(text, return_tensors="pt")

    # Forward pass through BERT
    torch.set_grad_enabled(False)
    outputs = bert_model(**inputs)
    torch.set_grad_enabled(True)

    # Get the last hidden state (sequence output)
    last_hidden_state = (
        outputs.last_hidden_state
    )  # shape: [1, seq_len, hidden_size]

    # To get sentence embedding, use the [CLS] token (first token)
    cls_embedding = last_hidden_state[0, 0]  # shape: [hidden_size]
    return cls_embedding.sum()


def get_shap_value(
    position_title: str,
    agency: str,
    education: str,
    experience: int,
    eligibility: str,
    explainer: shap.KernelExplainer,
    bert_tokenizer,
    bert_model,
) -> float:
    """
    Predict the salary of a job posting based on its attributes.

    Args:
            position_title (str): The title of the position.
            agency (str): The agency offering the position.
            education (str): The required education for the position.
            experience (str): The required experience for the position.
            eligibility (str): The eligibility requirements for the position.

    Returns:
            float: The predicted salary for the job posting.
    """
    cls_embedding = get_cls_embedding(
        position_title, bert_tokenizer, bert_model
    ).sum()
    cls_embedding2 = get_cls_embedding(
        agency, bert_tokenizer, bert_model
    ).sum()
    cls_embedding3 = get_cls_embedding(
        education, bert_tokenizer, bert_model
    ).sum()
    cls_embedding5 = get_cls_embedding(
        eligibility, bert_tokenizer, bert_model
    ).sum()

    input_embeddings = np.array(
        [
            cls_embedding,
            cls_embedding2,
            cls_embedding3,
            experience,
            cls_embedding5,
        ]
    )

    return explainer.shap_values(input_embeddings).ravel()


def plot_shap_waterfall(
    position_title: str,
    agency: str,
    education: str,
    experience: int,
    eligibility: str,
    explainer: shap.KernelExplainer,
    X_test: np.ndarray,
    bert_tokenizer,
    bert_model,
    feature_names: list,
):
    """
    Plot a SHAP waterfall plot for a given job posting.

    Args:
        position_title (str): The title of the position.
        agency (str): The agency offering the position.
        education (str): The required education for the position.
        experience (str): The required experience for the position.
        eligibility (str): The eligibility requirements for the position.
        explainer (shap.KernelExplainer): The SHAP explainer object.
        X_test (np.ndarray): Test dataset.
        feature_names (list): List of feature names.

    Returns:
        None
    """
    test_input = get_shap_value(
        position_title,
        agency,
        education,
        experience,
        eligibility,
        explainer,
        bert_tokenizer,
        bert_model,
    )

    shap_val = test_input  # SHAP values for the first instance
    base_val = explainer.expected_value  # Scalar base value
    data_row = (
        X_test.iloc[0] if hasattr(X_test, "iloc") else X_test[0]
    )  # Row data

    # Wrap as Explanation object
    explanation = shap.Explanation(
        values=shap_val,
        base_values=base_val,
        data=data_row,
        feature_names=feature_names,
    )

    # Now plot
    shap.plots.waterfall(explanation)


def inference(
    position_title: str,
    agency: str,
    education: str,
    experience: str,
    eligibility: str,
    MODEL,
    bert_tokenizer,
    bert_model,
) -> float:
    """
    Predict the salary of a job posting based on its attributes.

    Args:
            position_title (str): The title of the position.
            agency (str): The agency offering the position.
            education (str): The required education for the position.
            experience (str): The required experience for the position.
            eligibility (str): The eligibility requirements for the position.

    Returns:
            float: The predicted salary for the job posting.
    """
    cls_embedding = get_cls_embedding(
        position_title, bert_tokenizer, bert_model
    ).sum()
    cls_embedding2 = get_cls_embedding(
        agency, bert_tokenizer, bert_model
    ).sum()
    cls_embedding3 = get_cls_embedding(
        education, bert_tokenizer, bert_model
    ).sum()

    cls_embedding5 = get_cls_embedding(
        eligibility, bert_tokenizer, bert_model
    ).sum()

    input_embeddings = np.array(
        [
            cls_embedding,
            cls_embedding2,
            cls_embedding3,
            experience,
            cls_embedding5,
        ]
    ).reshape(1, -1)

    return MODEL.predict(input_embeddings)
