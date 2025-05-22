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


def get_embeddings(
    position_title: str,
    agency: str,
    education: str,
    experience: int,
    eligibility: str,
    bert_tokenizer,
    bert_model,
) -> np.ndarray:
    """
    Predict the salary of a job posting based on its attributes.

    Args:
            position_title (str): The title of the position.
            agency (str): The agency offering the position.
            education (str): The required education for the position.
            experience (str): The required experience for the position.
            eligibility (str): The eligibility requirements for the position.

    Returns:
            np.ndarray: The input embeddings for the model.
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

    return input_embeddings


def plot_shap_waterfall(
    position_title: str,
    agency: str,
    education: str,
    experience: int,
    eligibility: str,
    tree_explainer: shap.TreeExplainer,
    feature_names: list,
    bert_tokenizer,
    bert_model,
) -> None:

    input_embeddings: np.ndarray = get_embeddings(
        position_title,
        agency,
        education,
        experience,
        eligibility,
        bert_tokenizer,
        bert_model,
    )

    # Compute SHAP values
    shap_values = tree_explainer.shap_values(input_embeddings)

    # Explanation
    explanation = shap.Explanation(
        values=shap_values,
        base_values=tree_explainer.expected_value,
        data=input_embeddings,
        feature_names=feature_names,
    )
    # Plot SHAP values
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
