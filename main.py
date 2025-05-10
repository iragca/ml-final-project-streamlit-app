import streamlit as st

import matplotlib.pyplot as plt
from src.config import X_TEST, KNN_MODEL
from src.inference import inference, plot_shap_waterfall
from src.models_init import init_shap, load_bert_model

# Load models
tokenizer, model = load_bert_model()
explainer = init_shap()


def main():
    st.title("Should I be hired by the government? 🇵🇭")
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        position = st.text_input(
            "Enter your desired position", placeholder="Data Scientist"
        )
        agency = st.text_input(
            "Agency you want to work for",
            placeholder="Department of Education",
        )
    with col2:
        education_level = st.text_input(
            "Education level", placeholder="Bachelor's Degree"
        )
        experience = st.number_input(
            "Years of experience",
            placeholder=0,
            min_value=0,
            max_value=60,
            step=1,
        )

    eligibility = st.text_input(
        "Elegibility",
        placeholder="Career Service (Subprofessional) First Level",
    )

    if st.button("Predict", type="primary"):

        prediction = ""
        prediction = inference(
            position_title=position,
            agency=agency,
            education=education_level,
            experience=experience,
            eligibility=eligibility,
            knn_model=KNN_MODEL,
            bert_tokenizer=tokenizer,
            bert_model=model,
        ).ravel()

        with st.expander(f"PHP {float(prediction[0]):,.2f}", expanded=True):

            fig, ax = plt.subplots()
            plot_shap_waterfall(
                position_title=position,
                agency=agency,
                education=education_level,
                experience=experience,
                eligibility=eligibility,
                explainer=explainer,
                X_test=X_TEST,
                bert_tokenizer=tokenizer,
                bert_model=model,
                feature_names=[
                    "Position Title",
                    "Agency",
                    "Education",
                    "Experience",
                    "Eligibility",
                ],
            )
            st.pyplot(fig, use_container_width=True)


if __name__ == "__main__":
    main()
