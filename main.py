import io

import streamlit as st

import matplotlib.pyplot as plt
from src.config import KNN_MODEL, X_TEST, RAW_DATA
from src.inference import inference, plot_shap_waterfall
from src.models_init import init_shap, load_bert_model
from src.utils import calculate_percentile

# Load models
tokenizer, model = load_bert_model()
explainer = init_shap()


def main():
    st.title("Should I be hired by the government? 🇵🇭")
    st.divider()

    with st.sidebar:
        st.title("About")
        st.markdown(
            "This is a [web app](https://github.com/iragca/ml-final-project-streamlit-app) that predicts monthly salaries (PHP) based on job listing features "
            "using the original uncased **BERT** model from Google, without any fine-tuning."
        )
        st.markdown(
            "It uses a **K-Nearest Neighbors (KNN) regressor** "
            "with 11 neighbors and the Manhattan distance metric. "
            "The model achieves an *R²* score of **0.874**."
        )

        st.header("Dataset")
        st.markdown(
            "Gathered from the [website](https://csc.gov.ph/career/) of the Civil Service Commission of the Philippines. "
            "These are job listings from November 2024 to May 2025. "
        )
        st.markdown(
            "**Download the [raw](https://github.com/iragca/ml-final-project/raw/refs/heads/master/data/processed/CivilServiceCommission/civilservicecommission-2.parquet) "
            "or [training](https://github.com/iragca/ml-final-project/raw/refs/heads/master/data/processed/CivilServiceCommission/civilservicecommission-unclean-training-data.parquet) datasets found in our GitHub repo.**"
        )
        st.dataframe(RAW_DATA.sample(5))
        st.subheader("Authors")
        st.write("Chris Irag and others, Group 2, DS3A ")
        st.caption("[Coursework for DS322 - Machine Learning](https://github.com/iragca/ml-final-project)")

    col1, col2 = st.columns(2)

    with col1:
        position = st.text_input(
            "Enter your desired position",
            "Data Scientist",
            placeholder="Data Scientist",
        )
        agency = st.text_input(
            "Agency you want to work for",
            "Department of Education",
            placeholder="Department of Education",
        )
    with col2:
        education_level = st.text_input(
            "Education", "Bachelor's Degree", placeholder="Bachelor's Degree"
        )
        experience = st.number_input(
            "Years of experience",
            placeholder=0,
            min_value=0,
            max_value=60,
            step=1,
        )

    eligibility = st.text_input(
        "Eligibility",
        "Career Service (Professional) First Level",
        placeholder="Career Service (Professional) First Level",
    )

    st.markdown("Example of a job listing:")
    experience_col_config = st.column_config.NumberColumn(
        label="Experience (Years)", width=30
    )
    st.dataframe(
        RAW_DATA.select(
            [
                "Position Title",
                "Agency",
                "Education",
                "experience_years",
                "Eligibility",
            ]
        ).sample(1),
        height=50,
        column_config={
            "experience_years": experience_col_config,
        },
    )
    st.button("Another example", use_container_width=True)
    st.divider()

    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Calculating..."):
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

            percentile = calculate_percentile(
                RAW_DATA, "MonthlySalary", prediction[0]
            )

            with st.expander(
                f"PHP {float(prediction[0]):,.2f} | Percentile: {percentile:.2f}%",
                expanded=True,
            ):
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
