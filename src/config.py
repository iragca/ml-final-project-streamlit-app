import os
from pathlib import Path

import joblib
import streamlit as st
import polars as pl
from dotenv import load_dotenv
from loguru import logger
from sklearn.model_selection import train_test_split


@st.cache_resource
def init_config():
    # Load environment variables from .env file

    load_dotenv()

    PROJ_ROOT = Path(__file__).resolve().parent.parent

    DATA_DIR = PROJ_ROOT / "data"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"

    MODELS_DIR = PROJ_ROOT / "models"

    MPL_STYLE_DIR = PROJ_ROOT / "src" / "matplotlib"

    RAW_DATA = pl.read_parquet(
        EXTERNAL_DATA_DIR
        / "civilservicecommission-unclean-training-data.parquet"
    )

    logger.info(f"{PROJ_ROOT=}")

    return MODELS_DIR, MPL_STYLE_DIR, RAW_DATA, DATA_DIR, EXTERNAL_DATA_DIR


@st.cache_resource
def init_data():
    TRAINING_DATA = RAW_DATA.select(
        [
            pl.col("positiontitle_embedding"),
            pl.col("agency_embedding"),
            pl.col("education_embedding"),
            pl.col("experience_years"),
            pl.col("eligibility_embedding"),
            pl.col("MonthlySalary"),
        ]
    )

    X = TRAINING_DATA.drop("MonthlySalary").to_numpy()
    y = TRAINING_DATA.select("MonthlySalary").to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


MODELS_DIR, MPL_STYLE_DIR, RAW_DATA, DATA_DIR, EXTERNAL_DATA_DIR = (
    init_config()
)

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = init_data()
KNN_MODEL = joblib.load(MODELS_DIR / "civilservicecommission-knn-model.joblib")
