from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
FEATURE_NAMES = (
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
)


@st.cache_resource
def load_model():
    return mlflow.pyfunc.load_model(str(MODEL_DIR))


st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏡",
    layout="centered",
)
st.title("California Housing Price Predictor")
st.write(
    "Enter the area's housing and demographic details to estimate its median "
    "house value."
)
st.caption(
    "This model was trained on the California Housing dataset. Its predictions "
    "are estimates, not property appraisals."
)

if not MODEL_DIR.is_dir():
    st.error("The exported model bundle is missing.")
    st.info("Run `python export_model.py` before starting this app.")
    st.stop()

model = load_model()

with st.form("prediction_form"):
    st.subheader("Area details")
    income_col, age_col = st.columns(2)
    with income_col:
        median_income = st.number_input(
            "Median income (× $10,000)",
            min_value=0.0,
            max_value=15.0,
            value=8.3252,
            step=0.1,
            help="Median household income for the block group.",
        )
    with age_col:
        house_age = st.number_input(
            "Median house age (years)",
            min_value=1.0,
            max_value=52.0,
            value=41.0,
            step=1.0,
        )

    rooms_col, bedrooms_col = st.columns(2)
    with rooms_col:
        average_rooms = st.number_input(
            "Average rooms per household",
            min_value=0.0,
            max_value=142.0,
            value=6.9841,
            step=0.1,
        )
    with bedrooms_col:
        average_bedrooms = st.number_input(
            "Average bedrooms per household",
            min_value=0.0,
            max_value=35.0,
            value=1.0238,
            step=0.1,
        )

    population_col, occupancy_col = st.columns(2)
    with population_col:
        population = st.number_input(
            "Area population",
            min_value=0.0,
            max_value=35682.0,
            value=322.0,
            step=1.0,
            format="%.0f",
        )
    with occupancy_col:
        average_occupancy = st.number_input(
            "Average occupants per household",
            min_value=0.0,
            max_value=1244.0,
            value=2.5556,
            step=0.1,
        )

    latitude_col, longitude_col = st.columns(2)
    with latitude_col:
        latitude = st.number_input(
            "Latitude",
            min_value=32.0,
            max_value=42.0,
            value=37.88,
            step=0.01,
            format="%.4f",
        )
    with longitude_col:
        longitude = st.number_input(
            "Longitude",
            min_value=-125.0,
            max_value=-114.0,
            value=-122.23,
            step=0.01,
            format="%.4f",
        )

    submitted = st.form_submit_button(
        "Estimate house value",
        type="primary",
        use_container_width=True,
    )

if submitted:
    features = {
        "MedInc": median_income,
        "HouseAge": house_age,
        "AveRooms": average_rooms,
        "AveBedrms": average_bedrooms,
        "Population": population,
        "AveOccup": average_occupancy,
        "Latitude": latitude,
        "Longitude": longitude,
    }
    input_frame = pd.DataFrame(
        [{name: float(features[name]) for name in FEATURE_NAMES}],
        columns=FEATURE_NAMES,
    )
    prediction = float(model.predict(input_frame)[0])
    estimated_price = prediction * 100000

    st.divider()
    st.subheader("Prediction")
    with st.container(border=True):
        st.metric("Estimated median house value", f"${estimated_price:,.0f}")
        st.caption(
            f"Model output: {prediction:.3f} × $100,000. "
            "Actual prices may differ."
        )
