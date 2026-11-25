
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy import sparse
from pathlib import Path


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 2. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #f7f9f6;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 38px;
        font-weight: 750;
        color: #174d2b;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #66756a;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 650;
        color: #205c35;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .prediction-card {
        background-color: #e8f5e9;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #c8e6c9;
        text-align: center;
    }

    .prediction-label {
        font-size: 17px;
        color: #386641;
    }

    .prediction-value {
        font-size: 42px;
        font-weight: 750;
        color: #174d2b;
    }

    div.stButton > button {
        background-color: #24733c;
        color: white;
        border-radius: 8px;
        height: 48px;
        width: 100%;
        font-size: 17px;
        font-weight: 600;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #174d2b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOAD MODEL AND PREPROCESSING ARTIFACTS
# --------------------------------------------------

ARTIFACT_PATH = Path(__file__).parent / "crop_yield_artifacts.pkl"


@st.cache_resource
def load_artifacts():
    artifacts = joblib.load(ARTIFACT_PATH)

    required_keys = ["model", "scaler", "encoder", "num_cols", "cat_cols"]

    for key in required_keys:
        if key not in artifacts:
            raise KeyError(
                f"Missing '{key}' in crop_yield_artifacts.pkl"
            )

    return artifacts


try:
    artifacts = load_artifacts()

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]
    num_cols = list(artifacts["num_cols"])
    cat_cols = list(artifacts["cat_cols"])

except FileNotFoundError:
    st.error(
        "Model artifact file not found. "
        "Place crop_yield_artifacts.pkl in the same folder as app.py."
    )
    st.stop()

except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()


# --------------------------------------------------
# 4. APPLICATION HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🌾 Crop Yield Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict agricultural crop yield using soil, climate, '
    'and farming parameters.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# 5. SIDEBAR INFORMATION
# --------------------------------------------------

with st.sidebar:
    st.title("🌱 About Project")

    st.write(
        "This Machine Learning application predicts "
        "crop yield based on agricultural input parameters."
    )

    st.markdown("---")

    st.subheader("Model Information")

    st.write("**Algorithm:** Random Forest Regressor")
    st.write("**Numerical Scaling:** MinMaxScaler")
    st.write("**Categorical Encoding:** OneHotEncoder")

    st.markdown("---")

    st.caption(
        "Enter values consistent with the units and "
        "ranges used in the training dataset."
    )


# --------------------------------------------------
# 6. INPUT FORM
# --------------------------------------------------

st.markdown(
    '<div class="section-title">1. Enter Agricultural Details</div>',
    unsafe_allow_html=True
)

with st.form("crop_prediction_form"):

    # ----------------------------------------------
    # SOIL AND NUTRIENT DETAILS
    # ----------------------------------------------

    st.subheader("🌱 Soil and Nutrient Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input(
            "Nitrogen (N)",
            min_value=30.0,
            max_value=179.0,
            value=105.0,
            step=1.0
        )

        P = st.number_input(
            "Phosphorus (P)",
            min_value=15.0,
            max_value=99.0,
            value=57.0,
            step=1.0
        )

        K = st.number_input(
            "Potassium (K)",
            min_value=20.0,
            max_value=149.0,
            value=85.0,
            step=1.0
        )

        Soil_pH = st.number_input(
            "Soil pH",
            min_value=4.8,
            max_value=8.2,
            value=6.5,
            step=0.1
        )

    with col2:
        Soil_Moisture = st.number_input(
            "Soil Moisture",
            min_value=15.0,
            max_value=64.99,
            value=40.0,
            step=0.1
        )

        Organic_Carbon = st.number_input(
            "Organic Carbon",
            min_value=0.30,
            max_value=1.60,
            value=0.96,
            step=0.01
        )

        Fertilizer_Used = st.number_input(
            "Fertilizer Used",
            min_value=60.0,
            max_value=349.96,
            value=205.0,
            step=1.0
        )

        Pesticide_Used = st.number_input(
            "Pesticide Used",
            min_value=0.0,
            max_value=25.0,
            value=12.5,
            step=0.1
        )

    with col3:
        Soil_Type = st.selectbox(
            "Soil Type",
            ["Clay", "Loamy", "Sandy", "Silt"]
        )

        Region = st.selectbox(
            "Region",
            ["Central", "East", "North", "South", "West"]
        )

        Altitude = st.number_input(
            "Altitude",
            min_value=0.0,
            max_value=2199.0,
            value=1103.0,
            step=1.0
        )

    st.divider()

    # ----------------------------------------------
    # CLIMATE DETAILS
    # ----------------------------------------------

    st.subheader("🌤️ Climate Details")

    col4, col5, col6 = st.columns(3)

    with col4:
        Temperature = st.number_input(
            "Temperature",
            min_value=10.0,
            max_value=40.0,
            value=25.0,
            step=0.1
        )

        Humidity = st.number_input(
            "Humidity",
            min_value=30.01,
            max_value=89.99,
            value=60.0,
            step=0.1
        )

    with col5:
        Rainfall = st.number_input(
            "Rainfall",
            min_value=300.70,
            max_value=2799.26,
            value=1546.0,
            step=1.0
        )

        Sunlight_Hours = st.number_input(
            "Sunlight Hours",
            min_value=4.0,
            max_value=11.0,
            value=7.5,
            step=0.1
        )

    with col6:
        Wind_Speed = st.number_input(
            "Wind Speed",
            min_value=1.0,
            max_value=18.0,
            value=9.5,
            step=0.1
        )

    st.divider()

    # ----------------------------------------------
    # FARMING DETAILS
    # ----------------------------------------------

    st.subheader("🚜 Farming Details")

    col7, col8, col9 = st.columns(3)

    with col7:
        Season = st.selectbox(
            "Season",
            ["Kharif", "Rabi", "Zaid"]
        )

    with col8:
        Crop_Type = st.selectbox(
            "Crop Type",
            [
                "Cotton",
                "Maize",
                "Potato",
                "Rice",
                "Sugarcane",
                "Wheat"
            ]
        )

    with col9:
        Irrigation_Type = st.selectbox(
            "Irrigation Type",
            ["Canal", "Drip", "Rainfed", "Sprinkler"]
        )

    st.markdown("---")

    predict_button = st.form_submit_button(
        "🌾 Predict Crop Yield"
    )


# --------------------------------------------------
# 7. PREDICTION
# --------------------------------------------------

if predict_button:

    try:
        # Create input DataFrame in the original
        # dataset's feature order.

        input_data = pd.DataFrame([{
            "N": N,
            "P": P,
            "K": K,
            "Soil_pH": Soil_pH,
            "Soil_Moisture": Soil_Moisture,
            "Soil_Type": Soil_Type,
            "Organic_Carbon": Organic_Carbon,
            "Temperature": Temperature,
            "Humidity": Humidity,
            "Rainfall": Rainfall,
            "Sunlight_Hours": Sunlight_Hours,
            "Wind_Speed": Wind_Speed,
            "Region": Region,
            "Altitude": Altitude,
            "Season": Season,
            "Crop_Type": Crop_Type,
            "Irrigation_Type": Irrigation_Type,
            "Fertilizer_Used": Fertilizer_Used,
            "Pesticide_Used": Pesticide_Used
        }])

        # ------------------------------------------
        # NUMERICAL TRANSFORMATION
        # ------------------------------------------

        numerical_input = input_data[num_cols]

        numerical_transformed = scaler.transform(
            numerical_input
        )

        # ------------------------------------------
        # CATEGORICAL TRANSFORMATION
        # ------------------------------------------

        categorical_input = input_data[cat_cols]

        categorical_transformed = encoder.transform(
            categorical_input
        )

        # Handle sparse OneHotEncoder output
        if sparse.issparse(categorical_transformed):
            categorical_transformed = (
                categorical_transformed.toarray()
            )

        # ------------------------------------------
        # COMBINE TRANSFORMED FEATURES
        # ------------------------------------------

        numerical_transformed = np.asarray(
            numerical_transformed
        )

        categorical_transformed = np.asarray(
            categorical_transformed
        )

        final_input = np.hstack([
            numerical_transformed,
            categorical_transformed
        ])

        # ------------------------------------------
        # MODEL PREDICTION
        # ------------------------------------------

        prediction = model.predict(final_input)[0]

        # ------------------------------------------
        # DISPLAY RESULT
        # ------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-title">2. Prediction Result</div>',
            unsafe_allow_html=True
        )

        st.success("Crop yield prediction completed!")

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Estimated Crop Yield
                </div>
                <div class="prediction-value">
                    {prediction:.2f}
                </div>
                <div class="prediction-label">
                    Tons per Hectare
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                label="Predicted Yield",
                value=f"{prediction:.2f} tons/hectare"
            )

        with result_col2:
            st.metric(
                label="Selected Crop",
                value=Crop_Type
            )

        # Display submitted input values
        with st.expander("View Entered Agricultural Details"):
            st.dataframe(
                input_data,
                use_container_width=True
            )

    except Exception as e:
        st.error(
            "Prediction failed. Please check your model "
            "artifacts and preprocessing configuration."
        )

        st.exception(e)


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#7a887d;">
        Crop Yield Prediction System |
        Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)
