import json
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# RIDEGUARDIAN
# ENGINE FAULT AI API
#
# Loads the already-trained Random Forest model and exposes
# prediction through HTTP.
#
# IMPORTANT:
# Fault 0/1/2/3 keep the original dataset meanings.
# We do not invent mechanical meanings for the classes.
# ============================================================


# ============================================================
# PATHS
# ============================================================

AI_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = (
    AI_DIR
    / "models"
    / "engine_fault"
)

MODEL_PATH = (
    MODEL_DIR
    / "engine_fault_random_forest.joblib"
)

FEATURES_PATH = (
    MODEL_DIR
    / "feature_names.json"
)

CLASS_NAMES_PATH = (
    MODEL_DIR
    / "class_names.json"
)


# ============================================================
# LOAD MODEL FILES
# ============================================================

required_files = [
    MODEL_PATH,
    FEATURES_PATH,
    CLASS_NAMES_PATH,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required Engine Fault AI file not found:\n"
            f"{file_path}"
        )


model = joblib.load(
    MODEL_PATH
)


with open(
    FEATURES_PATH,
    "r",
    encoding="utf-8"
) as file:
    feature_names = json.load(file)


with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:
    class_names = json.load(file)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="RideGuardian Engine Fault AI",
    version="1.0.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class EngineInput(BaseModel):

    AFR: float
    CO: float
    CO2: float

    consumption_l_100km: float
    consumption_l_h: float

    Force: float
    HC: float
    Lambda: float
    MAP: float
    O2: float
    Power: float
    RPM: float
    Speed: float
    TPS: float


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "service": "RideGuardian Engine Fault AI",
        "status": "online",
        "model": "Random Forest",
        "features": feature_names,
        "classes": class_names,
    }


# ============================================================
# MODEL RECOMMENDATION
# ============================================================

def get_recommendation(
    confidence: float,
) -> str:

    if confidence >= 0.90:

        return (
            "The engine data strongly matches the "
            "predicted fault class. Further inspection "
            "is recommended to determine the underlying "
            "mechanical cause."
        )

    if confidence >= 0.75:

        return (
            "The engine data moderately/strongly matches "
            "the predicted fault class. Additional engine "
            "measurements and inspection are recommended."
        )

    if confidence >= 0.60:

        return (
            "The engine data shows a moderate match. "
            "Additional engine measurements are recommended "
            "before making a maintenance decision."
        )

    return (
        "Prediction confidence is low. Collect additional "
        "or cleaner engine data before making a maintenance "
        "decision."
    )


# ============================================================
# CONFIDENCE LEVEL
# ============================================================

def get_confidence_level(
    confidence: float,
) -> str:

    if confidence >= 0.90:
        return "VERY HIGH"

    if confidence >= 0.75:
        return "HIGH"

    if confidence >= 0.60:
        return "MODERATE"

    return "LOW"


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict/engine")
def predict_engine(
    data: EngineInput,
) -> Dict[str, Any]:

    try:

        input_values = {
            "AFR": data.AFR,
            "CO": data.CO,
            "CO2": data.CO2,

            "Consumption L/100KM":
                data.consumption_l_100km,

            "Consumption L/H":
                data.consumption_l_h,

            "Force": data.Force,
            "HC": data.HC,
            "Lambda": data.Lambda,
            "MAP": data.MAP,
            "O2": data.O2,
            "Power": data.Power,
            "RPM": data.RPM,
            "Speed": data.Speed,
            "TPS": data.TPS,
        }


        # ----------------------------------------------------
        # Ensure exact feature ordering used by training
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [
                input_values
            ],
            columns=feature_names,
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )

        probabilities = model.predict_proba(
            input_data
        )


        predicted_fault = (
            prediction[0]
        )


        model_classes = list(
            model.classes_
        )


        predicted_index = (
            model_classes.index(
                predicted_fault
            )
        )


        predicted_probability = float(
            probabilities[0][
                predicted_index
            ]
        )


        # ----------------------------------------------------
        # Class name
        # ----------------------------------------------------

        if predicted_index < len(
            class_names
        ):

            predicted_class_name = (
                class_names[
                    predicted_index
                ]
            )

        else:

            predicted_class_name = (
                f"Fault {predicted_fault}"
            )


        confidence_level = (
            get_confidence_level(
                predicted_probability
            )
        )


        recommendation = (
            get_recommendation(
                predicted_probability
            )
        )


        # ----------------------------------------------------
        # All probabilities
        # ----------------------------------------------------

        class_probabilities = {}

        for i, model_class in enumerate(
            model_classes
        ):

            if i < len(class_names):

                class_name = (
                    class_names[i]
                )

            else:

                class_name = (
                    f"Fault {model_class}"
                )


            class_probabilities[
                class_name
            ] = round(
                float(
                    probabilities[0][i]
                ),
                6,
            )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "success": True,

            "model": "Random Forest",

            "predicted_fault":
                int(predicted_fault),

            "predicted_class":
                predicted_class_name,

            "confidence":
                round(
                    predicted_probability,
                    6,
                ),

            "confidence_percent":
                round(
                    predicted_probability * 100,
                    2,
                ),

            "confidence_level":
                confidence_level,

            "class_probabilities":
                class_probabilities,

            "recommendation":
                recommendation,

            "input": input_values,
        }


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )