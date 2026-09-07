from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
import requests


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RideGuardian AI API",
    description=(
        "Vehicle AI, Riding Risk ML, Federated Learning "
        "and Route Map Matching"
    ),
    version="1.0.0"
)


# ============================================================
# FEDERATED LEARNING
#
# Keep Federated Learning optional so the Vehicle AI backend
# can start independently.
# ============================================================

FL_AVAILABLE = False

try:

    from fl.storage import initialize_fl_database
    from fl.routes import router as fl_router

    initialize_fl_database()

    app.include_router(fl_router)

    FL_AVAILABLE = True

    print(
        "Federated Learning routes registered successfully"
    )

except Exception as error:

    print(
        "Federated Learning unavailable:",
        str(error)
    )


# ============================================================
# VEHICLE AI
#
# This is the primary system for this backend.
# ============================================================

VEHICLE_AI_AVAILABLE = False

try:

    from ai.api.vehicle_diagnosis_api import predict_vehicle

    app.add_api_route(
        "/predict/vehicle",
        predict_vehicle,
        methods=["POST"],
        tags=["Vehicle AI"],
        summary="Vehicle AI Diagnosis"
    )

    VEHICLE_AI_AVAILABLE = True

    print(
        "Vehicle AI endpoint registered successfully"
    )

except Exception as error:

    print(
        "Vehicle AI unavailable:",
        str(error)
    )


# ============================================================
# LOAD EXISTING RIDING RISK MODEL
#
# IMPORTANT:
# This backend can operate without the riding-risk model.
# If the file exists, the old Riding Risk ML system works.
# If it does not exist, Vehicle AI still works normally.
# ============================================================

MODEL_AVAILABLE = False

MODEL_PATH = Path(
    os.environ.get(
        "RIDEGUARDIAN_RISK_MODEL_PATH",
        str(
            BASE_DIR
            / "rideguardian_risk_event_model.joblib"
        )
    )
)

model = None
features = []

try:

    if MODEL_PATH.exists():

        model_data = joblib.load(
            MODEL_PATH
        )

        model = model_data["model"]

        features = model_data["features"]

        MODEL_AVAILABLE = True

        print(
            "RideGuardian AI riding-risk model "
            "loaded successfully"
        )

        print(
            f"Riding-risk model path: "
            f"{MODEL_PATH}"
        )

        print(
            f"Number of ML features: "
            f"{len(features)}"
        )

    else:

        print(
            "Riding-risk model not found."
        )

        print(
            "Vehicle AI will continue running normally."
        )

except Exception as error:

    print(
        "Failed to load riding-risk model:",
        str(error)
    )

    print(
        "Vehicle AI will continue running normally."
    )


# ============================================================
# SENSOR DATA FORMAT
# ============================================================

class SensorData(BaseModel):

    speedKmh: float

    accelerationX: float
    accelerationY: float
    accelerationZ: float

    gyroX: float
    gyroY: float
    gyroZ: float

    timeSincePreviousReadingSec: float
    speedChangeKmh: float
    speedAccelerationMs2: float
    gyroMagnitudeRadS: float


# ============================================================
# HEALTH CHECK / HOME ENDPOINT
# ============================================================

@app.get("/")

def home():

    systems = []

    if VEHICLE_AI_AVAILABLE:

        systems.append(
            "Vehicle AI"
        )

    if MODEL_AVAILABLE:

        systems.append(
            "Riding Risk ML"
        )

        systems.append(
            "Vehicle Wear Analysis"
        )

    if FL_AVAILABLE:

        systems.append(
            "Federated Learning"
        )

    systems.append(
        "Route Map Matching"
    )

    return {

        "status":
            "RideGuardian AI API running",

        "systems":
            systems
    }


# ============================================================
# ADDITIONAL HEALTH CHECK
# ============================================================

@app.get("/health")

def health():

    return {

        "status":
            "healthy",

        "modelLoaded":
            MODEL_AVAILABLE,

        "federatedLearning":
            (
                "available"
                if FL_AVAILABLE
                else "unavailable"
            ),

        "vehicleAI":
            (
                "available"
                if VEHICLE_AI_AVAILABLE
                else "unavailable"
            )
    }


# ============================================================
# HELPER FUNCTION
# LIMIT SCORE BETWEEN 0 AND 100
# ============================================================

def limit_score(value):

    return round(
        max(
            0,
            min(
                100,
                value
            )
        ),
        2
    )


# ============================================================
# VEHICLE WEAR / STRESS ENGINE
#
# ORIGINAL FUNCTION PRESERVED
# ============================================================

def calculate_vehicle_wear(

    data,

    acceleration_magnitude,

    gyro_magnitude,

    risk_probability
):

    # --------------------------------------------------------
    # NORMALIZED VALUES
    # --------------------------------------------------------

    speed_factor = min(

        data.speedKmh / 100,

        1
    )


    acceleration_factor = min(

        abs(
            data.speedAccelerationMs2
        ) / 10,

        1
    )


    gyro_factor = min(

        gyro_magnitude / 10,

        1
    )


    vibration_factor = min(

        abs(
            acceleration_magnitude - 9.81
        ) / 15,

        1
    )


    # ========================================================
    # BRAKE WEAR STRESS
    # ========================================================

    brake_stress = 0

    if data.speedAccelerationMs2 < -0.5:

        brake_stress = (

            abs(
                data.speedAccelerationMs2
            ) * 12

            + speed_factor * 25

            + risk_probability * 0.20
        )


    brake_stress = limit_score(
        brake_stress
    )


    # ========================================================
    # TYRE WEAR STRESS
    # ========================================================

    tyre_stress = (

        speed_factor * 30

        + gyro_factor * 35

        + acceleration_factor * 20

        + risk_probability * 0.15
    )


    tyre_stress = limit_score(
        tyre_stress
    )


    # ========================================================
    # SUSPENSION STRESS
    # ========================================================

    suspension_stress = (

        vibration_factor * 55

        + gyro_factor * 20

        + speed_factor * 15
    )


    suspension_stress = limit_score(
        suspension_stress
    )


    # ========================================================
    # DRIVETRAIN STRESS
    # ========================================================

    drivetrain_stress = (

        acceleration_factor * 55

        + abs(
            data.speedChangeKmh
        ) / 20 * 25

        + risk_probability * 0.15
    )


    drivetrain_stress = limit_score(
        drivetrain_stress
    )


    # ========================================================
    # OVERALL VEHICLE STRESS
    # ========================================================

    overall_stress = (

        brake_stress * 0.25

        + tyre_stress * 0.30

        + suspension_stress * 0.25

        + drivetrain_stress * 0.20
    )


    overall_stress = limit_score(
        overall_stress
    )


    # ========================================================
    # DETERMINE VEHICLE CONDITION
    # ========================================================

    if overall_stress < 25:

        condition = "LOW_STRESS"

    elif overall_stress < 50:

        condition = "MODERATE_STRESS"

    elif overall_stress < 75:

        condition = "HIGH_STRESS"

    else:

        condition = "SEVERE_STRESS"


    return {

        "overallStress":
            overall_stress,

        "condition":
            condition,

        "tyreWearRisk":
            tyre_stress,

        "brakeWearRisk":
            brake_stress,

        "suspensionStress":
            suspension_stress,

        "drivetrainStress":
            drivetrain_stress
    }


# ============================================================
# AI PREDICTION ENDPOINT
#
# The original prediction logic is preserved.
# It only runs when the Riding Risk model exists.
# ============================================================

@app.post("/predict")

def predict(
    data: SensorData
):

    if not MODEL_AVAILABLE:

        raise HTTPException(

            status_code=503,

            detail=(
                "Riding Risk ML model is not installed "
                "in this backend. "
                "Vehicle AI is available at "
                "/predict/vehicle."
            )
        )


    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    acceleration_magnitude = np.sqrt(

        data.accelerationX ** 2

        + data.accelerationY ** 2

        + data.accelerationZ ** 2
    )


    gyro_magnitude_calculated = np.sqrt(

        data.gyroX ** 2

        + data.gyroY ** 2

        + data.gyroZ ** 2
    )


    # ========================================================
    # CREATE FEATURES
    # ========================================================

    feature_values = {

        "speedKmh":
            data.speedKmh,


        "accelerationX":
            data.accelerationX,

        "accelerationY":
            data.accelerationY,

        "accelerationZ":
            data.accelerationZ,


        "gyroX":
            data.gyroX,

        "gyroY":
            data.gyroY,

        "gyroZ":
            data.gyroZ,


        "timeSincePreviousReadingSec":
            data.timeSincePreviousReadingSec,


        "speedChangeKmh":
            data.speedChangeKmh,


        "speedAccelerationMs2":
            data.speedAccelerationMs2,


        "gyroMagnitudeRadS":
            data.gyroMagnitudeRadS,


        "accelerationMagnitude":
            acceleration_magnitude,


        "gyroMagnitudeCalculated":
            gyro_magnitude_calculated,


        "abs_accelerationX":
            abs(
                data.accelerationX
            ),

        "abs_accelerationY":
            abs(
                data.accelerationY
            ),

        "abs_accelerationZ":
            abs(
                data.accelerationZ
            ),


        "abs_gyroX":
            abs(
                data.gyroX
            ),

        "abs_gyroY":
            abs(
                data.gyroY
            ),

        "abs_gyroZ":
            abs(
                data.gyroZ
            ),


        "abs_speedAccelerationMs2":
            abs(
                data.speedAccelerationMs2
            )
    }


    # ========================================================
    # CREATE DATAFRAME WITH FEATURE NAMES
    # ========================================================

    input_data = pd.DataFrame(

        [[

            feature_values[
                feature
            ]

            for feature in features

        ]],

        columns=features
    )


    # ========================================================
    # RIDING RISK ML PREDICTION
    # ========================================================

    prediction = int(

        model.predict(
            input_data
        )[0]
    )


    probability = float(

        model.predict_proba(
            input_data
        )[0][1]
    )


    risk_probability = round(

        probability * 100,

        2
    )


    # ========================================================
    # VEHICLE WEAR / STRESS ANALYSIS
    # ========================================================

    vehicle_wear = calculate_vehicle_wear(

        data=data,

        acceleration_magnitude=
            acceleration_magnitude,

        gyro_magnitude=
            gyro_magnitude_calculated,

        risk_probability=
            risk_probability
    )


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "prediction":

            (
                "RISK_EVENT"

                if prediction == 1

                else "NORMAL"
            ),


        "riskProbability":

            risk_probability,


        "sensorAnalysis": {

            "accelerationMagnitude":

                round(

                    float(
                        acceleration_magnitude
                    ),

                    3
                ),


            "gyroMagnitude":

                round(

                    float(
                        gyro_magnitude_calculated
                    ),

                    3
                )
        },


        "vehicleWear":

            vehicle_wear
    }


# ============================================================
# ROUTE MAP MATCHING AND ROUTE CLEANING
#
# ORIGINAL FUNCTIONS PRESERVED
# ============================================================

class RoutePoint(
    BaseModel
):

    latitude: float

    longitude: float


class RouteMatchRequest(
    BaseModel
):

    points: List[
        RoutePoint
    ]


# ============================================================
# CALCULATE DISTANCE BETWEEN TWO GPS POINTS
# ============================================================

def route_distance_meters(

    point1: RoutePoint,

    point2: RoutePoint

) -> float:

    earth_radius = 6371000.0


    latitude1 = np.radians(
        point1.latitude
    )

    longitude1 = np.radians(
        point1.longitude
    )


    latitude2 = np.radians(
        point2.latitude
    )

    longitude2 = np.radians(
        point2.longitude
    )


    latitude_delta = (

        latitude2
        -
        latitude1
    )


    longitude_delta = (

        longitude2
        -
        longitude1
    )


    a = (

        np.sin(
            latitude_delta / 2
        ) ** 2

        +

        np.cos(
            latitude1
        )

        *

        np.cos(
            latitude2
        )

        *

        np.sin(
            longitude_delta / 2
        ) ** 2
    )


    return float(

        2

        *

        earth_radius

        *

        np.arctan2(

            np.sqrt(a),

            np.sqrt(
                1 - a
            )
        )
    )


# ============================================================
# REMOVE NEARLY DUPLICATE GPS POINTS
# ============================================================

def prepare_route_points(

    points: List[
        RoutePoint
    ]

) -> List[
    RoutePoint
]:

    if len(points) <= 2:

        return points


    cleaned_points = [

        points[0]
    ]


    for point in points[1:]:

        previous_point = (

            cleaned_points[-1]
        )


        distance = (

            route_distance_meters(

                previous_point,

                point
            )
        )


        # Ignore duplicate / extremely small movement.

        if distance >= 5:

            cleaned_points.append(
                point
            )


    # Always preserve the final point.

    final_point = points[-1]

    last_point = cleaned_points[-1]


    if (

        last_point.latitude
        !=
        final_point.latitude

        or

        last_point.longitude
        !=
        final_point.longitude
    ):

        cleaned_points.append(
            final_point
        )


    return cleaned_points


# ============================================================
# REMOVE OBVIOUS GPS JUMPS
# ============================================================

def remove_gps_jumps(

    points: List[
        RoutePoint
    ]

) -> List[
    RoutePoint
]:

    if len(points) <= 2:

        return points


    filtered_points = [

        points[0]
    ]


    for point in points[1:]:

        previous_point = (

            filtered_points[-1]
        )


        distance = (

            route_distance_meters(

                previous_point,

                point
            )
        )


        # Reject an isolated jump of more than 2 km.

        if distance <= 2000:

            filtered_points.append(
                point
            )


    if len(filtered_points) < 2:

        return points


    return filtered_points


# ============================================================
# SELECT POINTS FOR EXTERNAL ROUTING
# ============================================================

def select_route_points(

    points: List[
        RoutePoint
    ],

    maximum_points: int = 50

) -> List[
    RoutePoint
]:

    if len(points) <= maximum_points:

        return points


    selected_points = [

        points[0]
    ]


    step = (

        (
            len(points) - 1
        )

        /

        (
            maximum_points - 1
        )
    )


    previous_index = 0


    for index in range(

        1,

        maximum_points - 1
    ):

        point_index = round(

            index * step
        )


        point_index = max(

            previous_index + 1,

            point_index
        )


        point_index = min(

            point_index,

            len(points) - 2
        )


        selected_points.append(

            points[
                point_index
            ]
        )


        previous_index = point_index


    selected_points.append(
        points[-1]
    )


    return selected_points


# ============================================================
# CREATE OSRM COORDINATE STRING
# ============================================================

def create_osrm_coordinate_string(

    points: List[
        RoutePoint
    ]

) -> str:

    return ";".join(

        (
            f"{point.longitude},"
            f"{point.latitude}"
        )

        for point in points
    )


# ============================================================
# REMOVE CONSECUTIVE DUPLICATE ROUTE COORDINATES
# ============================================================

def remove_duplicate_coordinates(

    coordinates: list

) -> list:

    cleaned_coordinates = []


    for coordinate in coordinates:

        if len(coordinate) < 2:

            continue


        longitude = float(
            coordinate[0]
        )

        latitude = float(
            coordinate[1]
        )


        new_point = {

            "latitude":
                latitude,

            "longitude":
                longitude
        }


        if cleaned_coordinates:

            previous_point = (

                cleaned_coordinates[-1]
            )


            if (

                abs(

                    previous_point[
                        "latitude"
                    ]

                    -

                    latitude

                )

                < 0.0000001

                and

                abs(

                    previous_point[
                        "longitude"
                    ]

                    -

                    longitude

                )

                < 0.0000001
            ):

                continue


        cleaned_coordinates.append(
            new_point
        )


    return cleaned_coordinates


# ============================================================
# TRY OSRM MAP MATCHING
# ============================================================

def try_osrm_map_matching(

    points: List[
        RoutePoint
    ]

):

    if len(points) < 2:

        return None


    coordinate_string = (

        create_osrm_coordinate_string(
            points
        )
    )


    osrm_url = (

        "https://router.project-osrm.org/"

        +

        "match/v1/driving/"

        +

        coordinate_string
    )


    try:

        print(
            "Attempting OSRM map matching..."
        )


        response = requests.get(

            osrm_url,

            params={

                "overview":
                    "full",

                "geometries":
                    "geojson",

                "steps":
                    "false",

                "tidy":
                    "true",

                "gaps":
                    "ignore",

                "radiuses":

                    ";".join(

                        ["100"]
                        *
                        len(points)
                    )
            },

            headers={

                "User-Agent":

                    "RideGuardian-AI/1.0"
            },

            timeout=30
        )


        print(

            "OSRM MATCH STATUS:",

            response.status_code
        )


        if response.status_code != 200:

            print(

                "OSRM MATCH RESPONSE:",

                response.text[:500]
            )


            return None


        result = response.json()


        matchings = result.get(

            "matchings",

            []
        )


        if not matchings:

            return None


        all_coordinates = []


        for matching in matchings:

            geometry = matching.get(

                "geometry",

                {}
            )


            coordinates = geometry.get(

                "coordinates",

                []
            )


            all_coordinates.extend(
                coordinates
            )


        cleaned_coordinates = (

            remove_duplicate_coordinates(
                all_coordinates
            )
        )


        if len(
            cleaned_coordinates
        ) >= 2:

            return cleaned_coordinates


        return None


    except Exception as error:

        print(

            "OSRM MAP MATCH ERROR:",

            str(error)
        )


        return None


# ============================================================
# TRY OSRM ROUTE SNAP
# ============================================================

def try_osrm_route_snap(

    points: List[
        RoutePoint
    ]

):

    if len(points) < 2:

        return None


    coordinate_string = (

        create_osrm_coordinate_string(
            points
        )
    )


    osrm_url = (

        "https://router.project-osrm.org/"

        +

        "route/v1/driving/"

        +

        coordinate_string
    )


    try:

        print(
            "Attempting OSRM route snapping..."
        )


        response = requests.get(

            osrm_url,

            params={

                "overview":
                    "full",

                "geometries":
                    "geojson",

                "steps":
                    "false",

                "continue_straight":
                    "false"
            },

            headers={

                "User-Agent":

                    "RideGuardian-AI/1.0"
            },

            timeout=30
        )


        print(

            "OSRM ROUTE STATUS:",

            response.status_code
        )


        if response.status_code != 200:

            print(

                "OSRM ROUTE RESPONSE:",

                response.text[:500]
            )


            return None


        result = response.json()


        routes = result.get(

            "routes",

            []
        )


        if not routes:

            return None


        geometry = routes[0].get(

            "geometry",

            {}
        )


        coordinates = geometry.get(

            "coordinates",

            []
        )


        cleaned_coordinates = (

            remove_duplicate_coordinates(
                coordinates
            )
        )


        if len(
            cleaned_coordinates
        ) >= 2:

            return cleaned_coordinates


        return None


    except Exception as error:

        print(

            "OSRM ROUTE SNAP ERROR:",

            str(error)
        )


        return None


# ============================================================
# ROUTE MATCHING ENDPOINT
# ============================================================

@app.post("/match-route")

def match_route(

    request: RouteMatchRequest

):

    original_points = request.points


    # ========================================================
    # VALIDATION
    # ========================================================

    if len(original_points) < 2:

        return {

            "success":
                False,

            "matched":
                False,

            "matchingMethod":
                "NONE",

            "coordinates":
                [],

            "originalPointCount":
                len(
                    original_points
                ),

            "submittedPointCount":
                0,

            "matchedPointCount":
                0,

            "message":

                "At least two GPS points "
                "are required."
        }


    try:

        print(

            "========================================"
        )


        print(
            "ROUTE PROCESSING STARTED"
        )


        print(

            "ORIGINAL GPS POINTS:",

            len(
                original_points
            )
        )


        # STEP 1:
        # REMOVE DUPLICATES.

        prepared_points = (

            prepare_route_points(
                original_points
            )
        )


        # STEP 2:
        # REMOVE OBVIOUS GPS JUMPS.

        filtered_points = (

            remove_gps_jumps(
                prepared_points
            )
        )


        # STEP 3:
        # REDUCE POINTS ONLY FOR EXTERNAL ROUTING.

        submitted_points = (

            select_route_points(

                filtered_points,

                maximum_points=50
            )
        )


        print(

            "SUBMITTED ROUTE POINTS:",

            len(
                submitted_points
            )
        )


        if len(submitted_points) < 2:

            raise ValueError(

                "Not enough valid route points."
            )


        # STEP 4:
        # TRY TRUE MAP MATCHING.

        matched_coordinates = (

            try_osrm_map_matching(
                submitted_points
            )
        )


        if matched_coordinates:

            print(
                "MAP MATCHING SUCCESSFUL"
            )


            print(

                "MATCHED POINTS:",

                len(
                    matched_coordinates
                )
            )


            return {

                "success":
                    True,

                "matched":
                    True,

                "matchingMethod":
                    "OSRM_MAP_MATCH",

                "coordinates":
                    matched_coordinates,

                "originalPointCount":

                    len(
                        original_points
                    ),

                "submittedPointCount":

                    len(
                        submitted_points
                    ),

                "matchedPointCount":

                    len(
                        matched_coordinates
                    ),

                "message":

                    "Route successfully "
                    "matched to roads."
            }


        # STEP 5:
        # FALL BACK TO OSRM ROUTE SNAP.

        print(

            "MAP MATCH FAILED. "
            "TRYING ROUTE SNAP..."
        )


        routed_coordinates = (

            try_osrm_route_snap(
                submitted_points
            )
        )


        if routed_coordinates:

            print(
                "ROUTE SNAP SUCCESSFUL"
            )


            print(

                "ROUTED POINTS:",

                len(
                    routed_coordinates
                )
            )


            return {

                "success":
                    True,

                "matched":
                    True,

                "matchingMethod":
                    "OSRM_ROUTE_SNAP",

                "coordinates":
                    routed_coordinates,

                "originalPointCount":

                    len(
                        original_points
                    ),

                "submittedPointCount":

                    len(
                        submitted_points
                    ),

                "matchedPointCount":

                    len(
                        routed_coordinates
                    ),

                "message":

                    "Route was snapped "
                    "to nearby roads."
            }


        # FINAL FALLBACK:
        # NEVER LOSE THE USER'S RIDE.

        print(
            "ALL MAP SERVICES FAILED."
        )


        return {

            "success":
                True,

            "matched":
                False,

            "matchingMethod":
                "ORIGINAL_GPS",

            "coordinates": [

                {

                    "latitude":
                        point.latitude,

                    "longitude":
                        point.longitude
                }

                for point
                in original_points
            ],

            "originalPointCount":

                len(
                    original_points
                ),

            "submittedPointCount":

                len(
                    submitted_points
                ),

            "matchedPointCount":

                len(
                    original_points
                ),

            "message":

                "Road matching unavailable. "
                "Displaying the original GPS route."
        }


    except Exception as error:

        print(

            "========================================"
        )


        print(

            "ROUTE PROCESSING ERROR:",

            str(error)
        )


        print(

            "========================================"
        )


        return {

            "success":
                False,

            "matched":
                False,

            "matchingMethod":
                "ORIGINAL_GPS",

            "coordinates": [

                {

                    "latitude":
                        point.latitude,

                    "longitude":
                        point.longitude
                }

                for point
                in original_points
            ],

            "originalPointCount":

                len(
                    original_points
                ),

            "submittedPointCount":
                0,

            "matchedPointCount":

                len(
                    original_points
                ),

            "message":

                "Route processing failed: "

                +

                str(error)
        }


# ============================================================
# RAILWAY / PRODUCTION SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn


    port = int(

        os.environ.get(
            "PORT",
            8002
        )
    )


    print(

        f"Starting RideGuardian AI "
        f"server on port {port}"
    )


    uvicorn.run(

        app,

        host="0.0.0.0",

        port=port
    )