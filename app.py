# ============================================================
# SIGNOVA
# Real-Time Hand Sign Recognition & Sentence Builder
#
# Features:
# - Real-time webcam
# - MediaPipe 21-point hand landmark detection
# - Red landmark points
# - White skeleton / vectors
# - Hand centroid
# - Fingertip trajectory
# - Green bounding box when hand detected
# - Dataset-based KNN classification
# - A-Z and 0-9 recognition
# - Stability detection
# - Sentence / letter box
# - Clear / Delete / Space buttons
# - Dataset diagnostics
# - Safe train/test evaluation
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import zipfile
import threading
import time
from collections import deque

import av
import mediapipe as mp
import numpy as np
import streamlit as st

from PIL import Image, ImageDraw

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode
)


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SIGNOVA",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SETTINGS
# ============================================================

ZIP_CANDIDATES = [
    "archive.zip",
    "archive(1).zip"
]

EXTRACT_FOLDER = "signova_dataset"

EXPECTED_CLASSES = (
    [str(i) for i in range(10)]
    +
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
)

ORIGINAL_IMAGES_PER_CLASS = 25
ORIGINAL_IMAGE_COUNT = 900

# Recognition settings
CONFIDENCE_THRESHOLD = 0.65

# Number of consecutive frames required
# before a sign is accepted.
STABLE_FRAMES = 8

# Minimum time between recorded signs.
RECORD_COOLDOWN = 0.70

# Maximum trajectory history
TRAJECTORY_LENGTH = 18


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
<style>

    /* ======================================================
       MAIN APPLICATION
    ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 12% 8%,
                rgba(124, 58, 237, 0.18),
                transparent 28%
            ),
            radial-gradient(
                circle at 88% 15%,
                rgba(14, 165, 233, 0.12),
                transparent 28%
            ),
            #070a12;

        color: #f8fafc;
    }


    [data-testid="stHeader"] {
        background: transparent;
    }


    #MainMenu {
        visibility: hidden;
    }


    footer {
        visibility: hidden;
    }


    /* ======================================================
       SIDEBAR
    ====================================================== */

    [data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #0b0f1a 0%,
                #070a12 100%
            );

        border-right:
            1px solid rgba(255,255,255,0.07);
    }


    /* ======================================================
       BRAND
    ====================================================== */

    .brand-icon {

        width: 58px;
        height: 58px;

        border-radius: 17px;

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 30px;

        background:
            linear-gradient(
                135deg,
                #7c3aed,
                #2563eb
            );

        box-shadow:
            0 10px 35px rgba(37,99,235,0.25);
    }


    .brand-title {

        font-size: 28px;

        font-weight: 900;

        margin-top: 13px;

        letter-spacing: 1px;
    }


    .brand-subtitle {

        color: #64748b;

        font-size: 12px;

        line-height: 1.6;
    }


    /* ======================================================
       HERO
    ====================================================== */

    .eyebrow {

        color: #818cf8;

        font-size: 11px;

        font-weight: 900;

        letter-spacing: 2px;

        text-transform: uppercase;
    }


    .hero-title {

        font-size: 54px;

        font-weight: 950;

        letter-spacing: -2px;

        margin-top: 3px;

        background:
            linear-gradient(
                90deg,
                #ffffff,
                #c4b5fd,
                #7dd3fc
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;
    }


    .hero-subtitle {

        color: #94a3b8;

        font-size: 16px;

        line-height: 1.7;

        max-width: 800px;

        margin-bottom: 20px;
    }


    /* ======================================================
       SENTENCE BOX
    ====================================================== */

    .sentence-box {

        margin: 18px 0;

        padding: 22px 25px;

        min-height: 105px;

        border-radius: 20px;

        background:
            linear-gradient(
                145deg,
                rgba(124,58,237,0.16),
                rgba(37,99,235,0.08)
            );

        border:
            1px solid rgba(139,92,246,0.30);

        box-shadow:
            0 20px 50px rgba(0,0,0,0.20);
    }


    .sentence-label {

        color: #818cf8;

        font-size: 10px;

        font-weight: 900;

        letter-spacing: 2px;

        text-transform: uppercase;
    }


    .sentence {

        min-height: 55px;

        display: flex;

        align-items: center;

        font-size: 36px;

        font-weight: 900;

        letter-spacing: 5px;

        margin-top: 8px;

        word-break: break-word;
    }


    .sentence-empty {

        color: #475569;

        font-size: 16px;

        font-weight: 500;

        letter-spacing: 0;
    }


    /* ======================================================
       CARDS
    ====================================================== */

    .card {

        padding: 20px;

        border-radius: 19px;

        background:
            rgba(15,23,42,0.76);

        border:
            1px solid rgba(255,255,255,0.07);

        box-shadow:
            0 18px 40px rgba(0,0,0,0.18);
    }


    .card-title {

        font-size: 18px;

        font-weight: 800;

        margin-bottom: 5px;
    }


    .card-text {

        color: #94a3b8;

        font-size: 13px;

        line-height: 1.7;
    }


    /* ======================================================
       PREDICTION
    ====================================================== */

    .prediction {

        text-align: center;

        padding: 27px;

        border-radius: 22px;

        background:
            linear-gradient(
                145deg,
                rgba(30,41,59,0.90),
                rgba(15,23,42,0.90)
            );

        border:
            1px solid rgba(255,255,255,0.08);
    }


    .prediction-label {

        color: #64748b;

        font-size: 10px;

        font-weight: 900;

        letter-spacing: 2px;

        text-transform: uppercase;
    }


    .prediction-letter {

        font-size: 105px;

        font-weight: 950;

        line-height: 1;

        margin: 15px;

        background:
            linear-gradient(
                135deg,
                #c4b5fd,
                #60a5fa
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;
    }


    .confidence {

        color: #cbd5e1;

        font-size: 16px;
    }


    /* ======================================================
       DETECTION STATUS
    ====================================================== */

    .detected {

        color: #4ade80;

        font-weight: 900;
    }


    .waiting {

        color: #64748b;

        font-weight: 900;
    }


    /* ======================================================
       METRIC
    ====================================================== */

    .metric {

        background:
            rgba(15,23,42,0.76);

        border:
            1px solid rgba(255,255,255,0.07);

        border-radius: 17px;

        padding: 18px;

        min-height: 95px;
    }


    .metric-value {

        font-size: 28px;

        font-weight: 900;

        background:
            linear-gradient(
                90deg,
                #a78bfa,
                #38bdf8
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;
    }


    .metric-label {

        color: #64748b;

        font-size: 10px;

        font-weight: 800;

        letter-spacing: 1px;

        margin-top: 5px;

        text-transform: uppercase;
    }


    /* ======================================================
       INFORMATION
    ====================================================== */

    .info {

        background:
            rgba(99,102,241,0.07);

        border-left:
            3px solid #6366f1;

        border-radius: 9px;

        padding: 17px;

        color: #cbd5e1;

        font-size: 13px;

        line-height: 1.8;
    }


    /* ======================================================
       PIPELINE
    ====================================================== */

    .pipeline {

        display: flex;

        align-items: center;

        gap: 8px;

        flex-wrap: wrap;

        margin: 20px 0;
    }


    .pipeline-step {

        background:
            rgba(30,41,59,0.75);

        border:
            1px solid rgba(255,255,255,0.07);

        border-radius: 10px;

        padding: 10px 13px;

        font-size: 12px;

        color: #cbd5e1;

        font-weight: 700;
    }


    .pipeline-arrow {

        color: #6366f1;

        font-weight: 900;
    }


    /* ======================================================
       SMALL STATUS
    ====================================================== */

    .live-status {

        padding: 13px 16px;

        border-radius: 14px;

        background:
            rgba(15,23,42,0.75);

        border:
            1px solid rgba(255,255,255,0.07);

        margin-top: 10px;
    }

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands


# ============================================================
# HAND CONNECTIONS
# ============================================================

# These connections form the hand skeleton / vectors.

HAND_CONNECTIONS = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17)
]


# ============================================================
# FIND ZIP FILE
# ============================================================

def find_zip_file():

    for filename in ZIP_CANDIDATES:

        if os.path.exists(filename):
            return filename

    return None


# ============================================================
# FIND DATASET DIRECTORY
# ============================================================

def find_dataset_directory():

    if not os.path.exists(EXTRACT_FOLDER):
        return None


    # Expected path first

    direct_path = os.path.join(
        EXTRACT_FOLDER,
        "DATASET"
    )

    if os.path.isdir(direct_path):
        return direct_path


    # Recursive fallback

    for root, dirs, files in os.walk(
        EXTRACT_FOLDER
    ):

        if os.path.basename(root).upper() == "DATASET":
            return root


    return None


# ============================================================
# EXTRACT ZIP
# ============================================================

def prepare_dataset():

    existing_dataset = find_dataset_directory()

    if existing_dataset is not None:
        return existing_dataset


    zip_file = find_zip_file()


    if zip_file is None:

        st.error(
            """
            SIGNOVA cannot find the dataset ZIP.

            Make sure your GitHub repository contains:

            archive.zip
            """
        )

        st.stop()


    try:

        os.makedirs(
            EXTRACT_FOLDER,
            exist_ok=True
        )


        with zipfile.ZipFile(
            zip_file,
            "r"
        ) as archive:

            archive.extractall(
                EXTRACT_FOLDER
            )


    except zipfile.BadZipFile:

        st.error(
            "The dataset ZIP appears to be corrupted."
        )

        st.stop()


    dataset_folder = find_dataset_directory()


    if dataset_folder is None:

        st.error(
            """
            SIGNOVA could not locate the DATASET folder.

            Expected:

            DATASET/
                A/
                B/
                ...
                0/
                1/
            """
        )

        st.stop()


    return dataset_folder


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def create_features(landmarks):

    """
    Convert MediaPipe's 21 landmarks into a normalized
    numerical feature vector.

    Features include:

    1. Normalized X/Y/Z landmark coordinates
    2. Connected-joint vectors
    3. Selected pairwise distances

    This makes classification less dependent on the
    hand's absolute position in the image.
    """


    points = np.asarray(
        [
            [
                landmark.x,
                landmark.y,
                landmark.z
            ]

            for landmark in landmarks
        ],

        dtype=np.float32
    )


    # ========================================================
    # 1. TRANSLATE RELATIVE TO WRIST
    # ========================================================

    wrist = points[0].copy()

    points = points - wrist


    # ========================================================
    # 2. NORMALIZE SIZE
    # ========================================================

    distances_from_wrist = np.linalg.norm(
        points[:, :2],
        axis=1
    )


    scale = float(
        np.max(
            distances_from_wrist
        )
    )


    if scale > 1e-6:

        points = points / scale


    # ========================================================
    # 3. MIRROR NORMALIZATION
    #
    # Makes left/right mirrored hand images more comparable.
    # ========================================================

    # Index MCP = 5
    # Pinky MCP = 17

    if points[5, 0] > points[17, 0]:

        points[:, 0] *= -1


    # ========================================================
    # 4. JOINT VECTORS
    # ========================================================

    vector_features = []


    for point_a, point_b in HAND_CONNECTIONS:

        vector = (
            points[point_b]
            -
            points[point_a]
        )


        vector_features.extend(
            vector.tolist()
        )


    vector_features = np.asarray(
        vector_features,
        dtype=np.float32
    )


    # ========================================================
    # 5. USEFUL DISTANCES
    # ========================================================

    # Fingertips
    fingertips = [
        4,
        8,
        12,
        16,
        20
    ]


    distance_features = []


    # Distances from each fingertip to wrist

    for tip in fingertips:

        distance = np.linalg.norm(
            points[tip, :2]
        )

        distance_features.append(
            distance
        )


    # Distances between neighbouring fingertips

    for i in range(
        len(fingertips) - 1
    ):

        point1 = points[
            fingertips[i],
            :2
        ]

        point2 = points[
            fingertips[i + 1],
            :2
        ]


        distance = np.linalg.norm(
            point1 - point2
        )


        distance_features.append(
            distance
        )


    distance_features = np.asarray(
        distance_features,
        dtype=np.float32
    )


    # ========================================================
    # FINAL FEATURE VECTOR
    # ========================================================

    features = np.concatenate(
        [
            points.flatten(),
            vector_features,
            distance_features
        ]
    )


    return features.astype(
        np.float32
    )


# ============================================================
# LOAD DATASET + EXTRACT LANDMARKS
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_landmark_dataset(dataset_path):

    X = []
    y = []

    detected_counts = {
        class_name: 0
        for class_name in EXPECTED_CLASSES
    }


    failed_counts = {
        class_name: 0
        for class_name in EXPECTED_CLASSES
    }


    with mp_hands.Hands(

        static_image_mode=True,

        max_num_hands=1,

        model_complexity=1,

        min_detection_confidence=0.35

    ) as detector:


        for class_name in EXPECTED_CLASSES:

            class_folder = os.path.join(
                dataset_path,
                class_name
            )


            if not os.path.isdir(
                class_folder
            ):

                continue


            image_files = sorted(
                [
                    filename

                    for filename
                    in os.listdir(
                        class_folder
                    )

                    if filename.lower().endswith(
                        (
                            ".jpg",
                            ".jpeg",
                            ".png"
                        )
                    )
                ]
            )


            for filename in image_files:

                image_path = os.path.join(
                    class_folder,
                    filename
                )


                try:

                    image = Image.open(
                        image_path
                    ).convert(
                        "RGB"
                    )


                    image_array = np.asarray(
                        image
                    )


                    result = detector.process(
                        image_array
                    )


                    if not result.multi_hand_landmarks:

                        failed_counts[
                            class_name
                        ] += 1

                        continue


                    landmarks = (
                        result
                        .multi_hand_landmarks[0]
                        .landmark
                    )


                    features = create_features(
                        landmarks
                    )


                    X.append(
                        features
                    )

                    y.append(
                        class_name
                    )


                    detected_counts[
                        class_name
                    ] += 1


                except Exception:

                    failed_counts[
                        class_name
                    ] += 1


    return (
        np.asarray(
            X,
            dtype=np.float32
        ),
        np.asarray(y),
        detected_counts,
        failed_counts
    )


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def train_model(X, y):

    """
    The live recognition model is trained with ALL usable
    landmark samples.

    Accuracy evaluation uses a separate manually-balanced
    split only for classes with >= 2 detected samples.

    This avoids the previous train_test_split ValueError.
    """


    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(y)


    # ========================================================
    # CHECK DATA
    # ========================================================

    if len(X) == 0:

        raise ValueError(
            "No usable landmark samples were extracted."
        )


    # ========================================================
    # FINAL LIVE MODEL
    # ========================================================

    final_neighbors = max(
        1,
        min(
            5,
            len(X)
        )
    )


    final_model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=final_neighbors,
                    weights="distance",
                    metric="euclidean"
                )
            )
        ]
    )


    final_model.fit(
        X,
        y
    )


    # ========================================================
    # FIND CLASSES WITH AT LEAST 2 SAMPLES
    # ========================================================

    unique_classes, sample_counts = np.unique(
        y,
        return_counts=True
    )


    eligible_classes = unique_classes[
        sample_counts >= 2
    ]


    # Not enough classes for useful evaluation

    if len(eligible_classes) < 2:

        return (
            final_model,
            None,
            len(X),
            0,
            []
        )


    # ========================================================
    # ONLY USE ELIGIBLE CLASSES FOR ACCURACY EVALUATION
    # ========================================================

    evaluation_mask = np.isin(
        y,
        eligible_classes
    )


    X_eval = X[
        evaluation_mask
    ]


    y_eval = y[
        evaluation_mask
    ]


    # ========================================================
    # MANUAL BALANCED TRAIN / TEST SPLIT
    # ========================================================

    rng = np.random.default_rng(
        42
    )


    train_indices = []
    test_indices = []


    for class_name in eligible_classes:

        class_indices = np.where(
            y_eval == class_name
        )[0]


        rng.shuffle(
            class_indices
        )


        # Use around 20% for testing,
        # but always at least one.

        test_count = max(
            1,
            int(
                round(
                    len(class_indices)
                    *
                    0.20
                )
            )
        )


        # Always keep at least one
        # training sample.

        test_count = min(
            test_count,
            len(class_indices) - 1
        )


        test_indices.extend(
            class_indices[
                :test_count
            ]
        )


        train_indices.extend(
            class_indices[
                test_count:
            ]
        )


    train_indices = np.asarray(
        train_indices,
        dtype=int
    )


    test_indices = np.asarray(
        test_indices,
        dtype=int
    )


    # ========================================================
    # CHECK SPLIT
    # ========================================================

    if (
        len(train_indices) == 0
        or
        len(test_indices) == 0
    ):

        return (
            final_model,
            None,
            len(X),
            0,
            eligible_classes.tolist()
        )


    X_train = X_eval[
        train_indices
    ]


    y_train = y_eval[
        train_indices
    ]


    X_test = X_eval[
        test_indices
    ]


    y_test = y_eval[
        test_indices
    ]


    # ========================================================
    # EVALUATION MODEL
    # ========================================================

    evaluation_neighbors = max(
        1,
        min(
            5,
            len(X_train)
        )
    )


    evaluation_model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=evaluation_neighbors,
                    weights="distance",
                    metric="euclidean"
                )
            )
        ]
    )


    evaluation_model.fit(
        X_train,
        y_train
    )


    predictions = evaluation_model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    return (
        final_model,
        accuracy,
        len(X_train),
        len(X_test),
        eligible_classes.tolist()
    )


# ============================================================
# PREPARE DATASET
# ============================================================

dataset_path = prepare_dataset()


# ============================================================
# LOAD LANDMARK DATA
# ============================================================

with st.spinner(
    "SIGNOVA is analysing the hand landmarks in your dataset..."
):

    (
        X,
        y,
        detected_counts,
        failed_counts
    ) = load_landmark_dataset(
        dataset_path
    )


# ============================================================
# VALIDATE LANDMARK DATA
# ============================================================

if len(X) == 0:

    st.error(
        """
        SIGNOVA could not extract any usable hand landmarks
        from the supplied dataset.

        Check that the dataset images clearly contain hands.
        """
    )

    st.stop()


# ============================================================
# TRAIN
# ============================================================

with st.spinner(
    "SIGNOVA is training the hand-sign classifier..."
):

    (
        model,
        accuracy,
        train_count,
        test_count,
        evaluation_classes
    ) = train_model(
        X,
        y
    )


# ============================================================
# ACCURACY DISPLAY
# ============================================================

if accuracy is None:

    accuracy_text = "N/A"

else:

    accuracy_text = (
        f"{accuracy * 100:.1f}%"
    )


# ============================================================
# DATASET STATISTICS
# ============================================================

usable_landmark_count = len(X)


detection_rate = (
    usable_landmark_count
    /
    ORIGINAL_IMAGE_COUNT
    *
    100
)


detected_classes = sorted(
    np.unique(y).tolist()
)


missing_classes = [
    class_name

    for class_name in EXPECTED_CLASSES

    if class_name
    not in detected_classes
]


# ============================================================
# SHARED REAL-TIME STATE CLASS
# ============================================================

class SharedRecognitionState:

    def __init__(self):

        self.lock = threading.Lock()

        self.hand_detected = False

        self.prediction = "-"

        self.confidence = 0.0

        self.sentence = ""

        self.stable_prediction = None

        self.stable_count = 0

        self.last_recorded = None

        self.last_record_time = 0.0

        self.total_recorded = 0


# ============================================================
# CREATE / KEEP SHARED STATE
# ============================================================

if "recognition_state" not in st.session_state:

    st.session_state.recognition_state = (
        SharedRecognitionState()
    )


shared_state = (
    st.session_state.recognition_state
)


# ============================================================
# DRAW CAMERA OVERLAY
# ============================================================

def draw_hand_overlay(
    frame_array,
    landmarks,
    prediction,
    confidence,
    trajectory
):

    image = Image.fromarray(
        frame_array
    ).convert(
        "RGB"
    )


    draw = ImageDraw.Draw(
        image
    )


    width, height = image.size


    # ========================================================
    # LANDMARK COORDINATES
    # ========================================================

    coordinates = []


    for landmark in landmarks:

        x = int(
            landmark.x
            *
            width
        )


        y_point = int(
            landmark.y
            *
            height
        )


        coordinates.append(
            (
                x,
                y_point
            )
        )


    xs = [
        point[0]
        for point in coordinates
    ]


    ys = [
        point[1]
        for point in coordinates
    ]


    # ========================================================
    # GREEN HAND BOUNDING BOX
    # ========================================================

    left = max(
        min(xs) - 25,
        0
    )


    top = max(
        min(ys) - 25,
        0
    )


    right = min(
        max(xs) + 25,
        width
    )


    bottom = min(
        max(ys) + 25,
        height
    )


    draw.rectangle(
        [
            left,
            top,
            right,
            bottom
        ],

        outline=(0, 255, 80),

        width=5
    )


    # ========================================================
    # HAND DETECTED LABEL
    # ========================================================

    label_top = max(
        0,
        top - 30
    )


    draw.rectangle(
        [
            left,
            label_top,
            min(
                left + 160,
                width
            ),
            top
        ],

        fill=(0, 180, 70)
    )


    draw.text(
        (
            left + 8,
            label_top + 6
        ),

        "HAND DETECTED",

        fill="white"
    )


    # ========================================================
    # WHITE SKELETON / VECTORS
    # ========================================================

    for point_a, point_b in HAND_CONNECTIONS:

        x1, y1 = coordinates[
            point_a
        ]


        x2, y2 = coordinates[
            point_b
        ]


        draw.line(
            [
                (x1, y1),
                (x2, y2)
            ],

            fill=(255, 255, 255),

            width=3
        )


    # ========================================================
    # RED LANDMARK POINTS
    # ========================================================

    for index, (
        x,
        y_point
    ) in enumerate(coordinates):

        radius = 6


        draw.ellipse(
            [
                x - radius,
                y_point - radius,
                x + radius,
                y_point + radius
            ],

            fill=(255, 30, 30),

            outline=(255,255,255),

            width=1
        )


    # ========================================================
    # HAND CENTROID
    # ========================================================

    centroid_x = int(
        sum(xs)
        /
        len(xs)
    )


    centroid_y = int(
        sum(ys)
        /
        len(ys)
    )


    centroid_radius = 9


    draw.ellipse(
        [
            centroid_x - centroid_radius,
            centroid_y - centroid_radius,
            centroid_x + centroid_radius,
            centroid_y + centroid_radius
        ],

        fill=(255, 0, 0),

        outline=(255, 255, 0),

        width=3
    )


    # ========================================================
    # INDEX FINGER TRAJECTORY
    #
    # Landmark 8 = index fingertip.
    # ========================================================

    if len(trajectory) >= 2:

        trajectory_points = list(
            trajectory
        )


        for i in range(
            1,
            len(trajectory_points)
        ):

            previous = trajectory_points[
                i - 1
            ]


            current = trajectory_points[
                i
            ]


            draw.line(
                [
                    previous,
                    current
                ],

                fill=(0, 220, 255),

                width=3
            )


    # ========================================================
    # PREDICTION PANEL
    # ========================================================

    panel_left = 15
    panel_top = 15
    panel_right = 235
    panel_bottom = 105


    draw.rounded_rectangle(
        [
            panel_left,
            panel_top,
            panel_right,
            panel_bottom
        ],

        radius=14,

        fill=(8, 15, 30),

        outline=(0, 255, 80),

        width=3
    )


    draw.text(
        (
            30,
            30
        ),

        f"SIGN: {prediction}",

        fill=(255,255,255)
    )


    draw.text(
        (
            30,
            58
        ),

        f"CONFIDENCE: {confidence * 100:.1f}%",

        fill=(100,255,140)
    )


    return np.asarray(
        image
    )


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class SignovaVideoProcessor:

    def __init__(
        self,
        classifier,
        recognition_state
    ):

        self.classifier = classifier

        self.state = recognition_state


        # Create MediaPipe detector ONCE.

        self.hands = mp_hands.Hands(

            static_image_mode=False,

            max_num_hands=1,

            model_complexity=1,

            min_detection_confidence=0.60,

            min_tracking_confidence=0.60
        )


        # Processing lock in case WebRTC uses
        # concurrent callbacks.

        self.processing_lock = (
            threading.Lock()
        )


        # Index fingertip trajectory.

        self.trajectory = deque(
            maxlen=TRAJECTORY_LENGTH
        )


    # ========================================================
    # RECEIVE VIDEO FRAME
    # ========================================================

    def recv(
        self,
        frame
    ):

        with self.processing_lock:

            frame_array = frame.to_ndarray(
                format="rgb24"
            )


            # =================================================
            # MEDIAPIPE DETECTION
            # =================================================

            result = self.hands.process(
                frame_array
            )


            # =================================================
            # NO HAND
            # =================================================

            if not result.multi_hand_landmarks:

                self.trajectory.clear()


                with self.state.lock:

                    self.state.hand_detected = False

                    self.state.prediction = "-"

                    self.state.confidence = 0.0

                    self.state.stable_prediction = None

                    self.state.stable_count = 0


                    # Important:
                    # allows the same sign to be
                    # recorded again after hand removal.

                    self.state.last_recorded = None


                return av.VideoFrame.from_ndarray(
                    frame_array,
                    format="rgb24"
                )


            # =================================================
            # HAND FOUND
            # =================================================

            landmarks = (
                result
                .multi_hand_landmarks[0]
                .landmark
            )


            # =================================================
            # FEATURE EXTRACTION
            # =================================================

            features = create_features(
                landmarks
            )


            features = features.reshape(
                1,
                -1
            )


            # =================================================
            # KNN PROBABILITIES
            # =================================================

            probabilities = (
                self.classifier
                .predict_proba(
                    features
                )[0]
            )


            best_index = int(
                np.argmax(
                    probabilities
                )
            )


            prediction = str(
                self.classifier
                .classes_[
                    best_index
                ]
            )


            confidence = float(
                probabilities[
                    best_index
                ]
            )


            current_time = (
                time.time()
            )


            # =================================================
            # UPDATE TRAJECTORY
            # =================================================

            height, width, _ = (
                frame_array.shape
            )


            index_tip = landmarks[8]


            tip_x = int(
                index_tip.x
                *
                width
            )


            tip_y = int(
                index_tip.y
                *
                height
            )


            self.trajectory.append(
                (
                    tip_x,
                    tip_y
                )
            )


            # =================================================
            # UPDATE SHARED STATE
            # =================================================

            with self.state.lock:

                self.state.hand_detected = True

                self.state.prediction = prediction

                self.state.confidence = confidence


                # =============================================
                # STABILITY DETECTION
                # =============================================

                if (
                    self.state.stable_prediction
                    ==
                    prediction
                ):

                    self.state.stable_count += 1


                else:

                    self.state.stable_prediction = (
                        prediction
                    )

                    self.state.stable_count = 1


                # =============================================
                # ADD CHARACTER
                # =============================================

                if (

                    confidence
                    >=
                    CONFIDENCE_THRESHOLD

                    and

                    self.state.stable_count
                    >=
                    STABLE_FRAMES

                    and

                    self.state.last_recorded
                    !=
                    prediction

                    and

                    (
                        current_time
                        -
                        self.state.last_record_time
                    )
                    >=
                    RECORD_COOLDOWN

                ):

                    self.state.sentence += (
                        prediction
                    )


                    self.state.last_recorded = (
                        prediction
                    )


                    self.state.last_record_time = (
                        current_time
                    )


                    self.state.total_recorded += 1


            # =================================================
            # DRAW CAMERA OVERLAY
            # =================================================

            output_frame = draw_hand_overlay(

                frame_array,

                landmarks,

                prediction,

                confidence,

                self.trajectory
            )


            return av.VideoFrame.from_ndarray(
                output_frame,
                format="rgb24"
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand-icon">
            🤟
        </div>

        <div class="brand-title">
            SIGNOVA
        </div>

        <div class="brand-subtitle">
            Real-Time Hand Sign Recognition
            <br>
            Image Processing & Computer Vision
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("---")


    page = st.radio(
        "NAVIGATION",

        [
            "🎥 Live Translator",
            "📊 Dataset Lab",
            "🧠 Model Insights",
            "👁 Computer Vision",
            "ℹ️ About"
        ]
    )


    st.markdown("---")


    st.markdown(
        "### System Status"
    )


    st.success(
        "Recognition Engine Online"
    )


    st.caption(
        f"{len(detected_classes)} / 36 classes detected"
    )


    st.caption(
        f"{usable_landmark_count} landmark samples"
    )


    st.caption(
        f"{detection_rate:.1f}% landmark extraction rate"
    )


# ============================================================
# LIVE STATUS FRAGMENT
# ============================================================

@st.fragment(
    run_every=0.25
)
def live_result_panel():

    with shared_state.lock:

        sentence = (
            shared_state.sentence
        )

        prediction = (
            shared_state.prediction
        )

        confidence = (
            shared_state.confidence
        )

        hand_detected = (
            shared_state.hand_detected
        )

        stable_count = (
            shared_state.stable_count
        )


    # ========================================================
    # SENTENCE BOX
    # ========================================================

    if sentence:

        sentence_content = sentence

        sentence_class = "sentence"


    else:

        sentence_content = (
            "Detected signs will appear here..."
        )

        sentence_class = (
            "sentence sentence-empty"
        )


    st.markdown(
        f"""
        <div class="sentence-box">

            <div class="sentence-label">
                DETECTED SENTENCE / SEQUENCE
            </div>

            <div class="{sentence_class}">
                {sentence_content}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # CURRENT STATUS
    # ========================================================

    c1, c2 = st.columns(
        [1, 1]
    )


    with c1:

        if hand_detected:

            status = (
                "🟢 HAND DETECTED"
            )

            status_class = (
                "detected"
            )

        else:

            status = (
                "○ WAITING FOR HAND"
            )

            status_class = (
                "waiting"
            )


        st.markdown(
            f"""
            <div class="live-status">

                <div class="{status_class}">
                    {status}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="live-status">

                Current sign:
                <strong>
                    {prediction}
                </strong>

                &nbsp;&nbsp;

                Confidence:
                <strong>
                    {confidence * 100:.1f}%
                </strong>

                <br>

                Stability:
                <strong>
                    {min(stable_count, STABLE_FRAMES)}
                    /
                    {STABLE_FRAMES}
                </strong>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# PAGE — LIVE TRANSLATOR
# ============================================================

if page == "🎥 Live Translator":

    st.markdown(
        """
        <div class="eyebrow">
            Real-Time Computer Vision
        </div>

        <div class="hero-title">
            SIGNOVA
        </div>

        <div class="hero-subtitle">

            Place your hand inside the camera.

            SIGNOVA detects the hand, maps its landmarks,
            extracts hand vectors and compares the feature
            pattern with the supplied A–Z and 0–9 dataset.

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # PIPELINE
    # ========================================================

    st.markdown(
        """
        <div class="pipeline">

            <div class="pipeline-step">
                📷 Camera
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-step">
                🖐 Hand Detection
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-step">
                🔴 21 Points
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-step">
                📐 Vectors
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-step">
                🧠 KNN
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-step">
                🔤 Character
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-step">
                📝 Sentence
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # LIVE LETTER BOX
    # ========================================================

    live_result_panel()


    # ========================================================
    # SENTENCE BUTTONS
    # ========================================================

    button1, button2, button3, button4 = st.columns(
        [
            1,
            1,
            1,
            3
        ]
    )


    # CLEAR

    with button1:

        if st.button(
            "🗑 Clear",
            use_container_width=True
        ):

            with shared_state.lock:

                shared_state.sentence = ""

                shared_state.last_recorded = None

                shared_state.stable_prediction = None

                shared_state.stable_count = 0

                shared_state.total_recorded = 0


            st.rerun()


    # DELETE

    with button2:

        if st.button(
            "⌫ Delete",
            use_container_width=True
        ):

            with shared_state.lock:

                if shared_state.sentence:

                    shared_state.sentence = (
                        shared_state.sentence[
                            :-1
                        ]
                    )


                shared_state.last_recorded = None


            st.rerun()


    # SPACE

    with button3:

        if st.button(
            "␣ Space",
            use_container_width=True
        ):

            with shared_state.lock:

                if (
                    shared_state.sentence
                    and
                    not shared_state.sentence.endswith(
                        " "
                    )
                ):

                    shared_state.sentence += " "


                shared_state.last_recorded = None


            st.rerun()


    st.markdown("<br>", unsafe_allow_html=True)


    # ========================================================
    # CAMERA + PREDICTION
    # ========================================================

    camera_column, info_column = st.columns(
        [
            1.6,
            1
        ]
    )


    # ========================================================
    # CAMERA COLUMN
    # ========================================================

    with camera_column:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    Live Hand Analysis
                </div>

                <div class="card-text">

                    🟩 Green rectangle = detected hand
                    <br>

                    🔴 Red points = landmarks / key points
                    <br>

                    ⚪ White lines = joint vectors / skeleton
                    <br>

                    🔴 Large center point = hand centroid
                    <br>

                    🔵 Blue line = index fingertip trajectory

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        # Processor factory captures the trained
        # classifier and shared state.

        def processor_factory():

            return SignovaVideoProcessor(
                model,
                shared_state
            )


        webrtc_context = webrtc_streamer(

            key="signova-camera",

            mode=WebRtcMode.SENDRECV,

            video_processor_factory=
                processor_factory,

            media_stream_constraints={
                "video": {
                    "width": {
                        "ideal": 640
                    },

                    "height": {
                        "ideal": 480
                    }
                },

                "audio": False
            },

            async_processing=True
        )


    # ========================================================
    # INFORMATION COLUMN
    # ========================================================

    with info_column:

        with shared_state.lock:

            current_prediction = (
                shared_state.prediction
            )

            current_confidence = (
                shared_state.confidence
            )


        st.markdown(
            f"""
            <div class="prediction">

                <div class="prediction-label">
                    CURRENT PREDICTION
                </div>

                <div class="prediction-letter">
                    {current_prediction}
                </div>

                <div class="confidence">
                    Confidence:
                    <strong>
                        {current_confidence * 100:.1f}%
                    </strong>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="info">

            <strong>How to record a character</strong>

            <br><br>

            1. Put your hand clearly inside the camera.

            <br><br>

            2. Hold the sign steady.

            <br><br>

            3. SIGNOVA waits until the same prediction
            appears for approximately
            <strong>{STABLE_FRAMES} frames</strong>.

            <br><br>

            4. If confidence reaches at least
            <strong>{CONFIDENCE_THRESHOLD * 100:.0f}%</strong>,
            the character is recorded.

            <br><br>

            5. To type the same character twice,
            briefly remove your hand and show it again.

            <br><br>

            Example:

            <br>

            <strong>
                H → E → L → remove hand →
                L → O
            </strong>

            <br><br>

            Result:

            <br>

            <strong>HELLO</strong>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # SYSTEM METRICS
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    m1, m2, m3, m4, m5 = st.columns(
        5
    )


    with m1:

        st.markdown(
            """
            <div class="metric">

                <div class="metric-value">
                    36
                </div>

                <div class="metric-label">
                    Original Classes
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with m2:

        st.markdown(
            """
            <div class="metric">

                <div class="metric-value">
                    21
                </div>

                <div class="metric-label">
                    Hand Landmarks
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with m3:

        st.markdown(
            f"""
            <div class="metric">

                <div class="metric-value">
                    {usable_landmark_count}
                </div>

                <div class="metric-label">
                    Usable Samples
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with m4:

        st.markdown(
            f"""
            <div class="metric">

                <div class="metric-value">
                    {accuracy_text}
                </div>

                <div class="metric-label">
                    Test Accuracy
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with m5:

        st.markdown(
            """
            <div class="metric">

                <div class="metric-value">
                    KNN
                </div>

                <div class="metric-label">
                    Classifier
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# PAGE — DATASET LAB
# ============================================================

elif page == "📊 Dataset Lab":

    st.markdown(
        """
        <div class="eyebrow">
            Dataset Analysis
        </div>

        <div class="hero-title">
            Dataset Lab
        </div>

        <div class="hero-subtitle">

            Inspect the supplied A–Z and 0–9 hand-sign
            dataset and see how many samples MediaPipe
            successfully converts into hand landmarks.

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # DATASET METRICS
    # ========================================================

    c1, c2, c3, c4 = st.columns(
        4
    )


    with c1:

        st.metric(
            "Original Images",
            ORIGINAL_IMAGE_COUNT
        )


    with c2:

        st.metric(
            "Original Classes",
            36
        )


    with c3:

        st.metric(
            "Usable Landmarks",
            usable_landmark_count
        )


    with c4:

        st.metric(
            "Extraction Rate",
            f"{detection_rate:.1f}%"
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # ========================================================
    # DIAGNOSTIC TABLE
    # ========================================================

    st.subheader(
        "Landmark Extraction Diagnostics"
    )


    diagnostic_table = []


    for class_name in EXPECTED_CLASSES:

        detected = detected_counts.get(
            class_name,
            0
        )


        failed = failed_counts.get(
            class_name,
            0
        )


        diagnostic_table.append(
            {
                "Sign": class_name,

                "Original Images":
                    ORIGINAL_IMAGES_PER_CLASS,

                "Landmarks Detected":
                    detected,

                "Failed Detection":
                    failed,

                "Detection Rate":
                    f"{(
                        detected
                        /
                        ORIGINAL_IMAGES_PER_CLASS
                        *
                        100
                    ):.1f}%"
            }
        )


    st.dataframe(
        diagnostic_table,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # WARN ABOUT MISSING CLASSES
    # ========================================================

    if missing_classes:

        st.warning(
            "No usable landmarks were extracted for: "
            +
            ", ".join(
                missing_classes
            )
        )

    else:

        st.success(
            "At least one usable landmark sample "
            "was detected for all 36 classes."
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # ========================================================
    # CLASS IMAGE EXPLORER
    # ========================================================

    st.subheader(
        "Dataset Image Explorer"
    )


    selected_class = st.selectbox(
        "Choose a sign",
        EXPECTED_CLASSES
    )


    selected_folder = os.path.join(
        dataset_path,
        selected_class
    )


    if os.path.isdir(
        selected_folder
    ):

        images = sorted(
            [
                filename

                for filename
                in os.listdir(
                    selected_folder
                )

                if filename.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png"
                    )
                )
            ]
        )


        st.caption(
            f"{len(images)} original images "
            f"for sign {selected_class}"
        )


        gallery_columns = st.columns(
            5
        )


        for index, filename in enumerate(
            images[:20]
        ):

            filepath = os.path.join(
                selected_folder,
                filename
            )


            try:

                image = Image.open(
                    filepath
                )


                with gallery_columns[
                    index % 5
                ]:

                    st.image(
                        image,
                        use_container_width=True
                    )


            except Exception:

                pass


# ============================================================
# PAGE — MODEL INSIGHTS
# ============================================================

elif page == "🧠 Model Insights":

    st.markdown(
        """
        <div class="eyebrow">
            Machine Learning
        </div>

        <div class="hero-title">
            Model Insights
        </div>

        <div class="hero-subtitle">

            SIGNOVA converts the hand into numerical
            landmark and vector features before applying
            K-Nearest Neighbors classification.

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # METRICS
    # ========================================================

    a, b, c, d = st.columns(
        4
    )


    with a:

        st.metric(
            "Classifier",
            "KNN"
        )


    with b:

        st.metric(
            "Evaluation Training",
            train_count
        )


    with c:

        st.metric(
            "Evaluation Testing",
            test_count
        )


    with d:

        st.metric(
            "Accuracy",
            accuracy_text
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info">

        <strong>Important:</strong>

        <br><br>

        The final real-time classifier is trained using
        <strong>all usable landmark samples</strong>.

        <br><br>

        Accuracy evaluation is separate.

        Only classes containing at least two usable samples
        can be placed in both an evaluation training set and
        evaluation testing set.

        <br><br>

        This prevents the previous error caused by using a
        fixed 20% stratified split when too few MediaPipe
        landmark samples were available.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    st.subheader(
        "Feature Representation"
    )


    st.write(
        """
        SIGNOVA's feature vector contains:

        - 21 normalized landmark coordinates
        - X, Y and Z information
        - Wrist-relative coordinates
        - Hand-size normalization
        - Mirror normalization
        - Joint-to-joint vectors
        - Fingertip-to-wrist distances
        - Distances between neighbouring fingertips
        """
    )


    st.subheader(
        "Recognition Pipeline"
    )


    st.markdown(
        """
        <div class="pipeline">

            <div class="pipeline-step">
                Camera
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-step">
                Hand Detection
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-step">
                21 Landmarks
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-step">
                Normalization
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-step">
                Vector Features
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-step">
                KNN
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-step">
                Prediction
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE — COMPUTER VISION
# ============================================================

elif page == "👁 Computer Vision":

    st.markdown(
        """
        <div class="eyebrow">
            Image Processing
        </div>

        <div class="hero-title">
            Computer Vision
        </div>

        <div class="hero-subtitle">

            SIGNOVA visualizes the important computer-vision
            features directly on top of the live webcam.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info">

        <strong>🟩 Green Frame — Hand Detection</strong>

        <br><br>

        A green bounding rectangle is calculated from the
        minimum and maximum hand landmark coordinates.

        <br><br>

        <strong>🔴 Red Points — Landmarks</strong>

        <br><br>

        Twenty-one hand key points represent the wrist,
        finger joints and fingertips.

        <br><br>

        <strong>⚪ White Lines — Edges / Vectors</strong>

        <br><br>

        Connected landmarks form the skeletal structure of
        the hand. Their relative vectors are also used as
        classification features.

        <br><br>

        <strong>🔴 Large Centre Point — Centroid</strong>

        <br><br>

        SIGNOVA calculates the average X and Y position of
        the detected landmarks to visualize the approximate
        hand centre.

        <br><br>

        <strong>🔵 Trajectory — Motion History</strong>

        <br><br>

        SIGNOVA stores recent positions of landmark 8,
        the index fingertip, and connects them to visualize
        its short movement trajectory.

        <br><br>

        The current classifier mainly recognizes static
        hand shape. The trajectory is currently used as a
        computer-vision visualization rather than as a
        classification input.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    st.subheader(
        "MediaPipe 21 Hand Landmarks"
    )


    landmark_names = [

        "0 — Wrist",

        "1 — Thumb CMC",

        "2 — Thumb MCP",

        "3 — Thumb IP",

        "4 — Thumb Tip",

        "5 — Index MCP",

        "6 — Index PIP",

        "7 — Index DIP",

        "8 — Index Tip",

        "9 — Middle MCP",

        "10 — Middle PIP",

        "11 — Middle DIP",

        "12 — Middle Tip",

        "13 — Ring MCP",

        "14 — Ring PIP",

        "15 — Ring DIP",

        "16 — Ring Tip",

        "17 — Pinky MCP",

        "18 — Pinky PIP",

        "19 — Pinky DIP",

        "20 — Pinky Tip"
    ]


    landmark_columns = st.columns(
        3
    )


    for index, name in enumerate(
        landmark_names
    ):

        with landmark_columns[
            index % 3
        ]:

            st.write(
                f"🔴 {name}"
            )


# ============================================================
# PAGE — ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.markdown(
        """
        <div class="eyebrow">
            Image Processing & Computer Vision Project
        </div>

        <div class="hero-title">
            About SIGNOVA
        </div>

        <div class="hero-subtitle">

            A real-time static hand-sign recognition and
            sentence construction system.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info">

        <strong>SIGNOVA</strong> is designed to demonstrate
        how computer vision and machine learning can work
        together to recognize static hand signs.

        <br><br>

        The supplied dataset contains:

        <br><br>

        <strong>
        900 images
        <br>
        36 classes
        <br>
        A–Z
        <br>
        0–9
        </strong>

        <br><br>

        Each dataset image is analysed using MediaPipe hand
        detection. Successfully detected hands are converted
        into normalized landmark and vector features.

        <br><br>

        A K-Nearest Neighbors classifier learns the
        relationship between those features and their
        corresponding sign labels.

        <br><br>

        During real-time use, the webcam follows the same
        feature-extraction pipeline and compares the current
        hand configuration against the trained patterns.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    st.subheader(
        "Technology Stack"
    )


    st.write(
        """
        - Python
        - Streamlit
        - Streamlit-WebRTC
        - MediaPipe
        - PyAV
        - NumPy
        - Pillow
        - Scikit-learn
        - K-Nearest Neighbors
        """
    )
