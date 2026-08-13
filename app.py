import os
import zipfile
import time
import threading
from collections import deque

import av
import numpy as np
import streamlit as st

from PIL import Image, ImageDraw

import mediapipe as mp

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode
)


# ============================================================
# SIGNOVA
# REAL-TIME HAND SIGN RECOGNITION
#
# Computer Vision Pipeline:
#
# Camera
#    ↓
# MediaPipe Hand Detection
#    ↓
# 21 Hand Landmarks
#    ↓
# Red Points + Skeleton Lines
#    ↓
# Normalized Hand Geometry
#    ↓
# Dataset Comparison
#    ↓
# KNN Classifier
#    ↓
# Letter Box
#    ↓
# Sentence Builder
# ============================================================


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SIGNOVA | Hand Sign Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(124, 58, 237, 0.16),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(14, 165, 233, 0.12),
            transparent 25%
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


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0d1220 0%,
            #080b13 100%
        );

    border-right:
        1px solid rgba(255,255,255,0.06);
}


/* =========================================================
   BRAND
   ========================================================= */

.brand-icon {
    width: 52px;
    height: 52px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 16px;

    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #2563eb
        );

    font-size: 27px;

    box-shadow:
        0 10px 30px rgba(124,58,237,0.25);
}

.brand-name {
    margin-top: 13px;

    font-size: 25px;

    font-weight: 950;

    letter-spacing: 2px;
}

.brand-description {
    color: #64748b;

    font-size: 11px;

    margin-top: 3px;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    padding:
        12px 0 25px 0;
}

.eyebrow {
    color: #818cf8;

    font-size: 11px;

    font-weight: 900;

    letter-spacing: 3px;
}

.hero-title {
    font-size: 52px;

    font-weight: 950;

    letter-spacing: -3px;

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

.hero-description {
    color: #94a3b8;

    font-size: 15px;

    line-height: 1.7;

    max-width: 760px;
}


/* =========================================================
   CARDS
   ========================================================= */

.card {
    background:
        rgba(15,23,42,0.72);

    border:
        1px solid rgba(148,163,184,0.10);

    border-radius: 22px;

    padding: 22px;

    box-shadow:
        0 20px 55px rgba(0,0,0,0.20);
}

.card-title {
    color: #f8fafc;

    font-size: 16px;

    font-weight: 850;

    margin-bottom: 7px;
}

.card-description {
    color: #64748b;

    font-size: 12px;

    line-height: 1.6;
}


/* =========================================================
   SENTENCE BOX
   ========================================================= */

.sentence-box {
    background:
        linear-gradient(
            145deg,
            rgba(124,58,237,0.14),
            rgba(37,99,235,0.07)
        );

    border:
        1px solid rgba(139,92,246,0.30);

    border-radius: 22px;

    padding: 22px;

    margin-bottom: 20px;

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

.sentence-content {
    min-height: 62px;

    display: flex;

    align-items: center;

    margin-top: 8px;

    color: #f8fafc;

    font-size: 30px;

    font-weight: 850;

    letter-spacing: 4px;

    word-break: break-word;
}

.sentence-empty {
    color: #475569;

    font-size: 16px;

    letter-spacing: 0;

    font-weight: 500;
}


/* =========================================================
   LETTER BOX
   ========================================================= */

.letter-box {
    background:
        radial-gradient(
            circle at 50% 30%,
            rgba(124,58,237,0.16),
            transparent 55%
        ),
        #0b1020;

    border:
        1px solid rgba(139,92,246,0.25);

    border-radius: 24px;

    padding: 25px;

    text-align: center;

    min-height: 245px;

    display: flex;

    flex-direction: column;

    justify-content: center;

    box-shadow:
        inset 0 0 50px rgba(124,58,237,0.04);
}

.letter-label {
    color: #64748b;

    font-size: 10px;

    font-weight: 900;

    letter-spacing: 3px;

    text-transform: uppercase;
}

.letter {
    font-size: 105px;

    line-height: 1;

    font-weight: 950;

    margin: 14px 0;

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
    color: #94a3b8;

    font-size: 12px;
}


/* =========================================================
   STATUS
   ========================================================= */

.status-live {
    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 8px 13px;

    border-radius: 999px;

    background:
        rgba(34,197,94,0.09);

    border:
        1px solid rgba(34,197,94,0.20);

    color: #86efac;

    font-size: 11px;

    font-weight: 800;
}

.status-ready {
    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 8px 13px;

    border-radius: 999px;

    background:
        rgba(99,102,241,0.09);

    border:
        1px solid rgba(99,102,241,0.20);

    color: #a5b4fc;

    font-size: 11px;

    font-weight: 800;
}


/* =========================================================
   METRICS
   ========================================================= */

.metric-card {
    background:
        rgba(15,23,42,0.72);

    border:
        1px solid rgba(255,255,255,0.07);

    border-radius: 18px;

    padding: 17px;
}

.metric-number {
    font-size: 27px;

    font-weight: 950;

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

    font-size: 9px;

    font-weight: 850;

    letter-spacing: 1.5px;

    text-transform: uppercase;

    margin-top: 2px;
}


/* =========================================================
   INSTRUCTION
   ========================================================= */

.instruction {
    background:
        rgba(99,102,241,0.06);

    border-left:
        3px solid #6366f1;

    border-radius: 9px;

    padding: 15px 17px;

    color: #cbd5e1;

    font-size: 12px;

    line-height: 1.8;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;

    color: #475569;

    font-size: 10px;

    padding: 40px 0 15px 0;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

ZIP_FILE = "archive.zip"

DATASET_FOLDER = "signova_dataset"

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

RANDOM_STATE = 42

MIN_CONFIDENCE = 0.55

STABLE_FRAMES = 8

RECORD_COOLDOWN = 1.2

MAX_TRAJECTORY = 20


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands

mp_connections = mp_hands.HAND_CONNECTIONS


# ============================================================
# GLOBAL HAND DETECTOR FOR DATASET
# ============================================================

dataset_hands = mp_hands.Hands(
    static_image_mode=True,

    max_num_hands=1,

    min_detection_confidence=0.45,

    min_tracking_confidence=0.45
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def landmarks_to_features(landmarks):

    """
    Convert 21 MediaPipe landmarks into normalized
    geometric features.

    Each landmark contains:
        x
        y
        z

    We translate the hand relative to the wrist,
    then normalize its scale.

    This makes the comparison less dependent on:

        - hand position
        - hand size
        - camera distance
    """

    points = np.array(
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


    # --------------------------------------------------------
    # Wrist = landmark 0
    # --------------------------------------------------------

    wrist = points[0].copy()

    points = points - wrist


    # --------------------------------------------------------
    # Scale normalization
    # --------------------------------------------------------

    distances = np.linalg.norm(
        points,
        axis=1
    )

    scale = np.max(
        distances
    )


    if scale < 1e-6:

        scale = 1.0


    points = points / scale


    # --------------------------------------------------------
    # Create additional vector features
    # --------------------------------------------------------

    vectors = []

    for connection in mp_connections:

        start = connection[0]

        end = connection[1]

        vector = (
            points[end] -
            points[start]
        )

        vectors.extend(
            vector.tolist()
        )


    # --------------------------------------------------------
    # Flatten landmark coordinates
    # --------------------------------------------------------

    landmark_features = (
        points.flatten()
    )


    vector_features = np.array(
        vectors,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Final feature vector
    # --------------------------------------------------------

    features = np.concatenate(
        [
            landmark_features,
            vector_features
        ]
    )


    return features.astype(
        np.float32
    )


# ============================================================
# DETECT HAND IN PIL IMAGE
# ============================================================

def detect_hand_image(image):

    """
    Detect one hand in a PIL image.
    """

    rgb = np.asarray(
        image.convert("RGB")
    )


    results = dataset_hands.process(
        rgb
    )


    if not results.multi_hand_landmarks:

        return None


    hand = results.multi_hand_landmarks[0]


    return hand.landmark


# ============================================================
# DRAW RED LANDMARKS
# ============================================================

def draw_hand_landmarks(
    image,
    landmarks,
    trajectory=None
):

    """
    Draw:

        🔴 red landmark points
        🔴 red skeleton lines
        ➡ trajectory points
    """

    image = image.convert(
        "RGB"
    )

    draw = ImageDraw.Draw(
        image
    )


    width, height = image.size


    # --------------------------------------------------------
    # Convert landmarks to pixel positions
    # --------------------------------------------------------

    points = []

    for landmark in landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        points.append(
            (x, y)
        )


    # --------------------------------------------------------
    # Draw skeleton lines
    # --------------------------------------------------------

    for connection in mp_connections:

        start = connection[0]

        end = connection[1]

        x1, y1 = points[start]

        x2, y2 = points[end]


        draw.line(
            [
                (x1, y1),
                (x2, y2)
            ],

            fill=(255, 40, 40),

            width=3
        )


    # --------------------------------------------------------
    # Draw red points
    # --------------------------------------------------------

    for index, (x, y) in enumerate(
        points
    ):

        radius = 5


        draw.ellipse(
            [
                x - radius,
                y - radius,
                x + radius,
                y + radius
            ],

            fill=(255, 0, 0),

            outline=(255, 255, 255),

            width=1
        )


    # --------------------------------------------------------
    # Draw trajectory
    # --------------------------------------------------------

    if trajectory:

        for i in range(
            1,
            len(trajectory)
        ):

            p1 = trajectory[i - 1]

            p2 = trajectory[i]


            draw.line(
                [
                    p1,
                    p2
                ],

                fill=(255, 80, 80),

                width=2
            )


    return image


# ============================================================
# EXTRACT DATASET ZIP
# ============================================================

def locate_dataset_folder():

    if not os.path.exists(
        DATASET_FOLDER
    ):

        return None


    # Direct DATASET folder

    direct = os.path.join(
        DATASET_FOLDER,
        "DATASET"
    )


    if os.path.isdir(
        direct
    ):

        return direct


    # Search recursively

    for root, dirs, files in os.walk(
        DATASET_FOLDER
    ):

        for directory in dirs:

            if directory.upper() == "DATASET":

                return os.path.join(
                    root,
                    directory
                )


    return None


def extract_dataset():

    existing = locate_dataset_folder()


    if existing:

        return existing


    if not os.path.exists(
        ZIP_FILE
    ):

        st.error(
            "❌ archive.zip was not found."
        )

        st.info(
            "Upload archive.zip to the same GitHub repository as app.py."
        )

        st.stop()


    os.makedirs(
        DATASET_FOLDER,
        exist_ok=True
    )


    try:

        with zipfile.ZipFile(
            ZIP_FILE,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                DATASET_FOLDER
            )

    except zipfile.BadZipFile:

        st.error(
            "❌ archive.zip is not a valid ZIP file."
        )

        st.stop()


    dataset = locate_dataset_folder()


    if dataset is None:

        st.error(
            "❌ Could not find the DATASET folder inside archive.zip."
        )

        st.info(
            "Your ZIP should contain folders such as A, B, C, D..."
        )

        st.stop()


    return dataset


# ============================================================
# DATASET LOADING
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_landmark_dataset(
    dataset_path
):

    features = []

    labels = []

    image_counts = {}


    class_folders = []


    # --------------------------------------------------------
    # Find class folders
    # --------------------------------------------------------

    for item in sorted(
        os.listdir(
            dataset_path
        )
    ):

        folder_path = os.path.join(
            dataset_path,
            item
        )


        if not os.path.isdir(
            folder_path
        ):

            continue


        class_folders.append(
            item
        )


    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for class_name in class_folders:

        folder_path = os.path.join(
            dataset_path,
            class_name
        )


        count = 0


        for filename in os.listdir(
            folder_path
        ):

            if not filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                continue


            path = os.path.join(
                folder_path,
                filename
            )


            try:

                image = Image.open(
                    path
                ).convert("RGB")


                landmarks = detect_hand_image(
                    image
                )


                if landmarks is None:

                    continue


                vector = landmarks_to_features(
                    landmarks
                )


                features.append(
                    vector
                )

                labels.append(
                    class_name
                )

                count += 1


            except Exception:

                continue


        image_counts[
            class_name
        ] = count


    if not features:

        return (
            np.empty(
                (0, 1)
            ),

            np.array(
                []
            ),

            image_counts
        )


    return (
        np.asarray(
            features,
            dtype=np.float32
        ),

        np.asarray(
            labels
        ),

        image_counts
    )


# ============================================================
# TRAIN KNN
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def train_knn(
    X,
    y
):

    if len(X) < 2:

        return (
            None,
            0.0
        )


    unique_classes = np.unique(
        y
    )


    if len(unique_classes) < 2:

        return (
            None,
            0.0
        )


    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y
        )

    except Exception:

        # Dataset too small for stratification

        X_train = X
        X_test = X
        y_train = y
        y_test = y


    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    n_neighbors = min(
        5,
        len(X_train)
    )


    model = KNeighborsClassifier(
        n_neighbors=max(
            1,
            n_neighbors
        ),

        weights="distance",

        metric="euclidean"
    )


    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    try:

        predictions = model.predict(
            X_test
        )


        accuracy = accuracy_score(
            y_test,
            predictions
        )

    except Exception:

        accuracy = 0.0


    return (
        model,
        float(accuracy)
    )


# ============================================================
# LOAD DATASET
# ============================================================

with st.spinner(
    "SIGNOVA is loading the hand-sign dataset..."
):

    dataset_path = extract_dataset()


with st.spinner(
    "SIGNOVA is extracting hand landmarks from the dataset..."
):

    X, y, image_counts = load_landmark_dataset(
        dataset_path
    )


if len(X) == 0:

    st.error(
        "SIGNOVA could not detect hands in the dataset images."
    )

    st.info(
        "Make sure the dataset contains clear hand images inside class folders."
    )

    st.stop()


with st.spinner(
    "Training the SIGNOVA hand geometry classifier..."
):

    model, model_accuracy = train_knn(
        X,
        y
    )


if model is None:

    st.error(
        "The dataset does not contain enough classes to train SIGNOVA."
    )

    st.stop()


classes = sorted(
    np.unique(y)
)


# ============================================================
# REAL-TIME STATE
# ============================================================

class SignState:

    def __init__(self):

        self.lock = threading.Lock()

        self.prediction = "-"

        self.confidence = 0.0

        self.sentence = ""

        self.stable_prediction = None

        self.stable_count = 0

        self.last_recorded = None

        self.last_record_time = 0

        self.hand_detected = False

        self.landmark_count = 0

        self.trajectory = deque(
            maxlen=MAX_TRAJECTORY
        )

        self.camera_running = False


sign_state = SignState()


# ============================================================
# REAL-TIME PREDICTION
# ============================================================

def predict_landmarks(
    landmarks
):

    features = landmarks_to_features(
        landmarks
    )


    features = features.reshape(
        1,
        -1
    )


    prediction = model.predict(
        features
    )[0]


    # --------------------------------------------------------
    # KNN confidence
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        features
    )[0]


    confidence = float(
        np.max(
            probabilities
        )
    )


    return (
        str(prediction),
        confidence
    )


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class SignovaVideoProcessor:

    def __init__(self):

        self.hands = mp_hands.Hands(

            static_image_mode=False,

            max_num_hands=1,

            min_detection_confidence=0.50,

            min_tracking_confidence=0.50
        )


        self.trajectory = deque(
            maxlen=MAX_TRAJECTORY
        )


    def recv(
        self,
        frame
    ):

        # ----------------------------------------------------
        # Convert WebRTC frame to RGB
        # ----------------------------------------------------

        image = frame.to_image().convert(
            "RGB"
        )


        rgb = np.asarray(
            image
        )


        # ----------------------------------------------------
        # MediaPipe hand detection
        # ----------------------------------------------------

        results = self.hands.process(
            rgb
        )


        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            landmarks = hand.landmark


            # ------------------------------------------------
            # Hand detected
            # ------------------------------------------------

            with sign_state.lock:

                sign_state.hand_detected = True

                sign_state.landmark_count = 21


            # ------------------------------------------------
            # Calculate palm center
            # ------------------------------------------------

            width, height = image.size


            center_x = int(
                landmarks[9].x *
                width
            )


            center_y = int(
                landmarks[9].y *
                height
            )


            self.trajectory.append(
                (
                    center_x,
                    center_y
                )
            )


            # ------------------------------------------------
            # Predict sign
            # ------------------------------------------------

            try:

                prediction, confidence = predict_landmarks(
                    landmarks
                )

            except Exception:

                prediction = "-"

                confidence = 0.0


            current_time = time.time()


            # ------------------------------------------------
            # Update shared prediction
            # ------------------------------------------------

            with sign_state.lock:

                sign_state.prediction = prediction

                sign_state.confidence = confidence


                # --------------------------------------------
                # Stable sign detection
                # --------------------------------------------

                if confidence >= MIN_CONFIDENCE:

                    if (
                        sign_state.stable_prediction
                        == prediction
                    ):

                        sign_state.stable_count += 1

                    else:

                        sign_state.stable_prediction = (
                            prediction
                        )

                        sign_state.stable_count = 1


                    # ----------------------------------------
                    # Record letter
                    # ----------------------------------------

                    if (
                        sign_state.stable_count
                        >= STABLE_FRAMES

                        and

                        sign_state.last_recorded
                        != prediction

                        and

                        (
                            current_time
                            -
                            sign_state.last_record_time
                        )
                        >= RECORD_COOLDOWN
                    ):

                        # ------------------------------------
                        # Handle special dataset labels
                        # ------------------------------------

                        normalized = prediction.lower()


                        if normalized in (
                            "space",
                            "blank",
                            "_",
                            " "
                        ):

                            sign_state.sentence += " "

                        elif normalized in (
                            "delete",
                            "del"
                        ):

                            if sign_state.sentence:

                                sign_state.sentence = (
                                    sign_state.sentence[:-1]
                                )

                        elif normalized not in (
                            "nothing",
                            "none",
                            "background"
                        ):

                            sign_state.sentence += (
                                prediction
                            )


                        sign_state.last_recorded = (
                            prediction
                        )

                        sign_state.last_record_time = (
                            current_time
                        )


                else:

                    sign_state.stable_count = 0


                    # ----------------------------------------
                    # Allow next sign
                    # ----------------------------------------

                    if confidence < 0.40:

                        sign_state.last_recorded = None


            # ------------------------------------------------
            # Draw red points and lines
            # ------------------------------------------------

            image = draw_hand_landmarks(
                image,
                landmarks,
                list(self.trajectory)
            )


        else:

            # ------------------------------------------------
            # No hand detected
            # ------------------------------------------------

            self.trajectory.clear()


            with sign_state.lock:

                sign_state.hand_detected = False

                sign_state.landmark_count = 0

                sign_state.stable_count = 0

                sign_state.last_recorded = None


        # ----------------------------------------------------
        # Return processed video
        # ----------------------------------------------------

        return av.VideoFrame.from_ndarray(
            np.asarray(image),
            format="rgb24"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div>

            <div class="brand-icon">
                🤟
            </div>

            <div class="brand-name">
                SIGNOVA
            </div>

            <div class="brand-description">
                Real-Time Hand Sign Recognition
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    st.markdown("---")


    page = st.radio(
        "NAVIGATION",
        [
            "Live Translator",
            "Dataset Explorer",
            "Model",
            "Computer Vision",
            "About"
        ]
    )


    st.markdown("---")


    st.markdown(
        "**SYSTEM STATUS**"
    )


    st.success(
        "Recognition Engine Ready"
    )


    st.caption(
        f"{len(classes)} sign classes"
    )


    st.caption(
        f"{len(X)} landmark samples"
    )


    st.markdown("---")


    st.caption(
        "Image Processing & Computer Vision"
    )


# ============================================================
# LIVE TRANSLATOR
# ============================================================

if page == "Live Translator":

    st.markdown(
        """
        <div class="hero">

            <div class="eyebrow">
                COMPUTER VISION • REAL TIME
            </div>

            <div class="hero-title">
                SIGNOVA
            </div>

            <div class="hero-description">
                Transform hand signs captured by your webcam
                into letters using landmark geometry,
                vectors and dataset comparison.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # GET STATE
    # ========================================================

    with sign_state.lock:

        current_prediction = (
            sign_state.prediction
        )

        current_confidence = (
            sign_state.confidence
        )

        current_sentence = (
            sign_state.sentence
        )

        hand_detected = (
            sign_state.hand_detected
        )

        landmark_count = (
            sign_state.landmark_count
        )

        stable_count = (
            sign_state.stable_count
        )


    # ========================================================
    # SENTENCE BOX
    # ========================================================

    if current_sentence:

        sentence_html = (
            f"""
            <div class="sentence-content">
                {current_sentence}
            </div>
            """
        )

    else:

        sentence_html = (
            """
            <div class="sentence-content sentence-empty">
                Your detected letters will appear here...
            </div>
            """
        )


    st.markdown(
        f"""
        <div class="sentence-box">

            <div class="sentence-label">
                LIVE SENTENCE
            </div>

            {sentence_html}

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # CONTROL BUTTONS
    # ========================================================

    b1, b2, b3, b4 = st.columns(
        [1, 1, 1, 4]
    )


    with b1:

        if st.button(
            "🗑 Clear",
            use_container_width=True
        ):

            with sign_state.lock:

                sign_state.sentence = ""

                sign_state.last_recorded = None

                sign_state.stable_prediction = None

                sign_state.stable_count = 0


            st.rerun()


    with b2:

        if st.button(
            "⌫ Delete",
            use_container_width=True
        ):

            with sign_state.lock:

                if sign_state.sentence:

                    sign_state.sentence = (
                        sign_state.sentence[:-1]
                    )

                    sign_state.last_recorded = None


            st.rerun()


    # ========================================================
    # CAMERA / LETTER
    # ========================================================

    camera_col, letter_col = st.columns(
        [1.55, 1]
    )


    with camera_col:

        if hand_detected:

            status_html = (
                """
                <div class="status-live">
                    ● HAND DETECTED
                </div>
                """
            )

        else:

            status_html = (
                """
                <div class="status-ready">
                    ● SHOW YOUR HAND
                </div>
                """
            )


        st.markdown(
            f"""
            <div class="card">

                {status_html}

                <br>

                <div class="card-title">
                    Live Vision Camera
                </div>

                <div class="card-description">
                    Red points represent hand landmarks.
                    Red lines represent the hand skeleton.
                    The trajectory shows recent palm movement.
                </div>

                <br>

            </div>
            """,
            unsafe_allow_html=True
        )


        webrtc_streamer(
            key="signova-live-camera",

            mode=WebRtcMode.SENDRECV,

            video_processor_factory=(
                SignovaVideoProcessor
            ),

            media_stream_constraints={
                "video": True,
                "audio": False
            },

            async_processing=True
        )


    with letter_col:

        # ----------------------------------------------------
        # LETTER BOX
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="letter-box">

                <div class="letter-label">
                    DETECTED LETTER
                </div>

                <div class="letter">
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


        # ----------------------------------------------------
        # LANDMARK STATUS
        # ----------------------------------------------------

        l1, l2 = st.columns(2)


        with l1:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-number">
                        {landmark_count}
                    </div>

                    <div class="metric-label">
                        Hand Points
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with l2:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-number">
                        {stable_count}
                    </div>

                    <div class="metric-label">
                        Stable Frames
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
            """
            <div class="instruction">

            <strong>How to use SIGNOVA</strong>

            <br><br>

            <b>1.</b> Start the camera and allow webcam access.<br>

            <b>2.</b> Put your hand inside the camera frame.<br>

            <b>3.</b> Red points should appear on your hand.<br>

            <b>4.</b> Red lines show the detected hand structure.<br>

            <b>5.</b> Hold your sign steadily.<br>

            <b>6.</b> SIGNOVA compares the landmark geometry
            with the dataset.<br>

            <b>7.</b> The detected letter appears in the
            Letter Box.<br>

            <b>8.</b> The letter is automatically added to
            the sentence once stable.

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # METRICS
    # ========================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    m1, m2, m3, m4 = st.columns(4)


    with m1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-number">
                    {len(classes)}
                </div>

                <div class="metric-label">
                    Sign Classes
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with m2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-number">
                    {len(X)}
                </div>

                <div class="metric-label">
                    Hand Samples
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with m3:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-number">
                    {model_accuracy * 100:.1f}%
                </div>

                <div class="metric-label">
                    Test Accuracy
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with m4:

        st.markdown(
            """
            <div class="metric-card">

                <div class="metric-number">
                    KNN
                </div>

                <div class="metric-label">
                    Comparator
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":

    st.markdown(
        """
        <div class="hero">

            <div class="eyebrow">
                DATASET ANALYSIS
            </div>

            <div class="hero-title">
                Dataset Explorer
            </div>

            <div class="hero-description">
                Explore the hand-sign samples used by
                SIGNOVA to build its landmark-based
                recognition model.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    d1, d2, d3 = st.columns(3)


    with d1:

        st.metric(
            "Classes",
            len(classes)
        )


    with d2:

        st.metric(
            "Usable Images",
            len(X)
        )


    with d3:

        st.metric(
            "Landmarks / Hand",
            21
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    selected_class = st.selectbox(
        "Select a sign class",
        classes
    )


    st.markdown(
        f"""
        <div class="card">

            <div class="card-title">
                Sign Class: {selected_class}
            </div>

            <div class="card-description">
                {image_counts.get(selected_class, 0)}
                images successfully produced detectable
                hand landmarks.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL PAGE
# ============================================================

elif page == "Model":

    st.markdown(
        """
        <div class="hero">

            <div class="eyebrow">
                MACHINE LEARNING
            </div>

            <div class="hero-title">
                Recognition Model
            </div>

            <div class="hero-description">
                SIGNOVA compares normalized hand geometry
                instead of relying on raw image pixels.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Classifier",
            "KNN"
        )


    with c2:

        st.metric(
            "Classes",
            len(classes)
        )


    with c3:

        st.metric(
            "Accuracy",
            f"{model_accuracy * 100:.2f}%"
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                Why Landmark-Based Recognition?
            </div>

            <br>

            Raw images contain many unnecessary differences,
            such as background, lighting and camera position.

            <br><br>

            SIGNOVA instead converts the hand into a set of
            geometric landmarks.

            <br><br>

            Each hand contains:

            <br><br>

            🔴 <strong>21 landmark points</strong><br>

            🔴 <strong>Hand skeleton connections</strong><br>

            ➡️ <strong>Relative vectors</strong><br>

            📐 <strong>Normalized distances and geometry</strong>

            <br><br>

            These features are then compared with the
            landmark features extracted from the dataset.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# COMPUTER VISION PAGE
# ============================================================

elif page == "Computer Vision":

    st.markdown(
        """
        <div class="hero">

            <div class="eyebrow">
                IMAGE PROCESSING PIPELINE
            </div>

            <div class="hero-title">
                Computer Vision
            </div>

            <div class="hero-description">
                The complete processing pipeline used by
                SIGNOVA to transform a camera image into
                a recognized letter.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    pipeline = [

        (
            "01",
            "Camera Frame",
            "The webcam continuously provides RGB video frames."
        ),

        (
            "02",
            "Hand Detection",
            "MediaPipe Hands identifies the user's hand and detects its 21 landmarks."
        ),

        (
            "03",
            "Centroid / Points",
            "The landmark coordinates represent important points such as fingertips, joints and wrist."
        ),

        (
            "04",
            "Edges / Skeleton",
            "Connections between landmarks form a geometric hand skeleton."
        ),

        (
            "05",
            "Vectors",
            "Relative vectors between connected landmarks describe finger directions and hand shape."
        ),

        (
            "06",
            "Normalization",
            "Coordinates are translated relative to the wrist and normalized by hand size."
        ),

        (
            "07",
            "Dataset Comparison",
            "The normalized live-hand feature vector is compared against landmark vectors extracted from the dataset."
        ),

        (
            "08",
            "KNN Classification",
            "The nearest landmark samples determine the most likely hand-sign class."
        ),

        (
            "09",
            "Letter Output",
            "The predicted class is displayed in the SIGNOVA Letter Box."
        ),

        (
            "10",
            "Sentence Builder",
            "Stable predictions are added one time to the live sentence."
        )
    ]


    for number, title, description in pipeline:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    {number} — {title}
                </div>

                <div class="card-description">
                    {description}
                </div>

            </div>

            <br>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "About":

    st.markdown(
        """
        <div class="hero">

            <div class="eyebrow">
                PROJECT INFORMATION
            </div>

            <div class="hero-title">
                About SIGNOVA
            </div>

            <div class="hero-description">
                A real-time hand-sign recognition system
                created for an Image Processing and
                Computer Vision project.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                Project Objective
            </div>

            <br>

            SIGNOVA demonstrates how computer vision and
            machine learning can be combined to recognize
            hand signs from a live camera.

            <br><br>

            Instead of directly comparing complete images,
            SIGNOVA extracts the geometric structure of the
            hand.

            <br><br>

            The system identifies:

            <br><br>

            🔴 Hand landmark points<br>
            🔴 Finger joint positions<br>
            🔴 Hand skeleton edges<br>
            ➡️ Relative vectors<br>
            📐 Normalized hand geometry

            <br><br>

            The extracted features are compared with the
            supplied hand-sign dataset.

            <br><br>

            The result is then displayed as a letter and
            stored in the sentence builder.

            <br><br>

            <strong>Technology Stack</strong>

            <br><br>

            Python • Streamlit • MediaPipe • NumPy •
            Pillow • Scikit-learn • WebRTC

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="footer">

            SIGNOVA<br>
            Real-Time Hand Sign Recognition<br>
            Image Processing & Computer Vision

        </div>
        """,
        unsafe_allow_html=True
    )
