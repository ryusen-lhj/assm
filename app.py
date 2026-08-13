import os
import zipfile
import time
import threading

import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration
)


# ============================================================
# SIGNOVA
# REAL-TIME HAND SIGN SENTENCE BUILDER
# (Hand-Landmark Edition — red dot / red skeleton overlay)
# ============================================================

st.set_page_config(
    page_title="SIGNOVA | Real-Time Sign Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(124, 58, 237, 0.16),
            transparent 28%
        ),
        radial-gradient(
            circle at 85% 20%,
            rgba(14, 165, 233, 0.12),
            transparent 25%
        ),
        #080b14;

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


/* SIDEBAR */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0d1220 0%,
            #090d17 100%
        );

    border-right: 1px solid rgba(255,255,255,0.06);
}


/* BRAND */

.brand {
    padding: 10px 0 25px 0;
}

.brand-icon {
    width: 50px;
    height: 50px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 15px;

    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #2563eb
        );

    font-size: 26px;
}

.brand-name {
    font-size: 25px;
    font-weight: 900;
    letter-spacing: 1px;
    margin-top: 12px;
}

.brand-text {
    color: #64748b;
    font-size: 12px;
}


/* HERO */

.hero {
    padding: 20px 0 25px 0;
}

.eyebrow {
    color: #818cf8;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero-title {
    font-size: 50px;
    font-weight: 950;
    letter-spacing: -2px;

    background:
        linear-gradient(
            90deg,
            #f8fafc,
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
    max-width: 760px;
}


/* SENTENCE BOX */

.sentence-container {
    background:
        linear-gradient(
            145deg,
            rgba(124,58,237,0.15),
            rgba(37,99,235,0.08)
        );

    border:
        1px solid rgba(139,92,246,0.30);

    border-radius: 22px;

    padding: 22px;

    margin: 15px 0 25px 0;

    box-shadow:
        0 20px 50px rgba(0,0,0,0.25);
}

.sentence-label {
    color: #818cf8;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.sentence-text {
    min-height: 65px;

    display: flex;
    align-items: center;

    font-size: 32px;
    font-weight: 750;

    letter-spacing: 4px;

    color: #f8fafc;

    padding-top: 10px;

    word-break: break-word;
}

.empty-sentence {
    color: #475569;
    font-size: 17px;
    letter-spacing: 0;
    font-weight: 500;
}


/* CAMERA CARD */

.camera-card {
    background:
        rgba(15,23,42,0.75);

    border:
        1px solid rgba(148,163,184,0.10);

    border-radius: 22px;

    padding: 20px;

    box-shadow:
        0 20px 50px rgba(0,0,0,0.20);
}


/* STATUS */

.status {
    display: inline-flex;
    align-items: center;

    gap: 8px;

    padding: 8px 13px;

    border-radius: 999px;

    background:
        rgba(34,197,94,0.10);

    border:
        1px solid rgba(34,197,94,0.20);

    color: #86efac;

    font-size: 12px;
    font-weight: 700;
}

.status-warning {
    background: rgba(239,68,68,0.10);
    border: 1px solid rgba(239,68,68,0.25);
    color: #fca5a5;
}


/* PREDICTION */

.prediction-box {
    background:
        rgba(15,23,42,0.75);

    border:
        1px solid rgba(148,163,184,0.10);

    border-radius: 22px;

    padding: 25px;

    text-align: center;
}

.prediction-title {
    color: #64748b;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 2px;

    text-transform: uppercase;
}

.prediction-letter {
    font-size: 90px;

    line-height: 1;

    font-weight: 950;

    margin: 15px 0;

    background:
        linear-gradient(
            135deg,
            #c4b5fd,
            #60a5fa
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


/* INSTRUCTIONS */

.instruction {
    background:
        rgba(99,102,241,0.07);

    border-left:
        3px solid #6366f1;

    border-radius: 9px;

    padding: 14px 17px;

    color: #cbd5e1;

    font-size: 13px;

    line-height: 1.7;
}


/* METRIC */

.metric {
    background:
        rgba(15,23,42,0.70);

    border:
        1px solid rgba(255,255,255,0.07);

    border-radius: 18px;

    padding: 18px;
}

.metric-number {
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

    text-transform: uppercase;
}


/* FOOTER */

.footer {
    text-align: center;

    color: #475569;

    font-size: 11px;

    padding: 40px 0 10px 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

ZIP_FILE = "archive.zip"

EXTRACT_FOLDER = "signova_dataset"

RANDOM_STATE = 42

# Cap per class so MediaPipe doesn't have to crawl through
# tens of thousands of images while building the training set.
MAX_IMAGES_PER_CLASS = 300

# Minimum detection confidence required for a dataset image
# to be accepted as a valid landmark sample.
DATASET_MIN_DETECTION_CONFIDENCE = 0.5

# 21 landmarks * (x, y, z) = 63 point features per hand.
LANDMARKS_PER_HAND = 21


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Colors are in RGB order because every frame we touch in this
# app is kept as an RGB ndarray end-to-end.
RED = (255, 0, 0)

LANDMARK_DRAW_SPEC = mp_drawing.DrawingSpec(
    color=RED,
    thickness=-1,
    circle_radius=4
)

CONNECTION_DRAW_SPEC = mp_drawing.DrawingSpec(
    color=RED,
    thickness=2
)

# ------------------------------------------------------------
# EDGE / VECTOR GEOMETRY
#
# mp_hands.HAND_CONNECTIONS is the exact same set of landmark
# pairs used to draw the red skeleton lines on screen (via
# mp_drawing.draw_landmarks below). We reuse that identical
# list of pairs to build "edge vector" features, so what the
# SVM learns from is literally the same red dots + red lines
# that are drawn on screen — not just the raw dots.
#
# Sorted once at import time so every call (dataset build and
# live camera) always produces edges in the same fixed order.
# ------------------------------------------------------------

HAND_EDGES = sorted(mp_hands.HAND_CONNECTIONS)
NUM_EDGES = len(HAND_EDGES)

POINT_FEATURES = LANDMARKS_PER_HAND * 3      # red-dot (point) features
EDGE_FEATURES = NUM_EDGES * 4                # red-line (edge/vector) features: dx, dy, dz, length
TOTAL_FEATURES = POINT_FEATURES + EDGE_FEATURES


# ============================================================
# SESSION STATE
# ============================================================

if "sentence" not in st.session_state:
    st.session_state.sentence = ""

if "last_recorded_sign" not in st.session_state:
    st.session_state.last_recorded_sign = None

if "last_record_time" not in st.session_state:
    st.session_state.last_record_time = 0

if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = "-"

if "current_confidence" not in st.session_state:
    st.session_state.current_confidence = 0.0


# ============================================================
# DATASET EXTRACTION
# ============================================================

def find_dataset_folder():

    if not os.path.exists(
        EXTRACT_FOLDER
    ):
        return None

    direct = os.path.join(
        EXTRACT_FOLDER,
        "DATASET"
    )

    if os.path.isdir(direct):
        return direct

    for root, dirs, files in os.walk(
        EXTRACT_FOLDER
    ):

        if os.path.basename(
            root
        ).upper() == "DATASET":

            return root

    return None


def extract_dataset():

    existing = find_dataset_folder()

    if existing:
        return existing

    if not os.path.exists(
        ZIP_FILE
    ):

        st.error(
            "archive.zip was not found. "
            "Please place archive.zip beside app.py."
        )

        st.stop()

    os.makedirs(
        EXTRACT_FOLDER,
        exist_ok=True
    )

    try:

        with zipfile.ZipFile(
            ZIP_FILE,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                EXTRACT_FOLDER
            )

    except zipfile.BadZipFile:

        st.error(
            "archive.zip is corrupted."
        )

        st.stop()

    dataset = find_dataset_folder()

    if dataset is None:

        st.error(
            "DATASET folder could not be found inside archive.zip."
        )

        st.stop()

    return dataset


# ============================================================
# HAND LANDMARK + EDGE/VECTOR FEATURE EXTRACTION
#
# Instead of comparing raw pixels, SIGNOVA detects the 21
# MediaPipe hand keypoints (the red dots) and turns them into
# a scale- and position-invariant feature vector made of TWO
# parts:
#
#   PART A — POINTS (red dots)
#   1. Every point is shifted so the wrist (landmark 0)
#      becomes the origin (translation invariance).
#   2. Every point is divided by the largest wrist distance
#      in that hand (scale invariance — a hand held close to
#      the camera produces the same vector as one held far).
#   -> 21 points * (x, y, z) = 63 numbers.
#
#   PART B — EDGES / VECTORS (red lines)
#   For every pair of landmarks connected by a red skeleton
#   line (mp_hands.HAND_CONNECTIONS — the same pairs used to
#   draw the overlay), we compute the vector between the two
#   normalized points (dx, dy, dz) plus that edge's length.
#   This directly encodes the geometry of the red lines: their
#   direction and how long each "bone" is relative to the
#   others, which is what actually distinguishes one hand sign
#   from another (e.g. a bent finger vs a straight finger).
#   -> NUM_EDGES edges * (dx, dy, dz, length) numbers.
#
# PART A and PART B are concatenated into one feature vector,
# so the SVM is trained on — and the live camera is matched
# against — both the red-dot positions AND the red-line
# edge/vector geometry, not points alone.
# ============================================================

def landmarks_to_features(hand_landmarks):

    coords = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in hand_landmarks.landmark
        ],
        dtype=np.float32
    )

    wrist = coords[0].copy()

    coords = coords - wrist

    scale = np.linalg.norm(
        coords,
        axis=1
    ).max()

    if scale > 1e-6:
        coords = coords / scale

    # ---- PART A: red-dot point features ----
    point_features = coords.flatten()

    # ---- PART B: red-line edge/vector features ----
    edge_features = np.zeros(
        (NUM_EDGES, 4),
        dtype=np.float32
    )

    for i, (start_idx, end_idx) in enumerate(HAND_EDGES):

        vector = coords[end_idx] - coords[start_idx]

        length = np.linalg.norm(vector)

        edge_features[i, 0:3] = vector
        edge_features[i, 3] = length

    edge_features = edge_features.flatten()

    return np.concatenate(
        [point_features, edge_features]
    )


def detect_hand_landmarks(image_rgb, detector):

    results = detector.process(
        image_rgb
    )

    if not results.multi_hand_landmarks:
        return None

    return results.multi_hand_landmarks[0]


def draw_hand_overlay(image_rgb, hand_landmarks):
    """Draws the red centroid dots and red skeleton lines
    directly onto the RGB frame (used for the live camera)."""

    mp_drawing.draw_landmarks(
        image_rgb,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        LANDMARK_DRAW_SPEC,
        CONNECTION_DRAW_SPEC
    )

    return image_rgb


# ============================================================
# LOAD DATASET (as hand-landmark + edge/vector feature vectors)
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_dataset(dataset_path):

    X = []
    y = []

    classes = []
    skipped = 0

    detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=DATASET_MIN_DETECTION_CONFIDENCE
    )

    rng = np.random.default_rng(RANDOM_STATE)

    try:

        for folder in sorted(os.listdir(dataset_path)):

            folder_path = os.path.join(
                dataset_path,
                folder
            )

            if not os.path.isdir(
                folder_path
            ):
                continue

            images = [
                file
                for file in os.listdir(
                    folder_path
                )
                if file.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png"
                    )
                )
            ]

            if not images:
                continue

            if len(images) > MAX_IMAGES_PER_CLASS:

                images = list(
                    rng.choice(
                        images,
                        MAX_IMAGES_PER_CLASS,
                        replace=False
                    )
                )

            found_any = False

            for filename in images:

                path = os.path.join(
                    folder_path,
                    filename
                )

                try:

                    image = Image.open(
                        path
                    ).convert("RGB")

                    rgb = np.array(image)

                    hand_landmarks = detect_hand_landmarks(
                        rgb,
                        detector
                    )

                    if hand_landmarks is None:
                        skipped += 1
                        continue

                    features = landmarks_to_features(
                        hand_landmarks
                    )

                    X.append(
                        features
                    )

                    y.append(
                        folder
                    )

                    found_any = True

                except Exception:
                    skipped += 1
                    continue

            if found_any:
                classes.append(
                    folder
                )

    finally:

        detector.close()

    return (
        np.asarray(X),
        np.asarray(y),
        sorted(classes),
        skipped
    )


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "svm",
                SVC(
                    C=10,
                    kernel="rbf",
                    gamma="scale",
                    probability=True,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    return (
        model,
        accuracy,
        len(X_train),
        len(X_test)
    )


# ============================================================
# PREPARE MODEL
# ============================================================

with st.spinner(
    "SIGNOVA is mapping hand landmarks + edge geometry across the dataset..."
):

    dataset_path = extract_dataset()

    X, y, classes, skipped_images = load_dataset(
        dataset_path
    )

    if len(X) == 0:

        st.error(
            "No hands could be detected in any dataset image. "
            "SIGNOVA cannot train a landmark-based model on this dataset."
        )

        st.stop()

    model, accuracy, train_count, test_count = train_model(
        X,
        y
    )


# ============================================================
# REAL-TIME SHARED STATE
# ============================================================

class SignState:

    def __init__(self):

        self.lock = threading.Lock()

        self.prediction = "-"

        self.confidence = 0.0

        self.hand_detected = False

        self.stable_sign = None

        self.stable_count = 0

        self.last_added = None

        self.last_add_time = 0

        self.sentence = ""


sign_state = SignState()


# ============================================================
# VIDEO PROCESSOR
# Detects the hand every frame, overlays red centroid dots
# and red skeleton lines, turns the landmarks + edge/vector
# geometry into a feature vector, and compares that vector
# against the dataset-trained SVM to produce the current
# predicted letter.
# ============================================================

class SignovaVideoProcessor:

    def __init__(self):

        self.detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=0
        )

    def __del__(self):

        try:
            self.detector.close()
        except Exception:
            pass

    def recv(self, frame):

        rgb = frame.to_ndarray(
            format="rgb24"
        )

        try:

            hand_landmarks = detect_hand_landmarks(
                rgb,
                self.detector
            )

            current_time = time.time()

            if hand_landmarks is not None:

                # ------------------------------------------------
                # DRAW RED DOTS (centroids) + RED LINES (edges)
                # ------------------------------------------------

                rgb = draw_hand_overlay(
                    rgb,
                    hand_landmarks
                )

                features = landmarks_to_features(
                    hand_landmarks
                ).reshape(1, -1)

                probabilities = model.predict_proba(
                    features
                )[0]

                index = np.argmax(
                    probabilities
                )

                prediction = model.classes_[
                    index
                ]

                confidence = probabilities[
                    index
                ]

                with sign_state.lock:

                    sign_state.hand_detected = True

                    sign_state.prediction = str(
                        prediction
                    )

                    sign_state.confidence = float(
                        confidence
                    )

                    # ------------------------------------------------
                    # STABILITY LOGIC
                    # ------------------------------------------------

                    if (
                        confidence >= 0.70
                    ):

                        if (
                            sign_state.stable_sign
                            == prediction
                        ):

                            sign_state.stable_count += 1

                        else:

                            sign_state.stable_sign = prediction

                            sign_state.stable_count = 1

                        # ------------------------------------------------
                        # RECORD SIGN
                        # ------------------------------------------------

                        if (
                            sign_state.stable_count >= 12
                            and
                            sign_state.last_added
                            != prediction
                            and
                            current_time -
                            sign_state.last_add_time
                            > 1.0
                        ):

                            sign_state.sentence += str(
                                prediction
                            )

                            sign_state.last_added = str(
                                prediction
                            )

                            sign_state.last_add_time = (
                                current_time
                            )

                    else:

                        sign_state.stable_count = 0

                        # ------------------------------------------------
                        # LOW CONFIDENCE = READY FOR NEXT SIGN
                        # ------------------------------------------------

                        if confidence < 0.45:

                            sign_state.last_added = None

            else:

                with sign_state.lock:

                    sign_state.hand_detected = False

                    sign_state.prediction = "-"

                    sign_state.confidence = 0.0

                    sign_state.stable_count = 0

        except Exception:

            pass

        return av.VideoFrame.from_ndarray(
            rgb,
            format="rgb24"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">

            <div class="brand-icon">
                🤟
            </div>

            <div class="brand-name">
                SIGNOVA
            </div>

            <div class="brand-text">
                Real-Time Hand Sign Recognition
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "SYSTEM",
        [
            "Live Translator",
            "Dataset",
            "Model",
            "How It Works",
            "About"
        ]
    )

    st.markdown("---")

    st.markdown(
        "**SYSTEM STATUS**"
    )

    st.success(
        "Landmark + edge geometry engine ready"
    )

    st.caption(
        f"{len(classes)} classes"
    )

    st.caption(
        f"{len(y)} landmark samples"
    )

    if skipped_images:

        st.caption(
            f"{skipped_images} dataset images skipped (no hand detected)"
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
                LIVE COMPUTER VISION
            </div>

            <div class="hero-title">
                SIGNOVA
            </div>

            <div class="hero-subtitle">
                Turn individual hand signs into a live
                sentence using your camera. Red dots mark
                each detected hand joint; red lines trace
                the vectors between them.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # SENTENCE BOX
    # ========================================================

    with sign_state.lock:

        current_sentence = sign_state.sentence

        current_prediction = (
            sign_state.prediction
        )

        current_confidence = (
            sign_state.confidence
        )

        hand_detected = (
            sign_state.hand_detected
        )


    if current_sentence:

        sentence_html = f"""
        <div class="sentence-text">
            {current_sentence}
        </div>
        """

    else:

        sentence_html = """
        <div class="sentence-text empty-sentence">
            Your sentence will appear here...
        </div>
        """


    st.markdown(
        f"""
        <div class="sentence-container">

            <div class="sentence-label">
                LIVE SENTENCE
            </div>

            {sentence_html}

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # BUTTONS
    # ========================================================

    button1, button2, button3 = st.columns(
        [1, 1, 3]
    )


    with button1:

        if st.button(
            "🗑 Clear",
            use_container_width=True
        ):

            with sign_state.lock:

                sign_state.sentence = ""

                sign_state.last_added = None

                sign_state.stable_sign = None

                sign_state.stable_count = 0

            st.rerun()


    with button2:

        if st.button(
            "⌫ Delete",
            use_container_width=True
        ):

            with sign_state.lock:

                if sign_state.sentence:

                    sign_state.sentence = (
                        sign_state.sentence[:-1]
                    )

                    sign_state.last_added = None

            st.rerun()


    # ========================================================
    # CAMERA + PREDICTION
    # ========================================================

    camera_column, prediction_column = st.columns(
        [1.55, 1]
    )


    with camera_column:

        status_class = "status" if True else "status status-warning"

        st.markdown(
            """
            <div class="camera-card">

                <div class="status">
                    ● LIVE CAMERA — red dots/lines show detected hand
                </div>

                <br><br>

            </div>
            """,
            unsafe_allow_html=True
        )


        webrtc_ctx = webrtc_streamer(
            key="signova-camera",

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


    with prediction_column:

        detection_note = (
            "Hand detected"
            if hand_detected
            else "No hand in frame"
        )

        st.markdown(
            f"""
            <div class="prediction-box">

                <div class="prediction-title">
                    Current Sign
                </div>

                <div class="prediction-letter">
                    {current_prediction}
                </div>

                <div>
                    Confidence:
                    <strong>
                        {current_confidence * 100:.1f}%
                    </strong>
                </div>

                <div style="margin-top:8px; color:#64748b; font-size:12px;">
                    {detection_note}
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

            <strong>How to use SIGNOVA</strong><br><br>

            1. Start the camera.<br>
            2. Show one hand clearly — red dots and lines will
            appear on your knuckles and fingertips once it's
            detected.<br>
            3. Form a sign and keep it steady for a moment.<br>
            4. SIGNOVA compares the landmark + edge pattern
            against the dataset and records the closest
            matching letter automatically.<br>
            5. Move to a different sign.<br>
            6. Continue until your sentence is complete.<br>
            7. Press <strong>Clear</strong> to empty the sentence.

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # SYSTEM METRICS
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)


    with m1:

        st.markdown(
            f"""
            <div class="metric">

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
            <div class="metric">

                <div class="metric-number">
                    {len(y)}
                </div>

                <div class="metric-label">
                    Landmark Samples
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with m3:

        st.markdown(
            f"""
            <div class="metric">

                <div class="metric-number">
                    {accuracy * 100:.1f}%
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
            f"""
            <div class="metric">

                <div class="metric-number">
                    21-pt + {NUM_EDGES}-edge
                </div>

                <div class="metric-label">
                    Landmarks + SVM
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "Dataset":

    st.markdown(
        """
        <div class="hero">

            <div class="eyebrow">
                DATASET EXPLORER
            </div>

            <div class="hero-title">
                Dataset Lab
            </div>

            <div class="hero-subtitle">
                Explore the images used to teach SIGNOVA
                different hand-sign classes.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    with c1:
        st.metric(
            "Landmark Samples",
            len(y)
        )


    with c2:
        st.metric(
            "Classes",
            len(classes)
        )


    with c3:
        st.metric(
            "Average/Class",
            f"{len(y)/len(classes):.1f}"
        )


    st.markdown("<br>", unsafe_allow_html=True)


    selected = st.selectbox(
        "Select a sign",
        classes
    )


    folder = os.path.join(
        dataset_path,
        selected
    )


    files = [
        f
        for f in os.listdir(
            folder
        )
        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ]


    st.subheader(
        f"Sign: {selected}"
    )


    st.caption(
        f"{len(files)} images in this class folder "
        "(some may have been skipped during training if no "
        "hand landmarks were detected)"
    )


    columns = st.columns(5)


    for i, filename in enumerate(
        files[:20]
    ):

        path = os.path.join(
            folder,
            filename
        )

        try:

            image = Image.open(
                path
            )

            with columns[
                i % 5
            ]:

                st.image(
                    image,
                    use_container_width=True
                )

        except Exception:

            pass


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
                Model Insights
            </div>

            <div class="hero-subtitle">
                Performance information for the SIGNOVA
                hand-landmark + edge/vector classifier.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )


    with c2:

        st.metric(
            "Training",
            train_count
        )


    with c3:

        st.metric(
            "Testing",
            test_count
        )


    st.markdown(
        """
        <div class="instruction">

        SIGNOVA uses an 80/20 stratified train-test split.
        The model learns from the training subset's landmark +
        edge/vector features and is evaluated against feature
        vectors from images it never saw during training.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.subheader(
        "Classification Method"
    )

    st.write(
        f"""
        **MediaPipe Hand Landmarks + Edge/Vector Geometry + Support Vector Machine (SVM)**

        Every image (dataset or live camera) is first run through
        MediaPipe's hand-landmark detector, which locates
        {LANDMARKS_PER_HAND} keypoints (fingertips, knuckles,
        wrist). Each point is drawn as a red dot, and the red
        lines connecting them mirror the physical bone structure
        of the hand — a vector skeleton of the sign being made.

        The feature vector fed to the SVM is built from **two**
        parts, both computed from those same red dots and red
        lines:

        - **Points ({POINT_FEATURES} numbers):** each of the
          {LANDMARKS_PER_HAND} landmarks, re-centered on the
          wrist and rescaled by hand size (x, y, z per point).
        - **Edges / vectors ({EDGE_FEATURES} numbers):** for
          every one of the {NUM_EDGES} red skeleton lines
          (the exact same `HAND_CONNECTIONS` pairs used to draw
          the overlay), the (dx, dy, dz) vector between its two
          points plus that edge's length — capturing finger
          bend/angle, not just where each point sits.

        Concatenated, that's a {TOTAL_FEATURES}-number feature
        vector, re-centered and rescaled so the same sign looks
        the same whether it's close to or far from the camera.
        An RBF-kernel SVM then classifies that vector into one
        of the available hand-sign categories.
        """
    )


# ============================================================
# HOW IT WORKS
# ============================================================

elif page == "How It Works":

    st.markdown(
        """
        <div class="hero">

            <div class="eyebrow">
                COMPUTER VISION PIPELINE
            </div>

            <div class="hero-title">
                How SIGNOVA Works
            </div>

            <div class="hero-subtitle">
                Understanding the journey from camera frame
                to sentence.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    steps = [

        (
            "01",
            "Live Camera",
            "The user's webcam continuously provides video frames."
        ),

        (
            "02",
            "Hand Detection",
            "MediaPipe locates the hand in the frame and identifies "
            "21 landmark points — fingertips, knuckles, palm base, "
            "and wrist."
        ),

        (
            "03",
            "Red Dot Overlay",
            "Each of the 21 landmarks is drawn on the video feed as "
            "a red dot (a centroid marking a specific hand joint)."
        ),

        (
            "04",
            "Red Line Overlay",
            "Red lines are drawn between anatomically connected "
            "landmarks (mp_hands.HAND_CONNECTIONS), forming a "
            "skeleton of edges/vectors that traces the shape of "
            "the hand."
        ),

        (
            "05",
            "Normalization",
            "Every point is shifted so the wrist sits at the origin, "
            "then rescaled by the hand's own size — making the "
            "pattern independent of distance from the camera."
        ),

        (
            "06",
            "Point + Edge Feature Vector",
            f"The {LANDMARKS_PER_HAND} normalized points become "
            f"{POINT_FEATURES} point features. Each of the same "
            f"{NUM_EDGES} red lines drawn on screen is turned into "
            f"a (dx, dy, dz, length) edge/vector — {EDGE_FEATURES} "
            "more numbers. Together they form one "
            f"{TOTAL_FEATURES}-number feature vector describing "
            "both where the joints are and how the 'bones' between "
            "them are angled."
        ),

        (
            "07",
            "Dataset Comparison",
            "The same point + edge process is run once over every "
            "training image, producing a labeled set of feature "
            "vectors for every letter/sign in the dataset."
        ),

        (
            "08",
            "SVM",
            "The live feature vector is passed to a trained "
            "Support Vector Machine, which compares it against the "
            "point + edge patterns learned from the dataset."
        ),

        (
            "09",
            "Confidence & Stability",
            "The classifier estimates a probability for each sign; "
            "a prediction must stay stable across several frames "
            "before it's accepted."
        ),

        (
            "10",
            "Sentence Builder",
            "The accepted sign is appended to the sentence exactly "
            "once."
        )
    ]


    for number, title, description in steps:

        st.markdown(
            f"""
            <div class="camera-card">

                <strong>
                    {number} — {title}
                </strong>

                <br><br>

                <span style="color:#94a3b8;">
                    {description}
                </span>

            </div>

            <br>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.markdown(
        """
        <div class="hero">

            <div class="eyebrow">
                PROJECT
            </div>

            <div class="hero-title">
                About SIGNOVA
            </div>

            <div class="hero-subtitle">
                A real-time static hand-sign recognition
                prototype developed for an Image Processing
                and Computer Vision assignment.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="camera-card">

        <h3>Project Objective</h3>

        SIGNOVA demonstrates how hand-landmark detection and
        machine learning can be combined to recognize static
        hand signs from a live camera. Rather than comparing raw
        pixels, SIGNOVA tracks 21 skeletal points on the hand —
        visualized as red dots connected by red edge lines — and
        turns both the dot positions and the line/vector geometry
        between them into the feature vector compared against the
        labeled dataset.

        Instead of simply displaying one prediction, SIGNOVA uses
        a sentence builder that records stable predictions one at
        a time.

        <br><br>

        <strong>Technology:</strong>

        <br><br>

        Python • Streamlit • Streamlit-WebRTC • MediaPipe •
        OpenCV • NumPy • Pillow • Scikit-learn • SVM

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="footer">

        SIGNOVA<br>
        Real-Time Hand Sign Recognition System<br>
        Image Processing & Computer Vision

        </div>
        """,
        unsafe_allow_html=True
    )
