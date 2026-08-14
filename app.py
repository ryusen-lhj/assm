import os
import zipfile
import threading
import time

import av
import numpy as np
import streamlit as st

from PIL import Image, ImageDraw, ImageFont

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import mediapipe as mp

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode
)


# ============================================================
# SIGNOVA
# REAL-TIME HAND SIGN RECOGNITION
# ============================================================

st.set_page_config(
    page_title="SIGNOVA",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(124, 58, 237, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 85% 15%,
            rgba(14, 165, 233, 0.12),
            transparent 30%
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


/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #0b0f1a;
    border-right: 1px solid rgba(255,255,255,0.07);
}


/* BRAND */

.brand-icon {
    width: 55px;
    height: 55px;

    border-radius: 16px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 28px;

    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #2563eb
        );
}

.brand-title {
    font-size: 26px;
    font-weight: 900;
    margin-top: 12px;
    letter-spacing: 1px;
}

.brand-subtitle {
    color: #64748b;
    font-size: 12px;
}


/* HERO */

.eyebrow {
    color: #818cf8;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 2px;
}

.hero-title {
    font-size: 50px;
    font-weight: 950;
    letter-spacing: -2px;

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
}


/* SENTENCE */

.sentence-box {
    margin: 20px 0;

    padding: 20px 24px;

    border-radius: 20px;

    background:
        linear-gradient(
            145deg,
            rgba(124,58,237,0.16),
            rgba(37,99,235,0.08)
        );

    border:
        1px solid rgba(139,92,246,0.30);
}

.sentence-label {
    color: #818cf8;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 2px;
}

.sentence {
    min-height: 60px;

    display: flex;
    align-items: center;

    font-size: 34px;
    font-weight: 900;

    letter-spacing: 5px;

    margin-top: 8px;
}

.empty {
    color: #475569;
    font-size: 16px;
    letter-spacing: 0;
    font-weight: 500;
}


/* DETECTION */

.detection {
    border-radius: 18px;

    padding: 18px;

    background:
        rgba(15,23,42,0.75);

    border:
        1px solid rgba(255,255,255,0.07);

    text-align: center;
}

.detected {
    color: #4ade80;
    font-weight: 900;
}

.not-detected {
    color: #64748b;
    font-weight: 900;
}


/* PREDICTION */

.prediction {
    text-align: center;

    padding: 20px;

    border-radius: 20px;

    background:
        rgba(15,23,42,0.75);

    border:
        1px solid rgba(255,255,255,0.07);
}

.prediction-label {
    color: #64748b;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 2px;
}

.letter {
    font-size: 95px;
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


/* INFO */

.info {
    background:
        rgba(99,102,241,0.07);

    border-left:
        3px solid #6366f1;

    border-radius: 8px;

    padding: 15px;

    color: #cbd5e1;

    font-size: 13px;

    line-height: 1.7;
}


/* METRICS */

.metric {
    background:
        rgba(15,23,42,0.75);

    border:
        1px solid rgba(255,255,255,0.07);

    border-radius: 16px;

    padding: 18px;
}

.metric-value {
    font-size: 26px;
    font-weight: 900;
}

.metric-label {
    color: #64748b;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

ZIP_FILE = "archive.zip"

EXTRACT_FOLDER = "signova_dataset"

CONFIDENCE_THRESHOLD = 0.65

STABLE_FRAMES = 8

MIN_HAND_SIZE = 40


# ============================================================
# SESSION STATE
# ============================================================

if "sentence" not in st.session_state:
    st.session_state.sentence = ""


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils


# ============================================================
# DATASET
# ============================================================

def find_dataset():

    if not os.path.exists(
        EXTRACT_FOLDER
    ):
        return None

    for root, dirs, files in os.walk(
        EXTRACT_FOLDER
    ):

        if os.path.basename(
            root
        ).upper() == "DATASET":

            return root

    return None


def extract_dataset():

    dataset = find_dataset()

    if dataset:
        return dataset

    if not os.path.exists(
        ZIP_FILE
    ):

        st.error(
            "archive.zip was not found."
        )

        st.stop()

    os.makedirs(
        EXTRACT_FOLDER,
        exist_ok=True
    )

    with zipfile.ZipFile(
        ZIP_FILE,
        "r"
    ) as z:

        z.extractall(
            EXTRACT_FOLDER
        )

    dataset = find_dataset()

    if dataset is None:

        st.error(
            "DATASET folder was not found inside archive.zip."
        )

        st.stop()

    return dataset


# ============================================================
# LANDMARK FEATURE EXTRACTION
# ============================================================

def landmark_features(landmarks):

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
    # Normalize using wrist
    # --------------------------------------------------------

    wrist = points[0].copy()

    points = points - wrist


    # --------------------------------------------------------
    # Scale normalization
    # --------------------------------------------------------

    scale = np.max(
        np.linalg.norm(
            points[:, :2],
            axis=1
        )
    )

    if scale > 0:

        points = points / scale


    # --------------------------------------------------------
    # Add vector information
    # --------------------------------------------------------

    vectors = []

    connections = [
        (0,1),
        (1,2),
        (2,3),
        (3,4),

        (0,5),
        (5,6),
        (6,7),
        (7,8),

        (0,9),
        (9,10),
        (10,11),
        (11,12),

        (0,13),
        (13,14),
        (14,15),
        (15,16),

        (0,17),
        (17,18),
        (18,19),
        (19,20)
    ]


    for a, b in connections:

        vector = points[b] - points[a]

        vectors.extend(
            vector.tolist()
        )


    features = np.concatenate(
        [
            points.flatten(),
            np.array(
                vectors,
                dtype=np.float32
            )
        ]
    )


    return features


# ============================================================
# EXTRACT LANDMARKS FROM IMAGE
# ============================================================

def get_image_landmarks(
    image,
    hands_detector
):

    rgb = np.array(
        image.convert("RGB")
    )

    results = hands_detector.process(
        rgb
    )

    if not results.multi_hand_landmarks:

        return None

    return results.multi_hand_landmarks[0].landmark


# ============================================================
# LOAD DATASET LANDMARKS
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_landmark_dataset(
    dataset_path
):

    X = []
    y = []

    class_names = []


    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as detector:


        for class_name in sorted(
            os.listdir(dataset_path)
        ):

            folder = os.path.join(
                dataset_path,
                class_name
            )

            if not os.path.isdir(
                folder
            ):

                continue


            image_files = [
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


            valid_class = False


            for filename in image_files:

                path = os.path.join(
                    folder,
                    filename
                )

                try:

                    image = Image.open(
                        path
                    ).convert(
                        "RGB"
                    )

                    landmarks = get_image_landmarks(
                        image,
                        detector
                    )


                    if landmarks is None:

                        continue


                    features = landmark_features(
                        landmarks
                    )


                    X.append(
                        features
                    )

                    y.append(
                        class_name
                    )

                    valid_class = True

                except Exception:

                    continue


            if valid_class:

                class_names.append(
                    class_name
                )


    return (
        np.asarray(X),
        np.asarray(y),
        class_names
    )


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def train_classifier(
    X,
    y
):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=5,
                    weights="distance"
                )
            )
        ]
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    return (
        model,
        accuracy,
        len(X_train),
        len(X_test)
    )


# ============================================================
# PREPARE DATA
# ============================================================

dataset_path = extract_dataset()


with st.spinner(
    "SIGNOVA is analysing the hand-sign dataset..."
):

    X, y, classes = load_landmark_dataset(
        dataset_path
    )


if len(X) == 0:

    st.error(
        """
        No hand landmarks could be extracted from
        the dataset.

        Make sure the images contain clearly visible hands.
        """
    )

    st.stop()


model, accuracy, train_count, test_count = train_classifier(
    X,
    y
)


# ============================================================
# SHARED REAL-TIME STATE
# ============================================================

class DetectionState:

    def __init__(self):

        self.lock = threading.Lock()

        self.hand_detected = False

        self.prediction = "-"

        self.confidence = 0.0

        self.sentence = ""

        self.stable_prediction = None

        self.stable_count = 0

        self.last_recorded = None

        self.last_record_time = 0


state = DetectionState()


# ============================================================
# DRAW HAND OVERLAY
# ============================================================

def draw_hand_overlay(
    image,
    landmarks
):

    draw = ImageDraw.Draw(
        image
    )


    width, height = image.size


    # --------------------------------------------------------
    # GREEN BOUNDING BOX
    # --------------------------------------------------------

    xs = [
        int(lm.x * width)
        for lm in landmarks
    ]

    ys = [
        int(lm.y * height)
        for lm in landmarks
    ]


    min_x = max(
        min(xs) - 20,
        0
    )

    max_x = min(
        max(xs) + 20,
        width
    )

    min_y = max(
        min(ys) - 20,
        0
    )

    max_y = min(
        max(ys) + 20,
        height
    )


    # Green hand detection frame

    for offset in range(4):

        draw.rectangle(
            [
                min_x - offset,
                min_y - offset,
                max_x + offset,
                max_y + offset
            ],

            outline="lime",
            width=1
        )


    # --------------------------------------------------------
    # HAND CONNECTIONS
    # --------------------------------------------------------

    connections = [

        (0,1),
        (1,2),
        (2,3),
        (3,4),

        (0,5),
        (5,6),
        (6,7),
        (7,8),

        (0,9),
        (9,10),
        (10,11),
        (11,12),

        (0,13),
        (13,14),
        (14,15),
        (15,16),

        (0,17),
        (17,18),
        (18,19),
        (19,20),

        (5,9),
        (9,13),
        (13,17),
        (0,17),
        (0,5)
    ]


    # --------------------------------------------------------
    # DRAW VECTORS / EDGES
    # --------------------------------------------------------

    for a, b in connections:

        x1 = int(
            landmarks[a].x * width
        )

        y1 = int(
            landmarks[a].y * height
        )

        x2 = int(
            landmarks[b].x * width
        )

        y2 = int(
            landmarks[b].y * height
        )


        draw.line(
            [
                (x1, y1),
                (x2, y2)
            ],

            fill="white",
            width=3
        )


    # --------------------------------------------------------
    # RED LANDMARK POINTS
    # --------------------------------------------------------

    for landmark in landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )


        radius = 6


        draw.ellipse(
            [
                x - radius,
                y - radius,
                x + radius,
                y + radius
            ],

            fill="red",
            outline="white",
            width=1
        )


    return image


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class SignovaProcessor:

    def __init__(self):

        self.hands = mp_hands.Hands(

            static_image_mode=False,

            max_num_hands=1,

            min_detection_confidence=0.60,

            min_tracking_confidence=0.60
        )


    def recv(self, frame):

        image = frame.to_image().convert(
            "RGB"
        )


        image_array = np.array(
            image
        )


        # ----------------------------------------------------
        # MEDIAPIPE HAND DETECTION
        # ----------------------------------------------------

        results = self.hands.process(
            image_array
        )


        with state.lock:

            state.hand_detected = False


        # ----------------------------------------------------
        # HAND FOUND
        # ----------------------------------------------------

        if results.multi_hand_landmarks:

            landmarks = (
                results.multi_hand_landmarks[0]
                .landmark
            )


            # ------------------------------------------------
            # DRAW RED POINTS + WHITE VECTORS + GREEN BOX
            # ------------------------------------------------

            image = draw_hand_overlay(
                image,
                landmarks
            )


            # ------------------------------------------------
            # CLASSIFY
            # ------------------------------------------------

            features = landmark_features(
                landmarks
            )


            features = features.reshape(
                1,
                -1
            )


            probabilities = model.predict_proba(
                features
            )[0]


            best_index = np.argmax(
                probabilities
            )


            prediction = model.classes_[
                best_index
            ]


            confidence = float(
                probabilities[
                    best_index
                ]
            )


            current_time = time.time()


            with state.lock:

                state.hand_detected = True

                state.prediction = str(
                    prediction
                )

                state.confidence = confidence


                # --------------------------------------------
                # STABILITY
                # --------------------------------------------

                if (
                    state.stable_prediction
                    == prediction
                ):

                    state.stable_count += 1

                else:

                    state.stable_prediction = prediction

                    state.stable_count = 1


                # --------------------------------------------
                # RECORD LETTER
                # --------------------------------------------

                if (

                    confidence
                    >= CONFIDENCE_THRESHOLD

                    and

                    state.stable_count
                    >= STABLE_FRAMES

                    and

                    state.last_recorded
                    != prediction

                    and

                    current_time
                    -
                    state.last_record_time
                    > 0.8

                ):

                    state.sentence += str(
                        prediction
                    )

                    state.last_recorded = str(
                        prediction
                    )

                    state.last_record_time = (
                        current_time
                    )


        else:

            # ------------------------------------------------
            # NO HAND
            # ------------------------------------------------

            with state.lock:

                state.hand_detected = False

                state.prediction = "-"

                state.confidence = 0.0

                state.stable_prediction = None

                state.stable_count = 0

                # Allow another identical letter
                # after removing hand from camera

                state.last_recorded = None


        return av.VideoFrame.from_ndarray(
            np.array(image),
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
            Computer Vision Sign Recognition
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
            "Computer Vision",
            "About"
        ]
    )


    st.markdown("---")


    st.markdown(
        "**RECOGNITION ENGINE**"
    )


    st.success(
        "ONLINE"
    )


    st.caption(
        f"{len(classes)} hand-sign classes"
    )

    st.caption(
        f"{len(X)} valid hand samples"
    )


# ============================================================
# LIVE TRANSLATOR
# ============================================================

if page == "Live Translator":

    st.markdown(
        """
        <div class="eyebrow">
            REAL-TIME HAND VISION
        </div>

        <div class="hero-title">
            SIGNOVA
        </div>

        <div class="hero-subtitle">
            Show a hand sign to the camera.
            SIGNOVA detects the hand, maps its 21 landmarks,
            compares the hand structure with the dataset,
            and records the predicted letter.
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # SENTENCE BOX
    # ========================================================

    with state.lock:

        sentence = state.sentence

        prediction = state.prediction

        confidence = state.confidence

        hand_detected = state.hand_detected


    if sentence:

        sentence_content = sentence

        sentence_class = "sentence"

    else:

        sentence_content = (
            "Detected letters will appear here..."
        )

        sentence_class = "sentence empty"


    st.markdown(
        f"""
        <div class="sentence-box">

            <div class="sentence-label">
                DETECTED SENTENCE
            </div>

            <div class="{sentence_class}">
                {sentence_content}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # BUTTONS
    # ========================================================

    b1, b2, b3 = st.columns(
        [1,1,4]
    )


    with b1:

        if st.button(
            "🗑 Clear",
            use_container_width=True
        ):

            with state.lock:

                state.sentence = ""

                state.last_recorded = None

                state.stable_prediction = None

                state.stable_count = 0


            st.rerun()


    with b2:

        if st.button(
            "⌫ Delete",
            use_container_width=True
        ):

            with state.lock:

                if state.sentence:

                    state.sentence = (
                        state.sentence[:-1]
                    )

                    state.last_recorded = None


            st.rerun()


    st.markdown("<br>", unsafe_allow_html=True)


    # ========================================================
    # CAMERA
    # ========================================================

    camera, result = st.columns(
        [1.55, 1]
    )


    with camera:

        st.markdown(
            """
            <div class="detection">

                🟢
                <span class="detected">
                HAND TRACKING ENABLED
                </span>

                <br><br>

                🔴 Points = hand landmarks<br>
                ━ Lines = vectors / skeleton<br>
                🟩 Green = detected hand region

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown("<br>", unsafe_allow_html=True)


        webrtc_streamer(

            key="signova-hand-camera",

            mode=WebRtcMode.SENDRECV,

            video_processor_factory=SignovaProcessor,

            media_stream_constraints={
                "video": True,
                "audio": False
            },

            async_processing=True
        )


    # ========================================================
    # RESULT
    # ========================================================

    with result:

        if hand_detected:

            detection_text = (
                "🟢 HAND DETECTED"
            )

            detection_class = (
                "detected"
            )

        else:

            detection_text = (
                "○ NO HAND DETECTED"
            )

            detection_class = (
                "not-detected"
            )


        st.markdown(
            f"""
            <div class="detection">

                <div class="{detection_class}">
                    {detection_text}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown("<br>", unsafe_allow_html=True)


        st.markdown(
            f"""
            <div class="prediction">

                <div class="prediction-label">
                    PREDICTED LETTER
                </div>

                <div class="letter">
                    {prediction}
                </div>

                <div>
                    Confidence:
                    <strong>
                        {confidence * 100:.1f}%
                    </strong>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown("<br>", unsafe_allow_html=True)


        st.markdown(
            """
            <div class="info">

            <strong>How to create a sentence</strong>

            <br><br>

            Show one sign and keep your hand steady.
            SIGNOVA waits until the same sign is stable
            before adding it to the letter box.

            <br><br>

            Remove your hand briefly before showing
            the same letter again.

            <br><br>

            Example:

            <br><br>

            🤟 H → 🖐 E → 🤟 L → 🤟 L → 🤟 O

            <br><br>

            Result:

            <strong>HELLO</strong>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # METRICS
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)


    m1, m2, m3, m4 = st.columns(4)


    with m1:

        st.markdown(
            f"""
            <div class="metric">

                <div class="metric-value">
                    {len(classes)}
                </div>

                <div class="metric-label">
                    SIGN CLASSES
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with m2:

        st.markdown(
            f"""
            <div class="metric">

                <div class="metric-value">
                    21
                </div>

                <div class="metric-label">
                    HAND POINTS
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
                    {accuracy * 100:.1f}%
                </div>

                <div class="metric-label">
                    TEST ACCURACY
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with m4:

        st.markdown(
            """
            <div class="metric">

                <div class="metric-value">
                    KNN
                </div>

                <div class="metric-label">
                    LANDMARK MODEL
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
        <div class="eyebrow">
            TRAINING DATA
        </div>

        <div class="hero-title">
            Dataset Explorer
        </div>

        <div class="hero-subtitle">
            SIGNOVA extracts hand landmarks from the
            supplied dataset and uses them to train
            the recognition model.
        </div>
        """,
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Classes",
            len(classes)
        )


    with c2:

        st.metric(
            "Valid Samples",
            len(X)
        )


    with c3:

        st.metric(
            "Landmarks",
            21
        )


    st.markdown("<br>", unsafe_allow_html=True)


    st.subheader(
        "Available Hand Signs"
    )


    cols = st.columns(5)


    for i, class_name in enumerate(
        classes
    ):

        with cols[i % 5]:

            st.markdown(
                f"""
                <div class="metric">

                    <div class="metric-value">
                        {class_name}
                    </div>

                    <div class="metric-label">
                        HAND SIGN
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
        <div class="eyebrow">
            MACHINE LEARNING
        </div>

        <div class="hero-title">
            Recognition Model
        </div>

        <div class="hero-subtitle">
            The classifier compares normalized hand
            landmarks rather than raw image pixels.
        </div>
        """,
        unsafe_allow_html=True
    )


    a, b, c = st.columns(3)


    with a:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )


    with b:

        st.metric(
            "Training Samples",
            train_count
        )


    with c:

        st.metric(
            "Testing Samples",
            test_count
        )


    st.markdown("<br>", unsafe_allow_html=True)


    st.markdown(
        """
        <div class="info">

        <strong>Why landmarks?</strong>

        <br><br>

        Instead of comparing the complete image, SIGNOVA
        represents the hand using 21 landmark points.

        Each point contains:

        <br><br>

        • X coordinate<br>
        • Y coordinate<br>
        • Z coordinate

        <br>

        SIGNOVA also calculates vectors between connected
        landmarks. This allows the system to describe the
        shape and orientation of the hand.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("<br>", unsafe_allow_html=True)


    st.subheader(
        "Recognition Pipeline"
    )


    pipeline = [
        "Camera Frame",
        "Hand Detection",
        "21 Landmark Points",
        "Coordinate Normalisation",
        "Vector Extraction",
        "KNN Classification",
        "Confidence Score",
        "Stable Sign",
        "Sentence Builder"
    ]


    for i, step in enumerate(
        pipeline,
        1
    ):

        st.write(
            f"**{i}.** {step}"
        )


# ============================================================
# COMPUTER VISION PAGE
# ============================================================

elif page == "Computer Vision":

    st.markdown(
        """
        <div class="eyebrow">
            IMAGE PROCESSING
        </div>

        <div class="hero-title">
            Computer Vision
        </div>

        <div class="hero-subtitle">
            Visual representation of the features
            SIGNOVA uses to recognize a hand.
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info">

        <strong>🔴 Landmark Points</strong>

        <br>

        SIGNOVA detects 21 key points on the hand.
        These represent important joints such as the
        wrist and finger joints.

        <br><br>

        <strong>━ Edges / Vectors</strong>

        <br>

        Lines connect related landmarks. These vectors
        describe the direction and structure of each
        finger.

        <br><br>

        <strong>🟩 Detection Frame</strong>

        <br>

        The green rectangle represents the region
        containing the detected hand.

        <br><br>

        <strong>Feature Comparison</strong>

        <br>

        The landmark coordinates and vectors are
        normalized before being compared with the
        hand-sign samples extracted from the dataset.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("<br>", unsafe_allow_html=True)


    st.subheader(
        "21 Hand Landmarks"
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


    cols = st.columns(3)


    for i, name in enumerate(
        landmark_names
    ):

        with cols[i % 3]:

            st.write(
                f"🔴 {name}"
            )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.markdown(
        """
        <div class="eyebrow">
            SIGNOVA PROJECT
        </div>

        <div class="hero-title">
            About SIGNOVA
        </div>

        <div class="hero-subtitle">
            Real-time hand sign recognition using
            computer vision and machine learning.
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info">

        SIGNOVA is an Image Processing and Computer Vision
        project designed to recognize static hand signs
        through a webcam.

        <br><br>

        The system detects the user's hand, extracts
        21 landmark points, calculates hand vectors,
        compares the resulting features with the supplied
        dataset, and predicts the corresponding sign.

        <br><br>

        The predicted signs can then be accumulated into
        a sentence in real time.

        </div>
        """,
        unsafe_allow_html=True
    )
