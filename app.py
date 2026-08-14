import os
import zipfile
import threading
import time

import av
import numpy as np
import streamlit as st
import mediapipe as mp

from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from streamlit_webrtc import webrtc_streamer, WebRtcMode


# ============================================================
# SIGNOVA
# REAL-TIME HAND SIGN LANGUAGE RECOGNITION
# ============================================================

st.set_page_config(
    page_title="SIGNOVA",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

ZIP_FILE = "archive.zip"
EXTRACT_FOLDER = "signova_dataset"

CONFIDENCE_THRESHOLD = 0.65
STABLE_FRAMES = 8
RECORD_COOLDOWN = 0.8


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(124,58,237,0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 85% 15%,
            rgba(14,165,233,0.12),
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

[data-testid="stSidebar"] {
    background: #0b0f1a;
    border-right: 1px solid rgba(255,255,255,0.07);
}


/* BRAND */

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
}

.brand-title {
    font-size: 28px;
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
    font-size: 52px;
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
    margin: 22px 0;

    padding: 20px 25px;

    min-height: 95px;

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
    font-size: 35px;
    font-weight: 900;

    letter-spacing: 5px;

    margin-top: 8px;

    word-break: break-word;
}

.empty {
    color: #475569;
    font-size: 16px;
    letter-spacing: 0;
    font-weight: 500;
}


/* CARDS */

.card {
    padding: 22px;

    border-radius: 20px;

    background:
        rgba(15,23,42,0.75);

    border:
        1px solid rgba(255,255,255,0.07);
}


/* PREDICTION */

.prediction {
    text-align: center;

    padding: 30px;

    border-radius: 22px;

    background:
        rgba(15,23,42,0.80);

    border:
        1px solid rgba(255,255,255,0.08);
}

.prediction-label {
    color: #64748b;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 2px;
}

.letter {
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


/* STATUS */

.status-online {
    color: #4ade80;
    font-weight: 900;
}

.status-offline {
    color: #64748b;
    font-weight: 900;
}


/* METRIC */

.metric {
    background:
        rgba(15,23,42,0.75);

    border:
        1px solid rgba(255,255,255,0.07);

    border-radius: 16px;

    padding: 18px;
}

.metric-value {
    font-size: 27px;
    font-weight: 900;
}

.metric-label {
    color: #64748b;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}


/* INFO */

.info {
    background:
        rgba(99,102,241,0.07);

    border-left:
        3px solid #6366f1;

    border-radius: 8px;

    padding: 17px;

    color: #cbd5e1;

    font-size: 13px;

    line-height: 1.7;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Live Translator"


# ============================================================
# SHARED REAL-TIME STATE
# ============================================================

if "shared_state" not in st.session_state:

    st.session_state.shared_state = {
        "hand_detected": False,
        "prediction": "-",
        "confidence": 0.0,
        "sentence": "",
        "stable_prediction": None,
        "stable_count": 0,
        "last_recorded": None,
        "last_record_time": 0.0
    }


if "state_lock" not in st.session_state:
    st.session_state.state_lock = threading.Lock()


shared = st.session_state.shared_state
state_lock = st.session_state.state_lock


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands


# ============================================================
# DATASET EXTRACTION
# ============================================================

def locate_dataset():

    if not os.path.exists(EXTRACT_FOLDER):
        return None

    candidate = os.path.join(
        EXTRACT_FOLDER,
        "DATASET"
    )

    if os.path.isdir(candidate):
        return candidate

    return None


def prepare_dataset():

    existing = locate_dataset()

    if existing:
        return existing

    if not os.path.exists(ZIP_FILE):

        st.error(
            "archive.zip was not found in the GitHub repository."
        )

        st.stop()

    os.makedirs(
        EXTRACT_FOLDER,
        exist_ok=True
    )

    with zipfile.ZipFile(
        ZIP_FILE,
        "r"
    ) as archive:

        archive.extractall(
            EXTRACT_FOLDER
        )

    dataset = locate_dataset()

    if dataset is None:

        st.error(
            """
            SIGNOVA could not find:

            DATASET/

            inside archive.zip.
            """
        )

        st.stop()

    return dataset


dataset_path = prepare_dataset()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

CONNECTIONS = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (5, 9),
    (9, 13),
    (13, 17)
]


def create_features(landmarks):

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
    # TRANSLATION NORMALISATION
    # --------------------------------------------------------

    wrist = points[0].copy()

    points = points - wrist


    # --------------------------------------------------------
    # SCALE NORMALISATION
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
    # VECTOR FEATURES
    # --------------------------------------------------------

    vectors = []

    for a, b in CONNECTIONS:

        vector = (
            points[b]
            -
            points[a]
        )

        vectors.extend(
            vector.tolist()
        )


    vectors = np.asarray(
        vectors,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # FINAL FEATURE VECTOR
    # --------------------------------------------------------

    features = np.concatenate(
        [
            points.flatten(),
            vectors
        ]
    )


    return features.astype(
        np.float32
    )


# ============================================================
# DATASET LOADING
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_dataset(dataset_folder):

    X = []
    y = []

    class_names = []


    # IMPORTANT:
    # Dataset contains 0-9 and A-Z.

    expected_classes = (
        [str(i) for i in range(10)]
        +
        [
            chr(i)
            for i in range(
                ord("A"),
                ord("Z") + 1
            )
        ]
    )


    with mp_hands.Hands(

        static_image_mode=True,

        max_num_hands=1,

        min_detection_confidence=0.50

    ) as detector:


        for class_name in expected_classes:

            folder = os.path.join(
                dataset_folder,
                class_name
            )


            if not os.path.isdir(folder):
                continue


            class_count = 0


            image_files = sorted(
                [
                    filename

                    for filename
                    in os.listdir(folder)

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

                filepath = os.path.join(
                    folder,
                    filename
                )


                try:

                    image = Image.open(
                        filepath
                    ).convert(
                        "RGB"
                    )


                    image_array = np.asarray(
                        image
                    )


                    results = detector.process(
                        image_array
                    )


                    if not results.multi_hand_landmarks:
                        continue


                    landmarks = (
                        results
                        .multi_hand_landmarks[0]
                        .landmark
                    )


                    features = create_features(
                        landmarks
                    )


                    X.append(features)

                    y.append(class_name)

                    class_count += 1


                except Exception:
                    continue


            if class_count > 0:
                class_names.append(
                    class_name
                )


    return (
        np.asarray(X),
        np.asarray(y),
        class_names
    )


# ============================================================
# LOAD DATA
# ============================================================

with st.spinner(
    "SIGNOVA is extracting hand landmarks from your 900-image dataset..."
):

    X, y, classes = load_dataset(
        dataset_path
    )


if len(X) == 0:

    st.error(
        """
        No hands were detected in the dataset.

        Please check that the dataset images contain
        visible hands.
        """
    )

    st.stop()


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

        random_state=42,

        stratify=y
    )


    model = Pipeline(

        [
            (
                "normalizer",
                StandardScaler()
            ),

            (
                "knn",
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


model, accuracy, train_count, test_count = train_model(
    X,
    y
)


# ============================================================
# DRAW VISUAL OVERLAY
# ============================================================

def draw_overlay(
    frame,
    landmarks,
    prediction,
    confidence
):

    image = Image.fromarray(
        frame
    ).convert(
        "RGB"
    )

    draw = ImageDraw.Draw(
        image
    )


    width, height = image.size


    # --------------------------------------------------------
    # LANDMARK PIXELS
    # --------------------------------------------------------

    coords = [

        (
            int(lm.x * width),
            int(lm.y * height)
        )

        for lm in landmarks
    ]


    xs = [
        p[0]
        for p in coords
    ]

    ys = [
        p[1]
        for p in coords
    ]


    # --------------------------------------------------------
    # GREEN DETECTION FRAME
    # --------------------------------------------------------

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

        outline="lime",

        width=5
    )


    # --------------------------------------------------------
    # DRAW VECTORS / EDGES
    # --------------------------------------------------------

    for a, b in CONNECTIONS:

        x1, y1 = coords[a]

        x2, y2 = coords[b]


        draw.line(

            [
                (x1, y1),
                (x2, y2)
            ],

            fill="white",

            width=3
        )


    # --------------------------------------------------------
    # DRAW RED POINTS
    # --------------------------------------------------------

    for x, y_point in coords:

        radius = 6


        draw.ellipse(

            [
                x - radius,
                y_point - radius,
                x + radius,
                y_point + radius
            ],

            fill="red",

            outline="white",

            width=1
        )


    # --------------------------------------------------------
    # PREDICTION PANEL
    # --------------------------------------------------------

    panel_x = 20

    panel_y = 20

    panel_w = 250

    panel_h = 95


    draw.rounded_rectangle(

        [
            panel_x,
            panel_y,
            panel_x + panel_w,
            panel_y + panel_h
        ],

        radius=15,

        fill=(8, 15, 30),

        outline=(74, 222, 128),

        width=3
    )


    draw.text(

        (
            panel_x + 15,
            panel_y + 12
        ),

        f"SIGN: {prediction}",

        fill="white"
    )


    draw.text(

        (
            panel_x + 15,
            panel_y + 45
        ),

        f"CONFIDENCE: {confidence * 100:.1f}%",

        fill="lime"
    )


    return np.asarray(
        image
    )


# ============================================================
# REAL-TIME CALLBACK
# ============================================================

def video_frame_callback(frame):

    frame_array = frame.to_ndarray(
        format="rgb24"
    )


    with mp_hands.Hands(

        static_image_mode=False,

        max_num_hands=1,

        min_detection_confidence=0.60,

        min_tracking_confidence=0.60

    ) as detector:

        results = detector.process(
            frame_array
        )


        # ----------------------------------------------------
        # NO HAND
        # ----------------------------------------------------

        if not results.multi_hand_landmarks:

            with state_lock:

                shared["hand_detected"] = False

                shared["prediction"] = "-"

                shared["confidence"] = 0.0

                shared["stable_prediction"] = None

                shared["stable_count"] = 0

                shared["last_recorded"] = None


            return av.VideoFrame.from_ndarray(
                frame_array,
                format="rgb24"
            )


        # ----------------------------------------------------
        # HAND FOUND
        # ----------------------------------------------------

        landmarks = (
            results
            .multi_hand_landmarks[0]
            .landmark
        )


        features = create_features(
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


        prediction = str(
            model.classes_[best_index]
        )


        confidence = float(
            probabilities[best_index]
        )


        current_time = time.time()


        with state_lock:

            shared["hand_detected"] = True

            shared["prediction"] = prediction

            shared["confidence"] = confidence


            # ------------------------------------------------
            # STABILITY DETECTION
            # ------------------------------------------------

            if (
                shared["stable_prediction"]
                ==
                prediction
            ):

                shared["stable_count"] += 1

            else:

                shared["stable_prediction"] = prediction

                shared["stable_count"] = 1


            # ------------------------------------------------
            # RECORD LETTER
            # ------------------------------------------------

            if (

                confidence >= CONFIDENCE_THRESHOLD

                and

                shared["stable_count"]
                >= STABLE_FRAMES

                and

                shared["last_recorded"]
                != prediction

                and

                (
                    current_time
                    -
                    shared["last_record_time"]
                )
                >= RECORD_COOLDOWN

            ):

                shared["sentence"] += prediction

                shared["last_recorded"] = prediction

                shared["last_record_time"] = current_time


        # ----------------------------------------------------
        # DRAW VISUALIZATION
        # ----------------------------------------------------

        output = draw_overlay(

            frame_array,

            landmarks,

            prediction,

            confidence
        )


    return av.VideoFrame.from_ndarray(
        output,
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
            Intelligent Hand Sign Recognition
        </div>
        """,

        unsafe_allow_html=True
    )


    st.markdown("---")


    st.session_state.page = st.radio(

        "SYSTEM",

        [
            "Live Translator",
            "Dataset",
            "Model",
            "Computer Vision",
            "About"
        ],

        index=[
            "Live Translator",
            "Dataset",
            "Model",
            "Computer Vision",
            "About"
        ].index(
            st.session_state.page
        )
    )


    st.markdown("---")


    st.markdown(
        "**SYSTEM STATUS**"
    )


    st.success(
        "ONLINE"
    )


    st.caption(
        f"{len(classes)} classes available"
    )

    st.caption(
        f"{len(X)} hand samples extracted"
    )


# ============================================================
# LIVE TRANSLATOR
# ============================================================

if st.session_state.page == "Live Translator":

    st.markdown(
        """
        <div class="eyebrow">
            REAL-TIME COMPUTER VISION
        </div>

        <div class="hero-title">
            SIGNOVA
        </div>

        <div class="hero-subtitle">
            Translate hand signs into letters and numbers
            using real-time hand landmark detection.
        </div>
        """,

        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # SENTENCE
    # --------------------------------------------------------

    with state_lock:

        sentence = shared["sentence"]

        prediction = shared["prediction"]

        confidence = shared["confidence"]

        hand_detected = shared["hand_detected"]


    if sentence:

        sentence_html = sentence

        sentence_class = "sentence"

    else:

        sentence_html = (
            "Your detected signs will appear here..."
        )

        sentence_class = "sentence empty"


    st.markdown(

        f"""
        <div class="sentence-box">

            <div class="sentence-label">
                DETECTED SENTENCE / SEQUENCE
            </div>

            <div class="{sentence_class}">
                {sentence_html}
            </div>

        </div>
        """,

        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # BUTTONS
    # --------------------------------------------------------

    b1, b2, b3 = st.columns(
        [1, 1, 4]
    )


    with b1:

        if st.button(
            "🗑 Clear",
            use_container_width=True
        ):

            with state_lock:

                shared["sentence"] = ""

                shared["last_recorded"] = None

                shared["stable_prediction"] = None

                shared["stable_count"] = 0


            st.rerun()


    with b2:

        if st.button(
            "⌫ Delete",
            use_container_width=True
        ):

            with state_lock:

                if shared["sentence"]:

                    shared["sentence"] = (
                        shared["sentence"][:-1]
                    )

                    shared["last_recorded"] = None


            st.rerun()


    st.markdown("<br>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # CAMERA + PREDICTION
    # --------------------------------------------------------

    camera_col, result_col = st.columns(
        [1.55, 1]
    )


    with camera_col:

        st.markdown(
            """
            <div class="card">

            <strong>LIVE HAND TRACKING</strong>

            <br><br>

            🔴 <strong>Red points</strong> —
            21 hand landmarks

            <br>

            ━ <strong>White lines</strong> —
            hand vectors / edges

            <br>

            🟩 <strong>Green frame</strong> —
            detected hand region

            </div>
            """,

            unsafe_allow_html=True
        )


        st.markdown("<br>", unsafe_allow_html=True)


        ctx = webrtc_streamer(

            key="signova-camera",

            mode=WebRtcMode.SENDRECV,

            video_frame_callback=
                video_frame_callback,

            media_stream_constraints={
                "video": True,
                "audio": False
            },

            async_processing=True
        )


    with result_col:

        if hand_detected:

            status = (
                "🟢 HAND DETECTED"
            )

            status_class = (
                "status-online"
            )

        else:

            status = (
                "○ WAITING FOR HAND"
            )

            status_class = (
                "status-offline"
            )


        st.markdown(

            f"""
            <div class="card">

                <div class="{status_class}">
                    {status}
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
                    CURRENT PREDICTION
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

            <strong>How recording works</strong>

            <br><br>

            Hold a sign steady until SIGNOVA confirms
            the prediction.

            <br><br>

            The same sign is recorded only once.

            <br><br>

            To enter the same sign again, briefly
            remove your hand from the camera and
            show the sign again.

            <br><br>

            Example:

            <br>

            <strong>
            H → E → L → L → O
            </strong>

            <br><br>

            becomes:

            <br>

            <strong>
            HELLO
            </strong>

            </div>
            """,

            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # LIVE UPDATE LOOP
    # --------------------------------------------------------

    if ctx.state.playing:

        placeholder = st.empty()

        while ctx.state.playing:

            with state_lock:

                live_sentence = (
                    shared["sentence"]
                )

                live_prediction = (
                    shared["prediction"]
                )

                live_confidence = (
                    shared["confidence"]
                )

                live_hand = (
                    shared["hand_detected"]
                )


            if live_hand:

                status_text = (
                    "🟢 HAND DETECTED"
                )

            else:

                status_text = (
                    "○ WAITING FOR HAND"
                )


            placeholder.markdown(

                f"""
                <div class="card">

                    <strong>
                        {status_text}
                    </strong>

                    <br><br>

                    Current sign:
                    <strong>
                        {live_prediction}
                    </strong>

                    &nbsp;&nbsp;

                    Confidence:
                    <strong>
                        {live_confidence * 100:.1f}%
                    </strong>

                    <br><br>

                    Sentence:
                    <strong>
                        {live_sentence if live_sentence else "—"}
                    </strong>

                </div>
                """,

                unsafe_allow_html=True
            )


            time.sleep(0.15)


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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

            """
            <div class="metric">

                <div class="metric-value">
                    21
                </div>

                <div class="metric-label">
                    HAND LANDMARKS
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
                    {len(X)}
                </div>

                <div class="metric-label">
                    DATASET SAMPLES
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
                    {accuracy * 100:.1f}%
                </div>

                <div class="metric-label">
                    TEST ACCURACY
                </div>

            </div>
            """,

            unsafe_allow_html=True
        )


# ============================================================
# DATASET PAGE
# ============================================================

elif st.session_state.page == "Dataset":

    st.markdown(
        """
        <div class="eyebrow">
            DATASET ANALYSIS
        </div>

        <div class="hero-title">
            SIGNOVA Dataset
        </div>

        <div class="hero-subtitle">
            The supplied dataset contains hand-sign images
            representing digits and the English alphabet.
        </div>
        """,

        unsafe_allow_html=True
    )


    a, b, c = st.columns(3)


    with a:
        st.metric(
            "Classes",
            len(classes)
        )


    with b:
        st.metric(
            "Images",
            900
        )


    with c:
        st.metric(
            "Images / Class",
            25
        )


    st.markdown("<br>", unsafe_allow_html=True)


    st.subheader(
        "Digit Classes"
    )


    digit_cols = st.columns(10)


    for i in range(10):

        label = str(i)

        with digit_cols[i]:

            st.markdown(
                f"""
                <div class="metric">

                    <div class="metric-value">
                        {label}
                    </div>

                    <div class="metric-label">
                        DIGIT
                    </div>

                </div>
                """,

                unsafe_allow_html=True
            )


    st.markdown("<br>")


    st.subheader(
        "Alphabet Classes"
    )


    alphabet_cols = st.columns(6)


    for i in range(26):

        label = chr(
            ord("A") + i
        )

        with alphabet_cols[i % 6]:

            st.markdown(
                f"""
                <div class="metric">

                    <div class="metric-value">
                        {label}
                    </div>

                    <div class="metric-label">
                        LETTER
                    </div>

                </div>
                """,

                unsafe_allow_html=True
            )


# ============================================================
# MODEL PAGE
# ============================================================

elif st.session_state.page == "Model":

    st.markdown(
        """
        <div class="eyebrow">
            MACHINE LEARNING
        </div>

        <div class="hero-title">
            Recognition Model
        </div>

        <div class="hero-subtitle">
            SIGNOVA represents each hand using normalized
            landmark coordinates and hand vectors.
        </div>
        """,

        unsafe_allow_html=True
    )


    a, b, c = st.columns(3)


    with a:

        st.metric(
            "Classifier",
            "KNN"
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

        <strong>Feature extraction</strong>

        <br><br>

        Every detected hand is represented using
        21 landmark points.

        <br><br>

        Each landmark contains:

        <br>

        • X coordinate<br>
        • Y coordinate<br>
        • Z coordinate

        <br>

        SIGNOVA then calculates vectors between
        connected hand landmarks.

        <br><br>

        The coordinates are translated relative
        to the wrist and scale-normalised.

        <br><br>

        This produces a feature representation
        that is less dependent on where the hand
        appears in the camera.

        </div>
        """,

        unsafe_allow_html=True
    )


    st.markdown("<br>", unsafe_allow_html=True)


    st.subheader(
        "Recognition Pipeline"
    )


    pipeline = [

        "Webcam Frame",

        "Hand Detection",

        "21 Landmark Extraction",

        "Coordinate Normalisation",

        "Vector / Edge Extraction",

        "KNN Classification",

        "Confidence Evaluation",

        "Stable Sign Verification",

        "Sentence Construction"
    ]


    for number, step in enumerate(
        pipeline,
        1
    ):

        st.write(
            f"**{number}.** {step}"
        )


# ============================================================
# COMPUTER VISION PAGE
# ============================================================

elif st.session_state.page == "Computer Vision":

    st.markdown(
        """
        <div class="eyebrow">
            IMAGE PROCESSING
        </div>

        <div class="hero-title">
            Computer Vision
        </div>

        <div class="hero-subtitle">
            Visual features used by SIGNOVA to
            understand hand structure.
        </div>
        """,

        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="info">

        <strong>🔴 RED POINTS — LANDMARKS</strong>

        <br><br>

        SIGNOVA detects 21 important locations
        on the hand.

        <br><br>

        These points represent the wrist,
        finger joints and fingertips.

        <br><br>

        <strong>━ WHITE LINES — VECTORS</strong>

        <br><br>

        Lines connect the landmarks and represent
        the structure and direction of the fingers.

        <br><br>

        <strong>🟩 GREEN FRAME — HAND DETECTION</strong>

        <br><br>

        The green frame appears around the detected
        hand.

        <br><br>

        <strong>MODEL COMPARISON</strong>

        <br><br>

        The camera hand is converted into numerical
        features and compared with the landmark
        features extracted from the supplied
        900-image dataset.

        </div>
        """,

        unsafe_allow_html=True
    )


    st.markdown("<br>")


    st.subheader(
        "21 Landmark Structure"
    )


    landmarks = [

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


    columns = st.columns(3)


    for i, item in enumerate(
        landmarks
    ):

        with columns[i % 3]:

            st.write(
                f"🔴 {item}"
            )


# ============================================================
# ABOUT
# ============================================================

elif st.session_state.page == "About":

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

        SIGNOVA is an Image Processing and Computer
        Vision system designed to recognize static
        hand signs through a webcam.

        <br><br>

        The system uses the supplied dataset containing
        36 classes:

        <br><br>

        <strong>
        0–9 and A–Z
        </strong>

        <br><br>

        For each image, SIGNOVA detects the hand
        and extracts 21 landmarks.

        <br><br>

        The landmarks are normalized and combined
        with hand vectors to form a feature
        representation.

        <br><br>

        A K-Nearest Neighbors classifier is then
        used to identify the closest hand-sign
        patterns.

        <br><br>

        In real time, the recognized signs can be
        accumulated into a sentence or sequence.

        </div>
        """,

        unsafe_allow_html=True
    )
