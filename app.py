import os
import time
import threading
import zipfile
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
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ============================================================
# SIGNOVA CONFIG
# ============================================================
st.set_page_config(
    page_title="SIGNOVA",
    page_icon="🤟",
    layout="wide"
)

ZIP_NAMES = [
    "archive.zip",
    "archive(1).zip"
]

EXTRACT_DIR = "signova_dataset"

CLASSES = (
    [str(i) for i in range(10)]
    +
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
)

IMAGES_PER_CLASS = 25
TOTAL_IMAGES = 900

CONFIDENCE_THRESHOLD = 0.65
STABLE_FRAMES = 8
RECORD_COOLDOWN = 0.70
NO_HAND_REARM_FRAMES = 4
TRAJECTORY_LENGTH = 18

mp_hands = mp.solutions.hands


# ============================================================
# HAND CONNECTIONS
# ============================================================

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
# UI
# ============================================================

st.markdown(
    """
<style>

.stApp {

    background:
        radial-gradient(
            circle at 10% 5%,
            rgba(124,58,237,.18),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(14,165,233,.12),
            transparent 28%
        ),
        #070a12;

    color: #f8fafc;
}


[data-testid="stHeader"] {

    background: transparent;
}


[data-testid="stSidebar"] {

    background: #0b0f1a;

    border-right:
        1px solid rgba(255,255,255,.07);
}


#MainMenu,
footer {

    visibility: hidden;
}


.brand {

    font-size: 28px;

    font-weight: 900;
}


.muted {

    color: #64748b;

    font-size: 12px;
}


.eyebrow {

    color: #818cf8;

    font-size: 11px;

    font-weight: 900;

    letter-spacing: 2px;

    text-transform: uppercase;
}


.hero {

    font-size: 52px;

    font-weight: 950;

    letter-spacing: -2px;

    background:
        linear-gradient(
            90deg,
            #fff,
            #c4b5fd,
            #7dd3fc
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}


.subtitle {

    color: #94a3b8;

    line-height: 1.7;

    max-width: 820px;
}


.card {

    padding: 18px;

    border-radius: 18px;

    background:
        rgba(15,23,42,.78);

    border:
        1px solid rgba(255,255,255,.07);
}


.sentence {

    margin: 18px 0;

    padding: 22px 25px;

    min-height: 105px;

    border-radius: 20px;

    background:
        linear-gradient(
            145deg,
            rgba(124,58,237,.16),
            rgba(37,99,235,.08)
        );

    border:
        1px solid rgba(139,92,246,.30);
}


.slabel {

    color: #818cf8;

    font-size: 10px;

    font-weight: 900;

    letter-spacing: 2px;
}


.stext {

    font-size: 36px;

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


.good {

    color: #4ade80;

    font-weight: 900;
}


.wait {

    color: #64748b;

    font-weight: 900;
}


.info {

    background:
        rgba(99,102,241,.07);

    border-left:
        3px solid #6366f1;

    border-radius: 9px;

    padding: 17px;

    color: #cbd5e1;

    line-height: 1.8;

    font-size: 13px;
}


.pred {

    padding: 24px;

    text-align: center;

    border-radius: 20px;

    background:
        rgba(15,23,42,.82);

    border:
        1px solid rgba(255,255,255,.08);
}


.pletter {

    font-size: 96px;

    font-weight: 950;

    background:
        linear-gradient(
            135deg,
            #c4b5fd,
            #60a5fa
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}


.metricbox {

    padding: 16px;

    border-radius: 16px;

    background:
        rgba(15,23,42,.76);

    border:
        1px solid rgba(255,255,255,.07);
}


.mvalue {

    font-size: 27px;

    font-weight: 900;
}


.mlabel {

    color: #64748b;

    font-size: 10px;

    font-weight: 800;

    text-transform: uppercase;

    letter-spacing: 1px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# UI HELPERS
# ============================================================

def hero(
    eyebrow,
    title,
    subtitle
):

    st.markdown(
        f"""
        <div class="eyebrow">
            {eyebrow}
        </div>

        <div class="hero">
            {title}
        </div>

        <div class="subtitle">
            {subtitle}
        </div>
        """,
        unsafe_allow_html=True
    )


def metric_card(
    column,
    value,
    label
):

    with column:

        st.markdown(
            f"""
            <div class="metricbox">

                <div class="mvalue">
                    {value}
                </div>

                <div class="mlabel">
                    {label}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DATASET HELPERS
# ============================================================

def find_zip():

    for name in ZIP_NAMES:

        if os.path.exists(name):

            return name

    return None


def locate_dataset():

    if not os.path.exists(
        EXTRACT_DIR
    ):

        return None


    direct = os.path.join(
        EXTRACT_DIR,
        "DATASET"
    )


    if os.path.isdir(
        direct
    ):

        return direct


    for root, _, _ in os.walk(
        EXTRACT_DIR
    ):

        if os.path.basename(
            root
        ).upper() == "DATASET":

            return root


    return None


def prepare_dataset():

    existing = locate_dataset()


    if existing:

        return existing


    zip_name = find_zip()


    if not zip_name:

        st.error(
            """
            Put archive.zip in the same
            GitHub repository as app.py.
            """
        )

        st.stop()


    os.makedirs(
        EXTRACT_DIR,
        exist_ok=True
    )


    try:

        with zipfile.ZipFile(
            zip_name,
            "r"
        ) as archive:

            archive.extractall(
                EXTRACT_DIR
            )


    except zipfile.BadZipFile:

        st.error(
            "The dataset ZIP is invalid or corrupted."
        )

        st.stop()


    dataset = locate_dataset()


    if not dataset:

        st.error(
            """
            Could not find DATASET/
            inside archive.zip.
            """
        )

        st.stop()


    return dataset


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def features_from_landmarks(
    landmarks
):

    points = np.asarray(
        [
            [
                p.x,
                p.y,
                p.z
            ]
            for p in landmarks
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # WRIST RELATIVE
    # --------------------------------------------------------

    points -= points[
        0
    ].copy()


    # --------------------------------------------------------
    # SCALE NORMALIZATION
    # --------------------------------------------------------

    scale = float(
        np.max(
            np.linalg.norm(
                points[:, :2],
                axis=1
            )
        )
    )


    if scale > 1e-6:

        points /= scale


    # --------------------------------------------------------
    # MIRROR NORMALIZATION
    # --------------------------------------------------------

    if points[5, 0] > points[17, 0]:

        points[:, 0] *= -1


    # --------------------------------------------------------
    # VECTOR FEATURES
    # --------------------------------------------------------

    vectors = []


    for a, b in HAND_CONNECTIONS:

        vector = (
            points[b]
            -
            points[a]
        )

        vectors.extend(
            vector.tolist()
        )


    # --------------------------------------------------------
    # DISTANCE FEATURES
    # --------------------------------------------------------

    fingertips = [
        4,
        8,
        12,
        16,
        20
    ]


    distances = []


    for tip in fingertips:

        distance = np.linalg.norm(
            points[
                tip,
                :2
            ]
        )

        distances.append(
            float(distance)
        )


    for i in range(4):

        first = points[
            fingertips[i],
            :2
        ]

        second = points[
            fingertips[i + 1],
            :2
        ]


        distance = np.linalg.norm(
            first
            -
            second
        )


        distances.append(
            float(distance)
        )


    # --------------------------------------------------------
    # FINAL FEATURE VECTOR
    # --------------------------------------------------------

    return np.concatenate(
        [
            points.flatten(),

            np.asarray(
                vectors,
                dtype=np.float32
            ),

            np.asarray(
                distances,
                dtype=np.float32
            )
        ]
    ).astype(
        np.float32
    )


# ============================================================
# EXTRACT LANDMARKS FROM DATASET
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_dataset_landmarks(
    dataset_path
):

    X = []
    y = []


    detected = {

        label: 0

        for label in CLASSES
    }


    failed = {

        label: 0

        for label in CLASSES
    }


    with mp_hands.Hands(

        static_image_mode=True,

        max_num_hands=1,

        model_complexity=1,

        min_detection_confidence=0.35

    ) as detector:


        for label in CLASSES:

            folder = os.path.join(
                dataset_path,
                label
            )


            if not os.path.isdir(
                folder
            ):

                continue


            files = sorted(
                [
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
            )


            for filename in files:

                try:

                    image = Image.open(
                        os.path.join(
                            folder,
                            filename
                        )
                    ).convert(
                        "RGB"
                    )


                    result = detector.process(
                        np.asarray(
                            image
                        )
                    )


                    if not result.multi_hand_landmarks:

                        failed[
                            label
                        ] += 1

                        continue


                    landmarks = (
                        result
                        .multi_hand_landmarks[0]
                        .landmark
                    )


                    X.append(
                        features_from_landmarks(
                            landmarks
                        )
                    )


                    y.append(
                        label
                    )


                    detected[
                        label
                    ] += 1


                except Exception:

                    failed[
                        label
                    ] += 1


    return (

        np.asarray(
            X,
            dtype=np.float32
        ),

        np.asarray(
            y
        ),

        detected,

        failed
    )


# ============================================================
# SAFE TRAINING
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def train_classifier(
    X,
    y
):

    X = np.asarray(
        X,
        dtype=np.float32
    )


    y = np.asarray(
        y
    )


    if len(X) == 0:

        raise ValueError(
            "No usable landmark samples were extracted."
        )


    # ========================================================
    # LIVE MODEL
    #
    # Train using ALL usable samples.
    # ========================================================

    live_model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "knn",

                KNeighborsClassifier(

                    n_neighbors=max(
                        1,
                        min(
                            5,
                            len(X)
                        )
                    ),

                    weights="distance"
                )
            )
        ]
    )


    live_model.fit(
        X,
        y
    )


    # ========================================================
    # SAFE EVALUATION
    # ========================================================

    unique, counts = np.unique(
        y,
        return_counts=True
    )


    eligible = unique[
        counts >= 2
    ]


    if len(
        eligible
    ) < 2:

        return (

            live_model,

            None,

            len(X),

            0,

            eligible.tolist()
        )


    mask = np.isin(
        y,
        eligible
    )


    Xe = X[
        mask
    ]


    ye = y[
        mask
    ]


    rng = np.random.default_rng(
        42
    )


    train_idx = []

    test_idx = []


    # --------------------------------------------------------
    # MANUAL BALANCED SPLIT
    # --------------------------------------------------------

    for label in eligible:

        idx = np.where(
            ye == label
        )[0]


        rng.shuffle(
            idx
        )


        n_test = max(
            1,
            int(
                round(
                    len(idx)
                    *
                    0.20
                )
            )
        )


        n_test = min(
            n_test,
            len(idx) - 1
        )


        test_idx.extend(
            idx[
                :n_test
            ]
        )


        train_idx.extend(
            idx[
                n_test:
            ]
        )


    train_idx = np.asarray(
        train_idx,
        dtype=int
    )


    test_idx = np.asarray(
        test_idx,
        dtype=int
    )


    if (
        not len(train_idx)
        or
        not len(test_idx)
    ):

        return (

            live_model,

            None,

            len(X),

            0,

            eligible.tolist()
        )


    X_train = Xe[
        train_idx
    ]

    y_train = ye[
        train_idx
    ]


    X_test = Xe[
        test_idx
    ]

    y_test = ye[
        test_idx
    ]


    eval_model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "knn",

                KNeighborsClassifier(

                    n_neighbors=max(
                        1,
                        min(
                            5,
                            len(X_train)
                        )
                    ),

                    weights="distance"
                )
            )
        ]
    )


    eval_model.fit(
        X_train,
        y_train
    )


    predictions = eval_model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    return (

        live_model,

        float(
            accuracy
        ),

        len(
            X_train
        ),

        len(
            X_test
        ),

        eligible.tolist()
    )


# ============================================================
# INITIALIZE DATA
# ============================================================

dataset_path = prepare_dataset()


with st.spinner(
    "Extracting 21-point hand landmarks from the dataset..."
):

    (
        X,
        y,
        detected_counts,
        failed_counts
    ) = load_dataset_landmarks(
        dataset_path
    )


if len(
    X
) == 0:

    st.error(
        """
        MediaPipe could not extract
        any usable hand landmarks
        from the dataset.
        """
    )

    st.stop()


with st.spinner(
    "Training SIGNOVA..."
):

    (
        model,
        accuracy,
        eval_train_count,
        eval_test_count,
        eval_classes
    ) = train_classifier(
        X,
        y
    )


# ============================================================
# DATASET STATISTICS
# ============================================================

if accuracy is None:

    accuracy_text = "N/A"

else:

    accuracy_text = (
        f"{accuracy * 100:.1f}%"
    )


usable_samples = len(
    X
)


extraction_rate = (
    usable_samples
    /
    TOTAL_IMAGES
    *
    100
)


available_classes = sorted(
    np.unique(
        y
    ).tolist()
)


missing_classes = [

    label

    for label in CLASSES

    if label
    not in available_classes
]


# ============================================================
# SHARED CAMERA STATE
# ============================================================

class RecognitionState:

    def __init__(self):

        self.lock = (
            threading.Lock()
        )

        self.hand_detected = False

        self.prediction = "-"

        self.confidence = 0.0

        self.sentence = ""

        self.stable_prediction = None

        self.stable_count = 0

        self.last_recorded = None

        self.last_record_time = 0.0

        self.no_hand_count = 0


if "recognition_state" not in st.session_state:

    st.session_state.recognition_state = (
        RecognitionState()
    )


state = (
    st.session_state
    .recognition_state
)


# ============================================================
# CAMERA OVERLAY
# ============================================================

def draw_overlay(

    frame,

    landmarks,

    prediction,

    confidence,

    trajectory
):

    image = Image.fromarray(
        frame
    ).convert(
        "RGB"
    )


    draw = ImageDraw.Draw(
        image
    )


    width, height = (
        image.size
    )


    points = [

        (
            int(
                p.x
                *
                width
            ),

            int(
                p.y
                *
                height
            )
        )

        for p in landmarks
    ]


    xs = [

        p[0]

        for p in points
    ]


    ys = [

        p[1]

        for p in points
    ]


    # ========================================================
    # GREEN HAND FRAME
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
        width - 1
    )


    bottom = min(
        max(ys) + 25,
        height - 1
    )


    draw.rectangle(

        [
            left,
            top,
            right,
            bottom
        ],

        outline=(
            0,
            255,
            80
        ),

        width=5
    )


    # ========================================================
    # HAND DETECTED TAG
    # ========================================================

    tag_top = max(
        0,
        top - 28
    )


    draw.rectangle(

        [
            left,

            tag_top,

            min(
                left + 160,
                width - 1
            ),

            top
        ],

        fill=(
            0,
            170,
            70
        )
    )


    draw.text(

        (
            left + 8,
            tag_top + 6
        ),

        "HAND DETECTED",

        fill="white"
    )


    # ========================================================
    # WHITE LINES / VECTORS
    # ========================================================

    for a, b in HAND_CONNECTIONS:

        draw.line(

            [
                points[a],
                points[b]
            ],

            fill=(
                255,
                255,
                255
            ),

            width=3
        )


    # ========================================================
    # RED LANDMARKS
    # ========================================================

    for x, y_point in points:

        radius = 6


        draw.ellipse(

            [
                x - radius,
                y_point - radius,

                x + radius,
                y_point + radius
            ],

            fill=(
                255,
                25,
                25
            ),

            outline="white",

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


    radius = 9


    draw.ellipse(

        [
            centroid_x - radius,
            centroid_y - radius,

            centroid_x + radius,
            centroid_y + radius
        ],

        fill=(
            255,
            0,
            0
        ),

        outline=(
            255,
            255,
            0
        ),

        width=3
    )


    # ========================================================
    # INDEX FINGER TRAJECTORY
    # ========================================================

    history = list(
        trajectory
    )


    for i in range(
        1,
        len(history)
    ):

        draw.line(

            [
                history[
                    i - 1
                ],

                history[
                    i
                ]
            ],

            fill=(
                0,
                220,
                255
            ),

            width=3
        )


    # ========================================================
    # PREDICTION PANEL
    # ========================================================

    draw.rounded_rectangle(

        [
            15,
            15,
            245,
            108
        ],

        radius=14,

        fill=(
            8,
            15,
            30
        ),

        outline=(
            0,
            255,
            80
        ),

        width=3
    )


    draw.text(

        (
            30,
            31
        ),

        f"SIGN: {prediction}",

        fill="white"
    )


    draw.text(

        (
            30,
            62
        ),

        (
            f"CONFIDENCE: "
            f"{confidence * 100:.1f}%"
        ),

        fill=(
            100,
            255,
            140
        )
    )


    return np.asarray(
        image
    )


# ============================================================
# WEBRTC VIDEO PROCESSOR
# ============================================================

class SignovaProcessor:

    def __init__(

        self,

        classifier,

        shared_state
    ):

        self.model = classifier

        self.state = shared_state

        self.lock = (
            threading.Lock()
        )


        self.trajectory = deque(
            maxlen=TRAJECTORY_LENGTH
        )


        # Create MediaPipe detector ONCE.

        self.hands = mp_hands.Hands(

            static_image_mode=False,

            max_num_hands=1,

            model_complexity=1,

            min_detection_confidence=0.60,

            min_tracking_confidence=0.60
        )


    def recv(
        self,
        frame
    ):

        with self.lock:

            rgb = frame.to_ndarray(
                format="rgb24"
            )


            result = self.hands.process(
                rgb
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

                    self.state.no_hand_count += 1


                    if (
                        self.state.no_hand_count
                        >=
                        NO_HAND_REARM_FRAMES
                    ):

                        self.state.last_recorded = None


                return av.VideoFrame.from_ndarray(

                    rgb,

                    format="rgb24"
                )


            # =================================================
            # HAND DETECTED
            # =================================================

            landmarks = (
                result
                .multi_hand_landmarks[0]
                .landmark
            )


            feature = features_from_landmarks(
                landmarks
            ).reshape(
                1,
                -1
            )


            probabilities = (
                self.model
                .predict_proba(
                    feature
                )[0]
            )


            best = int(
                np.argmax(
                    probabilities
                )
            )


            prediction = str(
                self.model
                .classes_[
                    best
                ]
            )


            confidence = float(
                probabilities[
                    best
                ]
            )


            # =================================================
            # TRAJECTORY
            # =================================================

            height, width, _ = (
                rgb.shape
            )


            tip = landmarks[
                8
            ]


            self.trajectory.append(

                (
                    int(
                        tip.x
                        *
                        width
                    ),

                    int(
                        tip.y
                        *
                        height
                    )
                )
            )


            now = time.time()


            # =================================================
            # STATE UPDATE
            # =================================================

            with self.state.lock:

                self.state.hand_detected = True

                self.state.no_hand_count = 0

                self.state.prediction = prediction

                self.state.confidence = confidence


                # ---------------------------------------------
                # STABILITY
                # ---------------------------------------------

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


                # ---------------------------------------------
                # RECORD LETTER
                # ---------------------------------------------

                should_record = (

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
                        now
                        -
                        self.state.last_record_time
                    )
                    >=
                    RECORD_COOLDOWN
                )


                if should_record:

                    self.state.sentence += (
                        prediction
                    )


                    self.state.last_recorded = (
                        prediction
                    )


                    self.state.last_record_time = (
                        now
                    )


            # =================================================
            # DRAW
            # =================================================

            output = draw_overlay(

                rgb,

                landmarks,

                prediction,

                confidence,

                self.trajectory
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
        <div class="brand">
            🤟 SIGNOVA
        </div>

        <div class="muted">
            Real-Time Hand Sign Recognition
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "---"
    )


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


    st.markdown(
        "---"
    )


    st.success(
        "Recognition engine online"
    )


    st.caption(
        f"{len(available_classes)} / 36 classes usable"
    )


    st.caption(
        f"{usable_samples} landmark samples"
    )


    st.caption(
        f"{extraction_rate:.1f}% extraction rate"
    )


# ============================================================
# LIVE REFRESH AREA
# ============================================================

@st.fragment(
    run_every=0.25
)
def live_panel():

    with state.lock:

        sentence = (
            state.sentence
        )

        prediction = (
            state.prediction
        )

        confidence = (
            state.confidence
        )

        detected = (
            state.hand_detected
        )

        stable = (
            state.stable_count
        )


    # ========================================================
    # SENTENCE
    # ========================================================

    if sentence:

        text = sentence

        css = "stext"


    else:

        text = (
            "Detected signs will appear here..."
        )

        css = (
            "stext empty"
        )


    st.markdown(

        f"""
        <div class="sentence">

            <div class="slabel">
                DETECTED SENTENCE / SEQUENCE
            </div>

            <div class="{css}">
                {text}
            </div>

        </div>
        """,

        unsafe_allow_html=True
    )


    # ========================================================
    # STATUS
    # ========================================================

    c1, c2 = st.columns(
        2
    )


    with c1:

        if detected:

            status = (
                "🟢 HAND DETECTED"
            )

            status_css = (
                "good"
            )


        else:

            status = (
                "○ WAITING FOR HAND"
            )

            status_css = (
                "wait"
            )


        st.markdown(

            f"""
            <div class="card">

                <span class="{status_css}">
                    {status}
                </span>

            </div>
            """,

            unsafe_allow_html=True
        )


    with c2:

        st.markdown(

            f"""
            <div class="card">

                Sign:
                <strong>
                    {prediction}
                </strong>

                &nbsp;

                Confidence:
                <strong>
                    {confidence * 100:.1f}%
                </strong>

                <br>

                Stability:
                <strong>
                    {min(stable, STABLE_FRAMES)}
                    /
                    {STABLE_FRAMES}
                </strong>

            </div>
            """,

            unsafe_allow_html=True
        )


# ============================================================
# LIVE TRANSLATOR PAGE
# ============================================================

if page == "🎥 Live Translator":

    hero(

        "Real-Time Computer Vision",

        "SIGNOVA",

        """
        Show a static hand sign. SIGNOVA detects 21 hand
        landmarks, builds landmark/vector features, compares
        them with your A-Z and 0-9 dataset, and appends
        stable predictions to the letter box.
        """
    )


    live_panel()


    # ========================================================
    # BUTTONS
    # ========================================================

    b1, b2, b3, _ = st.columns(
        [
            1,
            1,
            1,
            3
        ]
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


    with b3:

        if st.button(

            "␣ Space",

            use_container_width=True
        ):

            with state.lock:

                if (
                    state.sentence
                    and
                    not state.sentence.endswith(
                        " "
                    )
                ):

                    state.sentence += " "


                state.last_recorded = None


            st.rerun()


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # ========================================================
    # CAMERA AREA
    # ========================================================

    camera_col, guide_col = st.columns(
        [
            1.6,
            1
        ]
    )


    with camera_col:

        st.markdown(
            """
            <div class="card">

                <strong>
                    Live Overlay
                </strong>

                <br>

                🟩 Green = hand frame

                <br>

                🔴 Red = landmarks / centroid

                <br>

                ⚪ White = vectors

                <br>

                🔵 Cyan = fingertip trajectory

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        def processor_factory():

            return SignovaProcessor(

                model,

                state
            )


        webrtc_streamer(

            key="signova-camera",

            mode=WebRtcMode.SENDRECV,

            video_processor_factory=(
                processor_factory
            ),

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


    with guide_col:

        st.markdown(

            f"""
            <div class="info">

            <strong>
                How to type
            </strong>

            <br><br>

            Hold a sign steady until it reaches:

            <strong>
                {STABLE_FRAMES}/{STABLE_FRAMES}
            </strong>

            <br><br>

            A prediction needs at least:

            <strong>
                {CONFIDENCE_THRESHOLD * 100:.0f}%
            </strong>

            confidence.

            <br><br>

            Remove your hand briefly before repeating
            the same letter.

            <br><br>

            Example:

            <strong>
                H → E → L → remove hand → L → O
            </strong>

            <br><br>

            Result:

            <strong>
                HELLO
            </strong>

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


    columns = st.columns(
        5
    )


    values = [

        36,

        21,

        usable_samples,

        accuracy_text,

        "KNN"
    ]


    labels = [

        "Classes",

        "Landmarks",

        "Usable Samples",

        "Accuracy",

        "Classifier"
    ]


    for (
        column,
        value,
        label
    ) in zip(
        columns,
        values,
        labels
    ):

        metric_card(
            column,
            value,
            label
        )


# ============================================================
# DATASET LAB PAGE
# ============================================================

elif page == "📊 Dataset Lab":

    hero(

        "Dataset Analysis",

        "Dataset Lab",

        """
        See which of the 900 supplied images
        MediaPipe can successfully convert into
        21-point landmark samples.
        """
    )


    c1, c2, c3, c4 = st.columns(
        4
    )


    c1.metric(
        "Original Images",
        TOTAL_IMAGES
    )


    c2.metric(
        "Classes",
        36
    )


    c3.metric(
        "Usable Samples",
        usable_samples
    )


    c4.metric(
        "Extraction Rate",
        f"{extraction_rate:.1f}%"
    )


    # ========================================================
    # DIAGNOSTIC TABLE
    # ========================================================

    st.subheader(
        "Landmark Extraction Diagnostics"
    )


    rows = []


    for label in CLASSES:

        detected = int(
            detected_counts.get(
                label,
                0
            )
        )


        failed = int(
            failed_counts.get(
                label,
                0
            )
        )


        percent = (
            detected
            /
            IMAGES_PER_CLASS
            *
            100
        )


        rows.append(
            {
                "Sign":
                    label,

                "Original Images":
                    IMAGES_PER_CLASS,

                "Landmarks Detected":
                    detected,

                "Failed Detection":
                    failed,

                "Detection Rate":
                    f"{percent:.1f}%"
            }
        )


    st.dataframe(

        rows,

        use_container_width=True,

        hide_index=True
    )


    # ========================================================
    # MISSING CLASSES
    # ========================================================

    if missing_classes:

        st.warning(

            "No usable samples for: "
            +
            ", ".join(
                missing_classes
            )
        )


    else:

        st.success(
            """
            All 36 classes have at least
            one usable landmark sample.
            """
        )


    # ========================================================
    # IMAGE EXPLORER
    # ========================================================

    st.subheader(
        "Image Explorer"
    )


    selected = st.selectbox(

        "Choose a sign",

        CLASSES
    )


    folder = os.path.join(

        dataset_path,

        selected
    )


    if os.path.isdir(
        folder
    ):

        files = sorted(
            [
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
        )


        gallery = st.columns(
            5
        )


        for i, filename in enumerate(
            files[:20]
        ):

            try:

                with gallery[
                    i % 5
                ]:

                    st.image(

                        Image.open(
                            os.path.join(
                                folder,
                                filename
                            )
                        ),

                        use_container_width=True
                    )


            except Exception:

                pass


# ============================================================
# MODEL PAGE
# ============================================================

elif page == "🧠 Model Insights":

    hero(

        "Machine Learning",

        "Model Insights",

        """
        The live model uses every usable landmark sample.
        Accuracy uses a separate safe class-balanced
        evaluation split.
        """
    )


    a, b, c, d = st.columns(
        4
    )


    a.metric(
        "Classifier",
        "KNN"
    )


    b.metric(
        "Evaluation Training",
        eval_train_count
    )


    c.metric(
        "Evaluation Testing",
        eval_test_count
    )


    d.metric(
        "Accuracy",
        accuracy_text
    )


    st.markdown(
        """
        <div class="info">

        <strong>
            Features used by SIGNOVA
        </strong>

        <br><br>

        • 21 X/Y/Z hand landmarks

        <br>

        • Wrist-relative normalization

        <br>

        • Hand-size normalization

        <br>

        • Mirror normalization

        <br>

        • Joint-to-joint vectors

        <br>

        • Fingertip distances

        <br><br>

        The previous fixed stratified 20% split
        is not used, so classes with only a few
        detected landmark samples no longer
        crash the model training.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# COMPUTER VISION PAGE
# ============================================================

elif page == "👁 Computer Vision":

    hero(

        "Image Processing",

        "Computer Vision",

        """
        The live camera visualizes the features
        SIGNOVA uses to describe the hand.
        """
    )


    st.markdown(
        """
        <div class="info">

        <strong>
            🟩 Green Frame
        </strong>

        <br>

        Detected hand region.

        <br><br>

        <strong>
            🔴 Red Points
        </strong>

        <br>

        21 hand landmarks.

        <br><br>

        <strong>
            ⚪ White Lines
        </strong>

        <br>

        Connected joint vectors / skeleton.

        <br><br>

        <strong>
            🔴 / 🟡 Centroid
        </strong>

        <br>

        Approximate hand centre.

        <br><br>

        <strong>
            🔵 Cyan Trajectory
        </strong>

        <br>

        Recent movement of the index fingertip.

        <br><br>

        The current classifier recognizes static hand
        shape. The trajectory is visualized as a
        computer-vision feature but is not yet used
        to classify dynamic signs.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.subheader(
        "21 Hand Landmarks"
    )


    names = [

        "0 Wrist",

        "1 Thumb CMC",

        "2 Thumb MCP",

        "3 Thumb IP",

        "4 Thumb Tip",

        "5 Index MCP",

        "6 Index PIP",

        "7 Index DIP",

        "8 Index Tip",

        "9 Middle MCP",

        "10 Middle PIP",

        "11 Middle DIP",

        "12 Middle Tip",

        "13 Ring MCP",

        "14 Ring PIP",

        "15 Ring DIP",

        "16 Ring Tip",

        "17 Pinky MCP",

        "18 Pinky PIP",

        "19 Pinky DIP",

        "20 Pinky Tip"
    ]


    cols = st.columns(
        3
    )


    for i, name in enumerate(
        names
    ):

        with cols[
            i % 3
        ]:

            st.write(
                "🔴 " + name
            )


# ============================================================
# ABOUT PAGE
# ============================================================

else:

    hero(

        "Image Processing & Computer Vision",

        "About SIGNOVA",

        """
        A real-time static hand-sign recognition
        and sequence construction system built around
        your 900-image, 36-class dataset.
        """
    )


    st.markdown(
        """
        <div class="info">

        SIGNOVA recognizes static A-Z and 0-9
        hand signs.

        <br><br>

        It extracts MediaPipe hand landmarks from
        the supplied dataset, trains a KNN model
        on normalized landmark/vector features,
        then applies the same feature extraction
        to live webcam frames.

        <br><br>

        Stable predictions are recorded into the
        letter box.

        <br><br>

        <strong>
            Technology
        </strong>

        <br><br>

        Python

        <br>

        Streamlit

        <br>

        Streamlit-WebRTC

        <br>

        MediaPipe

        <br>

        PyAV

        <br>

        NumPy

        <br>

        Pillow

        <br>

        Scikit-learn

        </div>
        """,
        unsafe_allow_html=True
    )
