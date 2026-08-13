import os
import zipfile
import time
import threading

import av
import numpy as np
import streamlit as st

from PIL import Image, ImageEnhance, ImageFilter

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

IMAGE_SIZE = (48, 48)

RANDOM_STATE = 42


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
# IMAGE PROCESSING
# ============================================================

def preprocess_image(image):

    gray = image.convert("L")

    gray = gray.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    gray = gray.filter(
        ImageFilter.GaussianBlur(
            radius=0.35
        )
    )

    gray = ImageEnhance.Contrast(
        gray
    ).enhance(1.15)

    array = np.asarray(
        gray,
        dtype=np.float32
    )

    array = array / 255.0

    return array


def extract_features(image):

    gray = preprocess_image(
        image
    )

    horizontal = np.diff(
        gray,
        axis=1,
        append=gray[:, -1:]
    )

    vertical = np.diff(
        gray,
        axis=0,
        append=gray[-1:, :]
    )

    magnitude = np.sqrt(
        horizontal ** 2 +
        vertical ** 2
    )

    magnitude = np.clip(
        magnitude,
        0,
        1
    )

    features = np.concatenate(
        [
            gray.flatten(),
            magnitude.flatten()
        ]
    )

    return features


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_dataset(dataset_path):

    X = []
    y = []

    classes = []

    for folder in os.listdir(
        dataset_path
    ):

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

        classes.append(
            folder
        )

        for filename in images:

            path = os.path.join(
                folder_path,
                filename
            )

            try:

                image = Image.open(
                    path
                ).convert("RGB")

                features = extract_features(
                    image
                )

                X.append(
                    features
                )

                y.append(
                    folder
                )

            except Exception:
                continue

    return (
        np.asarray(X),
        np.asarray(y),
        sorted(classes)
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
    "SIGNOVA is preparing the recognition engine..."
):

    dataset_path = extract_dataset()

    X, y, classes = load_dataset(
        dataset_path
    )

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

        self.stable_sign = None

        self.stable_count = 0

        self.last_added = None

        self.last_add_time = 0

        self.sentence = ""


sign_state = SignState()


# ============================================================
# REAL-TIME CLASSIFIER
# ============================================================

def predict_frame(frame):

    image = frame.to_image()

    features = extract_features(
        image
    )

    features = features.reshape(
        1,
        -1
    )

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

    return (
        prediction,
        confidence
    )


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class SignovaVideoProcessor:

    def recv(self, frame):

        image = frame.to_image()

        try:

            prediction, confidence = predict_frame(
                frame
            )

            current_time = time.time()

            with sign_state.lock:

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

        except Exception:

            pass


        # --------------------------------------------------------
        # DRAW SIMPLE STATUS ON CAMERA FRAME
        # --------------------------------------------------------

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
        "Recognition engine ready"
    )

    st.caption(
        f"{len(classes)} classes"
    )

    st.caption(
        f"{len(y)} images"
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
                sentence using your camera.
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


    sentence_display = (
        current_sentence
        if current_sentence
        else "Your sentence will appear here..."
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

        st.markdown(
            """
            <div class="camera-card">

                <div class="status">
                    ● LIVE CAMERA
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
            2. Show one hand sign clearly.<br>
            3. Keep the sign steady for a moment.<br>
            4. SIGNOVA records the letter automatically.<br>
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
                    Dataset Images
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
            """
            <div class="metric">

                <div class="metric-number">
                    SVM
                </div>

                <div class="metric-label">
                    Classifier
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
            "Images",
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
        f"{len(files)} images"
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
                hand-sign classifier.
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
        The model learns from the training subset and is
        evaluated against images that were not used during
        training.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.subheader(
        "Classification Method"
    )

    st.write(
        """
        **Support Vector Machine (SVM)**

        SIGNOVA uses an RBF-kernel SVM to classify the
        extracted image features into the available hand-sign
        categories.
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
            "Grayscale",
            "Each frame is converted from RGB into grayscale."
        ),

        (
            "03",
            "Resize",
            "The image is standardized to 48 × 48 pixels."
        ),

        (
            "04",
            "Normalization",
            "Pixel intensity values are converted from 0–255 into 0–1."
        ),

        (
            "05",
            "Feature Extraction",
            "SIGNOVA combines normalized pixel information with horizontal and vertical edge information."
        ),

        (
            "06",
            "SVM",
            "The extracted features are passed to the trained Support Vector Machine."
        ),

        (
            "07",
            "Confidence",
            "The classifier estimates the probability of each available sign."
        ),

        (
            "08",
            "Stability Check",
            "The same prediction must remain stable before it is accepted."
        ),

        (
            "09",
            "Sentence Builder",
            "The accepted sign is appended to the sentence exactly once."
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

        SIGNOVA demonstrates how image processing and
        machine learning can be combined to recognize
        static hand signs from a live camera.

        Instead of simply displaying one prediction,
        SIGNOVA uses a sentence builder that records
        stable predictions one at a time.

        <br><br>

        <strong>Technology:</strong>

        <br><br>

        Python • Streamlit • Streamlit-WebRTC • NumPy •
        Pillow • Scikit-learn • SVM

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
