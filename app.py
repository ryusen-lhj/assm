import os
import zipfile
import hashlib

import numpy as np
import streamlit as st

from PIL import Image, ImageEnhance, ImageFilter

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# ============================================================
# SIGNOVA
# Hand Sign Recognition System
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

st.markdown("""
<style>

    /* -------------------------------------------------------
       GLOBAL
    ------------------------------------------------------- */

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


    /* -------------------------------------------------------
       SIDEBAR
    ------------------------------------------------------- */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0d1220 0%,
                #090d17 100%
            );

        border-right: 1px solid rgba(255,255,255,0.06);
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }


    /* -------------------------------------------------------
       BRAND
    ------------------------------------------------------- */

    .brand {
        padding: 12px 0 28px 0;
    }

    .brand-mark {
        width: 48px;
        height: 48px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 14px;

        background:
            linear-gradient(
                135deg,
                #7c3aed,
                #2563eb
            );

        font-size: 25px;

        box-shadow:
            0 10px 30px rgba(37,99,235,0.25);
    }

    .brand-name {
        font-size: 25px;
        font-weight: 850;
        letter-spacing: 1px;
        margin-top: 12px;
    }

    .brand-description {
        color: #64748b;
        font-size: 12px;
        line-height: 1.6;
    }


    /* -------------------------------------------------------
       HERO
    ------------------------------------------------------- */

    .hero {
        padding: 25px 0 30px 0;
    }

    .hero-eyebrow {
        color: #818cf8;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 52px;
        line-height: 1.05;
        font-weight: 900;
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
        font-size: 17px;
        max-width: 760px;
        line-height: 1.7;
        margin-top: 15px;
    }


    /* -------------------------------------------------------
       CARDS
    ------------------------------------------------------- */

    .card {
        background:
            rgba(15, 23, 42, 0.72);

        border:
            1px solid rgba(148,163,184,0.10);

        border-radius: 20px;

        padding: 24px;

        box-shadow:
            0 18px 50px rgba(0,0,0,0.20);

        margin-bottom: 18px;
    }

    .card-title {
        font-size: 19px;
        font-weight: 750;
        margin-bottom: 7px;
    }

    .card-description {
        color: #64748b;
        font-size: 13px;
        line-height: 1.6;
    }


    /* -------------------------------------------------------
       METRICS
    ------------------------------------------------------- */

    .metric {
        background:
            rgba(15,23,42,0.70);

        border:
            1px solid rgba(255,255,255,0.07);

        border-radius: 18px;

        padding: 20px;

        min-height: 105px;
    }

    .metric-value {
        font-size: 29px;
        font-weight: 850;

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
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 5px;
    }


    /* -------------------------------------------------------
       PREDICTION
    ------------------------------------------------------- */

    .prediction {
        background:
            linear-gradient(
                145deg,
                rgba(124,58,237,0.17),
                rgba(37,99,235,0.12)
            );

        border:
            1px solid rgba(139,92,246,0.28);

        border-radius: 24px;

        padding: 35px 20px;

        text-align: center;

        box-shadow:
            0 25px 60px rgba(0,0,0,0.28);
    }

    .prediction-small {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .prediction-letter {
        font-size: 100px;
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

    .prediction-confidence {
        font-size: 21px;
        font-weight: 700;
    }


    /* -------------------------------------------------------
       PIPELINE
    ------------------------------------------------------- */

    .pipeline {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 9px;
        flex-wrap: wrap;
        margin: 20px 0;
    }

    .pipeline-step {
        background: rgba(30,41,59,0.75);
        border: 1px solid rgba(148,163,184,0.10);
        border-radius: 12px;
        padding: 12px 16px;
        color: #cbd5e1;
        font-size: 12px;
        font-weight: 650;
    }

    .pipeline-arrow {
        color: #6366f1;
        font-weight: 800;
    }


    /* -------------------------------------------------------
       CLASS BADGES
    ------------------------------------------------------- */

    .class-badge {
        display: inline-block;

        background:
            rgba(99,102,241,0.10);

        border:
            1px solid rgba(99,102,241,0.20);

        color: #c4b5fd;

        padding: 6px 10px;

        margin: 3px;

        border-radius: 8px;

        font-size: 12px;
        font-weight: 700;
    }


    /* -------------------------------------------------------
       INFO
    ------------------------------------------------------- */

    .info {
        border-left:
            3px solid #6366f1;

        background:
            rgba(99,102,241,0.07);

        border-radius: 10px;

        padding: 15px 18px;

        color: #cbd5e1;

        line-height: 1.7;

        margin: 15px 0;
    }


    /* -------------------------------------------------------
       SECTION
    ------------------------------------------------------- */

    .section-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #64748b;
        margin-bottom: 25px;
    }


    /* -------------------------------------------------------
       FOOTER
    ------------------------------------------------------- */

    .footer {
        text-align: center;
        color: #475569;
        font-size: 12px;
        padding: 35px 0 10px 0;
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
# DATASET EXTRACTION
# ============================================================

def extract_dataset():

    if os.path.exists(EXTRACT_FOLDER):
        dataset_folder = find_dataset_folder()

        if dataset_folder is not None:
            return dataset_folder

    if not os.path.exists(ZIP_FILE):

        st.error(
            "SIGNOVA could not find archive.zip. "
            "Please upload archive.zip to the same GitHub repository "
            "as app.py."
        )

        st.stop()

    try:

        os.makedirs(
            EXTRACT_FOLDER,
            exist_ok=True
        )

        with zipfile.ZipFile(
            ZIP_FILE,
            "r"
        ) as zip_ref:

            zip_ref.extractall(
                EXTRACT_FOLDER
            )

    except zipfile.BadZipFile:

        st.error(
            "archive.zip appears to be corrupted."
        )

        st.stop()

    dataset_folder = find_dataset_folder()

    if dataset_folder is None:

        st.error(
            "SIGNOVA could not locate the DATASET folder "
            "inside archive.zip."
        )

        st.stop()

    return dataset_folder


# ============================================================
# FIND DATASET FOLDER
# ============================================================

def find_dataset_folder():

    direct_path = os.path.join(
        EXTRACT_FOLDER,
        "DATASET"
    )

    if os.path.isdir(direct_path):
        return direct_path

    for root, dirs, files in os.walk(
        EXTRACT_FOLDER
    ):

        if os.path.basename(
            root
        ).upper() == "DATASET":

            return root

    return None


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # Convert to grayscale
    gray = image.convert("L")

    # Resize
    gray = gray.resize(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    # Slight smoothing to reduce random image noise
    gray = gray.filter(
        ImageFilter.GaussianBlur(
            radius=0.35
        )
    )

    # Improve contrast
    gray = ImageEnhance.Contrast(
        gray
    ).enhance(1.15)

    # Convert to NumPy
    array = np.asarray(
        gray,
        dtype=np.float32
    )

    # Normalize 0–255 → 0–1
    array = array / 255.0

    return array


# ============================================================
# SIGNOVA EDGE FEATURE
# ============================================================

def extract_features(image):

    """
    SIGNOVA feature representation.

    The image is converted into:
    1. Normalized grayscale information
    2. Horizontal edge changes
    3. Vertical edge changes
    4. Edge magnitude

    This gives the model information about both
    appearance and hand shape.
    """

    gray = preprocess_image(
        image
    )

    # Horizontal intensity changes
    gx = np.diff(
        gray,
        axis=1,
        append=gray[:, -1:]
    )

    # Vertical intensity changes
    gy = np.diff(
        gray,
        axis=0,
        append=gray[-1:, :]
    )

    # Edge magnitude
    magnitude = np.sqrt(
        (gx ** 2) +
        (gy ** 2)
    )

    # Compress edge magnitude
    magnitude = np.clip(
        magnitude,
        0,
        1
    )

    # Combine image appearance and shape information
    features = np.concatenate(
        [
            gray.flatten(),
            magnitude.flatten()
        ]
    )

    return features


# ============================================================
# DATASET LOADING
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

        image_files = [
            f
            for f in os.listdir(
                folder_path
            )
            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )
        ]

        if len(image_files) == 0:
            continue

        classes.append(
            folder
        )

        for filename in sorted(
            image_files
        ):

            image_path = os.path.join(
                folder_path,
                filename
            )

            try:

                image = Image.open(
                    image_path
                )

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

    classes = sorted(
        classes,
        key=lambda value: (
            not value.isdigit(),
            int(value)
            if value.isdigit()
            else value
        )
    )

    return (
        np.asarray(X),
        np.asarray(y),
        classes
    )


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def train_signova_model(
    X,
    y
):

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
                "classifier",
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

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=sorted(
            np.unique(y)
        )
    )

    return (
        model,
        accuracy,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        matrix
    )


# ============================================================
# LOAD SYSTEM
# ============================================================

with st.spinner(
    "SIGNOVA is preparing the recognition engine..."
):

    dataset_path = extract_dataset()

    X, y, classes = load_dataset(
        dataset_path
    )


if len(X) == 0:

    st.error(
        "No images were found in the dataset."
    )

    st.stop()


with st.spinner(
    "SIGNOVA is learning the hand-sign patterns..."
):

    (
        model,
        accuracy,
        X_train,
        X_test,
        y_train,
        y_test,
        predictions,
        matrix
    ) = train_signova_model(
        X,
        y
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">

            <div class="brand-mark">
                🤟
            </div>

            <div class="brand-name">
                SIGNOVA
            </div>

            <div class="brand-description">
                Hand Sign Recognition System
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "SYSTEM",
        [
            "◈ Recognition",
            "◈ Dataset Lab",
            "◈ Model Insights",
            "◈ How SIGNOVA Works",
            "◈ About"
        ],
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown(
        "**SYSTEM STATUS**"
    )

    st.success(
        "Recognition engine online"
    )

    st.caption(
        f"{len(classes)} sign classes detected"
    )

    st.caption(
        f"{len(y)} images loaded"
    )

    st.markdown("---")

    st.caption(
        "Image Processing & Computer Vision"
    )

    st.caption(
        "SIGNOVA Project"
    )


# ============================================================
# PAGE 1 — RECOGNITION
# ============================================================

if page == "◈ Recognition":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-eyebrow">
                COMPUTER VISION • HAND ANALYSIS
            </div>

            <div class="hero-title">
                SIGNOVA
            </div>

            <div class="hero-subtitle">
                A visual recognition system that analyzes
                hand-sign images and identifies the most
                likely letter or number from the trained
                dataset.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-value">
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
                <div class="metric-value">
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
                <div class="metric-value">
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
                <div class="metric-value">
                    SVM
                </div>
                <div class="metric-label">
                    Classifier
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="pipeline">

            <div class="pipeline-step">
                📷 Input
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-step">
                ⚙️ Preprocess
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-step">
                🧩 Feature Extraction
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-step">
                🧠 SVM
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-step">
                🎯 Prediction
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            Recognize a Sign
        </div>

        <div class="section-subtitle">
            Upload an image or capture a hand sign using your camera.
        </div>
        """,
        unsafe_allow_html=True
    )


    input_mode = st.radio(
        "Input source",
        [
            "Upload Image",
            "Camera"
        ],
        horizontal=True
    )


    image = None


    if input_mode == "Upload Image":

        uploaded = st.file_uploader(
            "Choose a hand-sign image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            help="Use a clear image with the hand visible."
        )

        if uploaded is not None:

            image = Image.open(
                uploaded
            )


    else:

        camera = st.camera_input(
            "Capture your hand sign"
        )

        if camera is not None:

            image = Image.open(
                camera
            )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if image is not None:

        st.markdown("<br>", unsafe_allow_html=True)

        left, right = st.columns(
            [1, 1]
        )


        with left:

            st.markdown(
                """
                <div class="card">

                    <div class="card-title">
                        Input Frame
                    </div>

                    <div class="card-description">
                        Image received by SIGNOVA.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.image(
                image,
                use_container_width=True
            )


        # Extract features
        features = extract_features(
            image
        )

        features = features.reshape(
            1,
            -1
        )


        # Probability
        probabilities = model.predict_proba(
            features
        )[0]

        model_classes = model.classes_

        ranking = np.argsort(
            probabilities
        )[::-1]

        best_index = ranking[0]

        predicted_sign = model_classes[
            best_index
        ]

        confidence = probabilities[
            best_index
        ]


        with right:

            st.markdown(
                """
                <div class="card">

                    <div class="card-title">
                        Recognition Result
                    </div>

                    <div class="card-description">
                        SIGNOVA's highest-confidence classification.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="prediction">

                    <div class="prediction-small">
                        Detected Sign
                    </div>

                    <div class="prediction-letter">
                        {predicted_sign}
                    </div>

                    <div class="prediction-confidence">
                        {confidence * 100:.2f}% confidence
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown("<br>", unsafe_allow_html=True)


            if confidence >= 0.80:

                st.success(
                    "High-confidence classification"
                )

            elif confidence >= 0.50:

                st.warning(
                    "Moderate-confidence classification"
                )

            else:

                st.error(
                    "Low-confidence classification. "
                    "Try a clearer image."
                )


        # ----------------------------------------------------
        # Top predictions
        # ----------------------------------------------------

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="section-title">
                Prediction Spectrum
            </div>

            <div class="section-subtitle">
                The five classes considered most likely by the model.
            </div>
            """,
            unsafe_allow_html=True
        )


        top_n = min(
            5,
            len(model_classes)
        )

        top_indices = ranking[
            :top_n
        ]

        chart_labels = [
            str(model_classes[i])
            for i in top_indices
        ]

        chart_values = [
            float(probabilities[i] * 100)
            for i in top_indices
        ]


        chart_data = {
            "Sign": chart_labels,
            "Confidence (%)": chart_values
        }

        st.bar_chart(
            chart_data,
            x="Sign",
            y="Confidence (%)"
        )


        # Details
        detail_columns = st.columns(
            top_n
        )

        for position, index in enumerate(
            top_indices
        ):

            with detail_columns[position]:

                st.metric(
                    f"#{position + 1}",
                    str(model_classes[index]),
                    f"{probabilities[index] * 100:.2f}%"
                )


# ============================================================
# PAGE 2 — DATASET LAB
# ============================================================

elif page == "◈ Dataset Lab":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-eyebrow">
                DATASET EXPLORATION
            </div>

            <div class="hero-title">
                Dataset Lab
            </div>

            <div class="hero-subtitle">
                Explore the hand-sign classes used to train SIGNOVA.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # Dataset summary

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "Total Images",
            len(y)
        )

    with b:
        st.metric(
            "Classes",
            len(classes)
        )

    with c:
        st.metric(
            "Average Images / Class",
            f"{len(y) / len(classes):.1f}"
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # Class distribution

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                Class Distribution
            </div>

            <div class="card-description">
                Number of training images available for each sign.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    counts = {
        class_name: int(
            np.sum(
                y == class_name
            )
        )
        for class_name in classes
    }


    distribution_data = {
        "Sign": list(counts.keys()),
        "Images": list(counts.values())
    }


    st.bar_chart(
        distribution_data,
        x="Sign",
        y="Images"
    )


    # --------------------------------------------------------
    # Sign explorer
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            Sign Explorer
        </div>

        <div class="section-subtitle">
            Select a class to inspect example images.
        </div>
        """,
        unsafe_allow_html=True
    )


    selected_class = st.selectbox(
        "Select sign",
        classes
    )


    selected_folder = os.path.join(
        dataset_path,
        selected_class
    )


    image_files = [
        file
        for file in os.listdir(
            selected_folder
        )
        if file.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ]


    st.write(
        f"### Sign `{selected_class}`"
    )

    st.caption(
        f"{len(image_files)} images found in this class."
    )


    preview_files = image_files[
        :20
    ]


    columns = st.columns(5)


    for index, filename in enumerate(
        preview_files
    ):

        image_path = os.path.join(
            selected_folder,
            filename
        )

        try:

            sample = Image.open(
                image_path
            )

            with columns[
                index % 5
            ]:

                st.image(
                    sample,
                    use_container_width=True
                )

        except Exception:
            pass


# ============================================================
# PAGE 3 — MODEL INSIGHTS
# ============================================================

elif page == "◈ Model Insights":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-eyebrow">
                MODEL EVALUATION
            </div>

            <div class="hero-title">
                Model Insights
            </div>

            <div class="hero-subtitle">
                A transparent view of how SIGNOVA performs
                on previously unseen test images.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # Performance cards

    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Test Accuracy",
            f"{accuracy * 100:.2f}%"
        )


    with c2:

        st.metric(
            "Training Images",
            len(X_train)
        )


    with c3:

        st.metric(
            "Testing Images",
            len(X_test)
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # Explanation

    st.markdown(
        """
        <div class="info">

        <strong>How should this accuracy be interpreted?</strong>

        SIGNOVA uses an 80/20 stratified train-test split.
        Approximately 80% of the available images are used
        to train the classifier, while 20% are kept separate
        for evaluation.

        The reported accuracy therefore measures how often
        SIGNOVA correctly identifies images that were not
        included in its training subset.

        </div>
        """,
        unsafe_allow_html=True
    )


    # Confusion matrix

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                Confusion Matrix
            </div>

            <div class="card-description">
                Rows represent the actual class while columns
                represent the predicted class.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    matrix_labels = sorted(
        np.unique(y)
    )


    # Normalize confusion matrix
    row_sums = matrix.sum(
        axis=1,
        keepdims=True
    )

    normalized_matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(
            matrix,
            dtype=float
        ),
        where=row_sums != 0
    )


    st.dataframe(
        normalized_matrix,
        use_container_width=True,
        height=600
    )


    st.caption(
        "Matrix values are normalized by actual class."
    )


# ============================================================
# PAGE 4 — HOW SIGNOVA WORKS
# ============================================================

elif page == "◈ How SIGNOVA Works":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-eyebrow">
                COMPUTER VISION PIPELINE
            </div>

            <div class="hero-title">
                How SIGNOVA Works
            </div>

            <div class="hero-subtitle">
                From a raw hand image to a machine-learning prediction.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="pipeline">

            <div class="pipeline-step">
                📷 Image
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-step">
                🌑 Grayscale
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-step">
                📐 Resize
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-step">
                🧩 Features
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-step">
                🧠 SVM
            </div>

            <div class="pipeline-arrow">
                →
            </div>

            <div class="pipeline-step">
                🎯 Sign
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Step 1
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                01 — Image Acquisition
            </div>

            <div class="card-description">

                SIGNOVA accepts an image either from a local
                upload or from the device camera.

                The original image is preserved for display,
                while a processed representation is created
                for classification.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Step 2
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                02 — Grayscale Conversion
            </div>

            <div class="card-description">

                The RGB image is converted into a single-channel
                grayscale image.

                This reduces the information required by the
                classifier while keeping the brightness and
                structural information of the hand.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Step 3
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                03 — Image Normalization
            </div>

            <div class="card-description">

                Every image is resized to 48 × 48 pixels and
                normalized from a 0–255 intensity range into
                values between 0 and 1.

                A small amount of smoothing and contrast
                enhancement is also applied.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Step 4
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                04 — Shape Feature Extraction
            </div>

            <div class="card-description">

                SIGNOVA calculates changes in pixel intensity
                horizontally and vertically.

                These changes form an edge-magnitude map.
                The system combines the normalized grayscale
                image with this edge information.

                This gives the classifier information about
                both the visual appearance and the shape
                boundaries of the hand.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Step 5
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                05 — SVM Classification
            </div>

            <div class="card-description">

                The resulting feature vector is passed to a
                Support Vector Machine classifier using an
                RBF kernel.

                The classifier learns the visual differences
                between the 36 available hand-sign classes.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Step 6
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                06 — Final Prediction
            </div>

            <div class="card-description">

                SIGNOVA calculates probabilities for the
                available classes and selects the class with
                the highest probability.

                The interface then presents the predicted
                sign, confidence level and the five strongest
                candidate predictions.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE 5 — ABOUT
# ============================================================

elif page == "◈ About":

    st.markdown(
        """
        <div class="hero">

            <div class="hero-eyebrow">
                PROJECT INFORMATION
            </div>

            <div class="hero-title">
                About SIGNOVA
            </div>

            <div class="hero-subtitle">
                A student-built Image Processing and Computer
                Vision project focused on static hand-sign
                classification.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    left, right = st.columns(
        [1.3, 1]
    )


    with left:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    Project Objective
                </div>

                <div class="card-description">

                    SIGNOVA demonstrates how image processing
                    techniques and supervised machine learning
                    can be combined to recognize static hand
                    signs.

                    The system focuses on individual letters
                    and numbers rather than continuous
                    sentence-level sign-language translation.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    Technology Stack
                </div>

                <div class="card-description">

                    • Python<br>
                    • Streamlit<br>
                    • NumPy<br>
                    • Pillow<br>
                    • Scikit-learn<br>
                    • Support Vector Machine

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with right:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    Dataset
                </div>

                <div class="card-description">

                    SIGNOVA is trained using the supplied
                    hand-sign image dataset.

                </div>

                <br>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="metric">

                <div class="metric-value">
                    {len(y)}
                </div>

                <div class="metric-label">
                    Total Images
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
            <div class="metric">

                <div class="metric-value">
                    {len(classes)}
                </div>

                <div class="metric-label">
                    Recognition Classes
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                Recognition Classes
            </div>

            <div class="card-description">
                SIGNOVA currently recognizes:
            </div>

            <br>

        </div>
        """,
        unsafe_allow_html=True
    )


    badges = ""

    for class_name in classes:

        badges += (
            f'<span class="class-badge">'
            f'{class_name}'
            f'</span>'
        )


    st.markdown(
        badges,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="footer">
            SIGNOVA • Hand Sign Recognition System
            <br>
            Image Processing & Computer Vision Project
        </div>
        """,
        unsafe_allow_html=True
    )
