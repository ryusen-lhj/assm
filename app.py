import streamlit as st
import numpy as np
import opencv-python-headless
import zipfile
import os
import shutil
import random
import pickle
import tempfile

from PIL import Image
from skimage.feature import hog

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SignVision AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #080b16 0%,
            #111827 50%,
            #0f172a 100%
        );
        color: white;
    }

    /* Hide Streamlit default menu */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Main title */
    .main-title {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;

        background: linear-gradient(
            90deg,
            #60a5fa,
            #a78bfa,
            #f472b6
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 18px;
        margin-bottom: 35px;
    }

    /* Cards */
    .card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 20px;

        box-shadow:
            0 10px 30px rgba(0,0,0,0.25);
    }

    /* Prediction card */
    .prediction-card {
        background: linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.25),
            rgba(124, 58, 237, 0.25)
        );

        border: 1px solid rgba(129, 140, 248, 0.35);
        border-radius: 22px;

        padding: 30px;

        text-align: center;

        box-shadow:
            0 15px 40px rgba(0,0,0,0.35);
    }

    .prediction-label {
        color: #94a3b8;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .prediction-value {
        font-size: 90px;
        font-weight: 900;

        background: linear-gradient(
            90deg,
            #60a5fa,
            #c084fc
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .confidence {
        font-size: 22px;
        color: #cbd5e1;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(30,41,59,0.7);
        border-radius: 15px;
        padding: 20px;
        text-align: center;

        border: 1px solid rgba(255,255,255,0.08);
    }

    .metric-number {
        font-size: 30px;
        font-weight: 800;
        color: #60a5fa;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 14px;
    }

    /* Section title */
    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Info box */
    .info-box {
        background: rgba(15, 23, 42, 0.8);
        border-left: 4px solid #60a5fa;
        border-radius: 10px;
        padding: 18px;
        margin: 15px 0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

ZIP_FILE = "archive.zip"
EXTRACT_FOLDER = "extracted_dataset"
MODEL_FILE = "sign_language_model.pkl"

IMAGE_SIZE = (128, 128)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤟 SignVision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Hand Sign Language Recognition using Image Processing & Computer Vision'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DATASET EXTRACTION
# ============================================================

def extract_dataset():

    if os.path.exists(EXTRACT_FOLDER):
        return

    if not os.path.exists(ZIP_FILE):
        st.error(
            "❌ archive.zip was not found. "
            "Please place archive.zip in the same folder as app.py."
        )
        st.stop()

    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)


# ============================================================
# FIND DATASET DIRECTORY
# ============================================================

def find_dataset_folder():

    possible_paths = [
        os.path.join(EXTRACT_FOLDER, "DATASET"),
        os.path.join(EXTRACT_FOLDER, "archive", "DATASET"),
        os.path.join(EXTRACT_FOLDER, "dataset", "DATASET")
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Search recursively
    for root, dirs, files in os.walk(EXTRACT_FOLDER):
        if os.path.basename(root).upper() == "DATASET":
            return root

    return None


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # Convert PIL image to RGB
    image = image.convert("RGB")

    # Convert to numpy
    image = np.array(image)

    # Convert RGB -> grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize
    gray = cv2.resize(gray, IMAGE_SIZE)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    return gray


# ============================================================
# HOG FEATURE EXTRACTION
# ============================================================

def extract_hog(image):

    gray = preprocess_image(image)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset(dataset_path):

    X = []
    y = []

    classes = sorted(
        [
            folder
            for folder in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, folder))
        ]
    )

    for label in classes:

        folder_path = os.path.join(dataset_path, label)

        for filename in os.listdir(folder_path):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                image_path = os.path.join(
                    folder_path,
                    filename
                )

                try:

                    image = Image.open(image_path)

                    features = extract_hog(image)

                    X.append(features)
                    y.append(label)

                except Exception:
                    continue

    return np.array(X), np.array(y), classes


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=True,
                random_state=42
            )
        )
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, accuracy, X_test, y_test, predictions


# ============================================================
# INITIALIZE DATASET
# ============================================================

with st.spinner("🔄 Preparing hand-sign dataset..."):

    extract_dataset()

    dataset_path = find_dataset_folder()

    if dataset_path is None:

        st.error(
            "❌ Could not locate the DATASET folder."
        )

        st.stop()

    X, y, classes = load_dataset(dataset_path)


# ============================================================
# TRAINING
# ============================================================

with st.spinner("🧠 Training computer vision model..."):

    model, accuracy, X_test, y_test, predictions = train_model(
        X,
        y
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🤟 SignVision AI"
    )

    st.markdown(
        "---"
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Recognition",
            "📊 Dataset Explorer",
            "🧠 How It Works",
            "ℹ️ About Project"
        ]
    )

    st.markdown("---")

    st.markdown("### 📊 Model Information")

    st.write(
        f"**Classes:** {len(classes)}"
    )

    st.write(
        f"**Images:** {len(y)}"
    )

    st.write(
        f"**Image Size:** 300 × 300"
    )

    st.write(
        f"**Features:** HOG"
    )

    st.write(
        f"**Classifier:** SVM"
    )

    st.write(
        f"**Test Accuracy:** {accuracy * 100:.2f}%"
    )

    st.markdown("---")

    st.caption(
        "Image Processing & Computer Vision Project"
    )


# ============================================================
# RECOGNITION PAGE
# ============================================================

if page == "🏠 Recognition":

    st.markdown(
        '<div class="section-title">'
        '🔍 Hand Sign Recognition'
        '</div>',
        unsafe_allow_html=True
    )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {len(classes)}
                </div>
                <div class="metric-label">
                    SIGN CLASSES
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {len(y)}
                </div>
                <div class="metric-label">
                    TRAINING IMAGES
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {accuracy * 100:.1f}%
                </div>
                <div class="metric-label">
                    MODEL ACCURACY
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number">
                    HOG
                </div>
                <div class="metric-label">
                    FEATURE METHOD
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Input method
    st.markdown(
        "### 📷 Choose Input Method"
    )

    input_method = st.radio(
        "Input",
        [
            "📤 Upload Image",
            "📸 Use Camera"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    image = None

    if input_method == "📤 Upload Image":

        uploaded_file = st.file_uploader(
            "Upload a hand sign image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            )

    else:

        camera_image = st.camera_input(
            "Take a picture of your hand sign"
        )

        if camera_image is not None:

            image = Image.open(
                camera_image
            )

    # Prediction
    if image is not None:

        col1, col2 = st.columns(
            [1, 1.2]
        )

        with col1:

            st.markdown(
                "### 🖼️ Input Image"
            )

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.markdown(
                "### 🤖 AI Prediction"
            )

            features = extract_hog(image)

            features = features.reshape(
                1,
                -1
            )

            probabilities = model.predict_proba(
                features
            )[0]

            classes_model = model.classes_

            # Sort predictions
            sorted_indices = np.argsort(
                probabilities
            )[::-1]

            best_index = sorted_indices[0]

            predicted_class = classes_model[
                best_index
            ]

            confidence = probabilities[
                best_index
            ]

            st.markdown(
                f"""
                <div class="prediction-card">

                    <div class="prediction-label">
                        Predicted Sign
                    </div>

                    <div class="prediction-value">
                        {predicted_class}
                    </div>

                    <div class="confidence">
                        Confidence:
                        <strong>
                            {confidence * 100:.2f}%
                        </strong>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Confidence message
            if confidence >= 0.80:

                st.success(
                    "🟢 High confidence prediction"
                )

            elif confidence >= 0.50:

                st.warning(
                    "🟡 Moderate confidence prediction"
                )

            else:

                st.error(
                    "🔴 Low confidence prediction. "
                    "Try another image."
                )

        # Top 5
        st.markdown(
            "### 🏆 Top 5 Predictions"
        )

        top_indices = sorted_indices[:5]

        top_classes = [
            classes_model[i]
            for i in top_indices
        ]

        top_probabilities = [
            probabilities[i] * 100
            for i in top_indices
        ]

        chart_data = {
            "Sign": top_classes,
            "Confidence (%)": top_probabilities
        }

        st.bar_chart(
            chart_data,
            x="Sign",
            y="Confidence (%)"
        )

        # Prediction table
        st.markdown(
            "#### Prediction Details"
        )

        for i in range(
            len(top_classes)
        ):

            st.write(
                f"**{i + 1}. "
                f"{top_classes[i]}** — "
                f"{top_probabilities[i]:.2f}%"
            )


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "📊 Dataset Explorer":

    st.markdown(
        '<div class="section-title">'
        '📊 Dataset Explorer'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        The dataset contains hand-sign images representing
        letters A–Z and numbers 0–9. Each class contains
        approximately 25 images.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Class selector
    selected_class = st.selectbox(
        "Select a sign",
        classes
    )

    folder = os.path.join(
        dataset_path,
        selected_class
    )

    image_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    st.write(
        f"### Sign: `{selected_class}`"
    )

    st.write(
        f"Number of images: **{len(image_files)}**"
    )

    # Display images
    cols = st.columns(5)

    for index, filename in enumerate(
        image_files
    ):

        image_path = os.path.join(
            folder,
            filename
        )

        image = Image.open(
            image_path
        )

        with cols[index % 5]:

            st.image(
                image,
                use_container_width=True
            )


# ============================================================
# HOW IT WORKS
# ============================================================

elif page == "🧠 How It Works":

    st.markdown(
        '<div class="section-title">'
        '🧠 How the System Works'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### 1️⃣ Image Acquisition

        The user provides a hand-sign image using either:

        - Image upload
        - Camera input

        The image becomes the input for the computer vision
        pipeline.

        ---

        ### 2️⃣ Image Preprocessing

        The input image is converted from RGB to grayscale.

        The image is then resized to:

        **128 × 128 pixels**

        Histogram Equalization is also applied to improve
        image contrast.

        ---

        ### 3️⃣ Feature Extraction

        The system uses:

        **HOG — Histogram of Oriented Gradients**

        HOG analyzes the edges and shapes within the hand.

        This is useful for hand-sign recognition because
        different signs have different finger orientations
        and shapes.

        ---

        ### 4️⃣ Machine Learning Classification

        The extracted HOG features are passed into:

        **Support Vector Machine (SVM)**

        The SVM learns the difference between the
        36 hand-sign classes.

        The classes are:

        **A–Z + 0–9**

        ---

        ### 5️⃣ Prediction

        The trained SVM calculates the probability for
        every possible hand sign.

        The sign with the highest probability becomes
        the final prediction.

        ---

        ### 6️⃣ Result Visualization

        The application displays:

        - Predicted sign
        - Confidence percentage
        - Top 5 predictions
        - Confidence chart

        This makes the computer vision system easier
        for users to understand.
        """
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="section-title">'
        'ℹ️ About SignVision AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <h2>🤟 SignVision AI</h2>

        <p>
        SignVision AI is a computer vision application
        designed to recognize hand signs representing
        letters and numbers.
        </p>

        <h3>🎯 Project Objective</h3>

        <p>
        The objective of this project is to demonstrate
        how image processing and machine learning can
        be combined to recognize hand gestures.
        </p>

        <h3>🔧 Technologies Used</h3>

        <ul>
            <li>Python</li>
            <li>Streamlit</li>
            <li>OpenCV</li>
            <li>Scikit-learn</li>
            <li>HOG Feature Extraction</li>
            <li>Support Vector Machine</li>
            <li>NumPy</li>
            <li>Pillow</li>
        </ul>

        <h3>📚 Computer Vision Pipeline</h3>

        <p>
        Image → Preprocessing → HOG Features →
        SVM Classification → Prediction
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### 📈 Model Performance"
    )

    st.metric(
        "Test Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    st.caption(
        "Accuracy is calculated using an 80/20 stratified "
        "train-test split."
    )
