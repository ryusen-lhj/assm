"""Streamlit Cloud entry point for SIGNOVA.

This wrapper keeps the existing SIGNOVA app intact while applying two cloud/
recognition fixes before app.py is imported:

1. Adds STUN servers for WebRTC connectivity on Streamlit Community Cloud.
2. Makes recognition more robust by removing MediaPipe's unstable Z/depth
   coordinate and replacing distance-weighted KNN with a balanced RBF-SVM.
"""

import mediapipe as mp
import streamlit_webrtc
import sklearn.neighbors

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC


# ============================================================
# 1. WEBRTC CLOUD CONFIGURATION
# ============================================================

_original_webrtc_streamer = streamlit_webrtc.webrtc_streamer


def _cloud_webrtc_streamer(*args, **kwargs):
    kwargs.setdefault(
        "rtc_configuration",
        {
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302",
                        "stun:stun1.l.google.com:19302",
                    ]
                }
            ]
        },
    )
    return _original_webrtc_streamer(*args, **kwargs)


streamlit_webrtc.webrtc_streamer = _cloud_webrtc_streamer


# ============================================================
# 2. FORCE 2D LANDMARK GEOMETRY
# ============================================================

# The training set consists of annotated 2D images while the live webcam can
# produce noticeably different MediaPipe Z/depth estimates.  Because the old
# app included Z in every feature vector, those depth differences could dominate
# nearest-neighbour distances and make one class (for example O) win constantly.
#
# Zeroing Z for BOTH training and live inference keeps the model focused on the
# actual visible hand shape: X/Y landmark geometry.

_original_hands_process = mp.solutions.hands.Hands.process


def _process_without_depth(self, image):
    result = _original_hands_process(self, image)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            for landmark in hand.landmark:
                landmark.z = 0.0

    return result


mp.solutions.hands.Hands.process = _process_without_depth


# ============================================================
# 3. ROBUST CLASSIFIER DROP-IN
# ============================================================

class RobustSignClassifier(BaseEstimator, ClassifierMixin):
    """Drop-in replacement for the app's KNeighborsClassifier.

    app.py still constructs `KNeighborsClassifier(...)`, but this wrapper
    internally trains a balanced RBF-SVM.  That avoids the old behaviour where
    distance-weighted KNN could return one class with a misleading 100% score.
    """

    def __init__(self, n_neighbors=5, weights="distance"):
        # Keep the same constructor arguments so sklearn/app.py can use this as
        # a transparent replacement.
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model_ = None

    def fit(self, X, y):
        self.model_ = SVC(
            kernel="rbf",
            C=8.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def score(self, X, y):
        return self.model_.score(X, y)


# app.py imports KNeighborsClassifier after this wrapper starts, so replacing
# the symbol here changes the classifier without needing another large rewrite
# of app.py.
sklearn.neighbors.KNeighborsClassifier = RobustSignClassifier


# ============================================================
# START SIGNOVA
# ============================================================

# Importing app executes the existing Streamlit application.
import app  # noqa: E402,F401
