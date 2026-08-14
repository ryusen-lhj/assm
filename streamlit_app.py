"""Streamlit Cloud entry point for SIGNOVA.

This wrapper keeps the existing SIGNOVA app intact while applying:
1. WebRTC STUN/TURN configuration for Streamlit Community Cloud.
2. 2D-only MediaPipe landmark geometry.
3. A balanced RBF-SVM drop-in classifier.
"""

import time

import mediapipe as mp
import streamlit as st
import streamlit_webrtc
import sklearn.neighbors

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from twilio.rest import Client


# ============================================================
# 1. WEBRTC CLOUD CONFIGURATION
# ============================================================

_original_webrtc_streamer = streamlit_webrtc.webrtc_streamer

_STUN_ONLY = {
    "iceServers": [
        {
            "urls": [
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302",
            ]
        }
    ]
}

_turn_cache = {
    "ice_servers": None,
    "expires_at": 0.0,
}


def _secret(name):
    """Read a Streamlit secret safely without crashing when it is absent."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return None


def _rtc_configuration():
    """Return Twilio TURN credentials when configured, otherwise STUN only."""
    sid = _secret("TWILIO_ACCOUNT_SID")
    auth_token = _secret("TWILIO_AUTH_TOKEN")

    if not sid or not auth_token:
        return _STUN_ONLY, False

    now = time.time()

    if (
        _turn_cache["ice_servers"] is not None
        and now < _turn_cache["expires_at"]
    ):
        return {"iceServers": _turn_cache["ice_servers"]}, True

    try:
        client = Client(sid, auth_token)
        token = client.tokens.create()

        # Keep TURN credentials fresh. Twilio tokens are temporary, so do not
        # cache them for the whole lifetime of the Streamlit process.
        _turn_cache["ice_servers"] = token.ice_servers
        _turn_cache["expires_at"] = now + 1800

        return {"iceServers": token.ice_servers}, True

    except Exception:
        # Never crash the whole SIGNOVA app if TURN token retrieval fails.
        return _STUN_ONLY, False


def _cloud_webrtc_streamer(*args, **kwargs):
    if "rtc_configuration" not in kwargs:
        rtc_config, has_turn = _rtc_configuration()
        kwargs["rtc_configuration"] = rtc_config

        if not has_turn:
            st.warning(
                "TURN relay is not configured. If the camera stays on "
                "'Connection is taking longer than expected', add "
                "TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in Streamlit Secrets."
            )

    return _original_webrtc_streamer(*args, **kwargs)


streamlit_webrtc.webrtc_streamer = _cloud_webrtc_streamer


# ============================================================
# 2. FORCE 2D LANDMARK GEOMETRY
# ============================================================

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
    """Drop-in replacement for KNeighborsClassifier using a balanced RBF-SVM."""

    def __init__(self, n_neighbors=5, weights="distance"):
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


sklearn.neighbors.KNeighborsClassifier = RobustSignClassifier


# ============================================================
# START SIGNOVA
# ============================================================

import app  # noqa: E402,F401
