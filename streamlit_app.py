"""Stable Streamlit Cloud entry point for SIGNOVA.

This file deliberately keeps all compatibility patches idempotent. Streamlit
reruns the entry script many times, so repeatedly monkey-patching WebRTC,
MediaPipe, or sklearn can corrupt lifecycle state. The original functions are
saved once and reused on every rerun.
"""

from pathlib import Path
import time

import mediapipe as mp
import streamlit as st
import streamlit_webrtc
import sklearn.neighbors

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from twilio.rest import Client


# ============================================================
# WEBRTC CLOUD CONFIGURATION
# ============================================================

# Save the REAL streamlit-webrtc function once. On later Streamlit reruns we
# reuse this reference instead of wrapping an already wrapped function.
if not hasattr(streamlit_webrtc, "_signova_original_webrtc_streamer"):
    streamlit_webrtc._signova_original_webrtc_streamer = (
        streamlit_webrtc.webrtc_streamer
    )

_original_webrtc_streamer = (
    streamlit_webrtc._signova_original_webrtc_streamer
)

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


def _secret(name):
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return None


def _rtc_configuration():
    """Use cached Twilio TURN credentials when available, else STUN."""
    sid = _secret("TWILIO_ACCOUNT_SID")
    auth_token = _secret("TWILIO_AUTH_TOKEN")

    if not sid or not auth_token:
        return _STUN_ONLY

    now = time.time()
    cache_key = "_signova_turn_cache"
    cached = st.session_state.get(cache_key)

    if cached:
        if now < cached.get("expires_at", 0):
            ice_servers = cached.get("ice_servers")
            if ice_servers:
                return {"iceServers": ice_servers}

    try:
        token = Client(sid, auth_token).tokens.create()
        ice_servers = token.ice_servers
        st.session_state[cache_key] = {
            "ice_servers": ice_servers,
            "expires_at": now + 1800,
        }
        return {"iceServers": ice_servers}
    except Exception:
        # Keep the application usable even if Twilio token generation fails.
        return _STUN_ONLY


def _cloud_webrtc_streamer(*args, **kwargs):
    kwargs.setdefault("rtc_configuration", _rtc_configuration())

    # The app uses the legacy class-based video_processor_factory API. Running
    # it synchronously is more stable on Community Cloud and avoids extra frame
    # worker lifecycle races during rapid Streamlit reruns / disconnects.
    kwargs["async_processing"] = False

    return _original_webrtc_streamer(*args, **kwargs)


streamlit_webrtc.webrtc_streamer = _cloud_webrtc_streamer


# ============================================================
# FORCE 2D LANDMARK GEOMETRY — IDEMPOTENT PATCH
# ============================================================

HandsClass = mp.solutions.hands.Hands

if not hasattr(HandsClass, "_signova_original_process"):
    HandsClass._signova_original_process = HandsClass.process

_original_hands_process = HandsClass._signova_original_process


def _process_without_depth(self, image):
    result = _original_hands_process(self, image)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            for landmark in hand.landmark:
                landmark.z = 0.0

    return result


HandsClass.process = _process_without_depth


# ============================================================
# ROBUST CLASSIFIER DROP-IN — IDEMPOTENT PATCH
# ============================================================

if not hasattr(sklearn.neighbors, "_signova_original_knn"):
    sklearn.neighbors._signova_original_knn = (
        sklearn.neighbors.KNeighborsClassifier
    )


class RobustSignClassifier(BaseEstimator, ClassifierMixin):
    """KNN-compatible constructor backed by a balanced probabilistic RBF-SVM."""

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
# EXECUTE THE REAL STREAMLIT APP ON EVERY RERUN
# ============================================================

# `import app` only executes once per Python process because of Python's import
# cache. Streamlit expects the main script to execute on every rerun. Executing
# app.py explicitly gives it normal Streamlit rerun semantics while preserving
# this compatibility layer.
_app_path = Path(__file__).with_name("app.py")
_app_code = compile(
    _app_path.read_text(encoding="utf-8"),
    str(_app_path),
    "exec",
)
exec(_app_code, globals(), globals())
