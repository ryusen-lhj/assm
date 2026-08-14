"""Stable Streamlit Cloud entry point for SIGNOVA.

This compatibility layer does three things before running app.py:
1. Requires a working Twilio TURN relay on Community Cloud.
2. Converts the legacy video_processor_factory into the current
   function-based video_frame_callback API.
3. Keeps recognition on 2D landmarks with a balanced RBF-SVM.
"""

from pathlib import Path
import runpy

import mediapipe as mp
import streamlit as st
import streamlit_webrtc
import sklearn.neighbors
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from twilio.rest import Client


# ============================================================
# WEBRTC: REQUIRE TURN + USE FUNCTION CALLBACK
# ============================================================

if not hasattr(streamlit_webrtc, "_signova_original_webrtc_streamer"):
    streamlit_webrtc._signova_original_webrtc_streamer = streamlit_webrtc.webrtc_streamer

_original_webrtc_streamer = streamlit_webrtc._signova_original_webrtc_streamer


def _secret(name):
    try:
        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def _twilio_ice_servers(account_sid, auth_token):
    token = Client(account_sid, auth_token).tokens.create(ttl=3600)
    return token.ice_servers


def _turn_configuration():
    sid = _secret("TWILIO_ACCOUNT_SID")
    auth = _secret("TWILIO_AUTH_TOKEN")

    if not sid or not auth:
        st.error(
            "WebRTC TURN is required for this Streamlit Cloud deployment. "
            "Add TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in Manage app → Secrets, "
            "then reboot the app."
        )
        st.stop()

    try:
        ice_servers = _twilio_ice_servers(sid, auth)
    except Exception as exc:
        st.error(
            "Twilio TURN could not be activated. Check the two Streamlit Secrets. "
            f"Error: {type(exc).__name__}: {str(exc)[:240]}"
        )
        st.stop()

    if not ice_servers:
        st.error("Twilio returned no ICE/TURN servers. Check the Twilio account and reboot.")
        st.stop()

    return {"iceServers": ice_servers}


def _cloud_webrtc_streamer(*args, **kwargs):
    # Do not silently fall back to STUN. The exact asyncio/aioice error the app
    # was seeing occurs after a failed ICE path on Community Cloud.
    kwargs["rtc_configuration"] = _turn_configuration()

    # streamlit-webrtc now recommends video_frame_callback instead of the
    # legacy class-based video_processor_factory API. Convert transparently so
    # the existing SIGNOVA app can stay small and stable.
    factory = kwargs.pop("video_processor_factory", None)
    if factory is not None and "video_frame_callback" not in kwargs:
        key = str(kwargs.get("key", "signova"))
        processor_key = f"_signova_processor_{key}"
        if processor_key not in st.session_state:
            st.session_state[processor_key] = factory()
        processor = st.session_state[processor_key]
        kwargs["video_frame_callback"] = processor.recv

    kwargs.pop("async_processing", None)
    kwargs.setdefault("media_toggle_controls", False)
    return _original_webrtc_streamer(*args, **kwargs)


streamlit_webrtc.webrtc_streamer = _cloud_webrtc_streamer


# ============================================================
# RECOGNITION: FORCE 2D LANDMARKS
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
# RECOGNITION: BALANCED RBF-SVM DROP-IN
# ============================================================

if not hasattr(sklearn.neighbors, "_signova_original_knn"):
    sklearn.neighbors._signova_original_knn = sklearn.neighbors.KNeighborsClassifier


class RobustSignClassifier(BaseEstimator, ClassifierMixin):
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
# RUN THE REAL STREAMLIT APP ON EVERY RERUN
# ============================================================

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
