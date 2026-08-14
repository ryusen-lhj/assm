"""SIGNOVA Streamlit Cloud compatibility entry point.

This file keeps the existing app.py UI/model intact while making the webcam
path conservative for Streamlit Community Cloud:
- Twilio TURN is required for remote WebRTC.
- Browser capture uses the simplest video-only constraint.
- Legacy video_processor_factory is adapted to the recommended
  video_frame_callback API.
- The MediaPipe processor is created lazily inside the video callback thread,
  not on Streamlit's main thread.
- Recognition uses 2D landmarks and a balanced RBF-SVM.
"""

from pathlib import Path
import runpy
import threading

import mediapipe as mp
import streamlit as st
import streamlit_webrtc
import sklearn.neighbors
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from twilio.rest import Client


# ============================================================
# WEBRTC / TURN
# ============================================================

if not hasattr(streamlit_webrtc, "_signova_original_webrtc_streamer"):
    streamlit_webrtc._signova_original_webrtc_streamer = (
        streamlit_webrtc.webrtc_streamer
    )

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
            "Camera relay is not configured. Add TWILIO_ACCOUNT_SID and "
            "TWILIO_AUTH_TOKEN in Manage app → Secrets, then reboot."
        )
        st.stop()

    try:
        ice_servers = _twilio_ice_servers(sid, auth)
    except Exception as exc:
        st.error(
            "Twilio TURN could not be activated. Check the two Streamlit "
            f"Secrets. {type(exc).__name__}: {str(exc)[:240]}"
        )
        st.stop()

    if not ice_servers:
        st.error("Twilio returned no STUN/TURN servers. Recheck the account.")
        st.stop()

    return {"iceServers": ice_servers}


def _cloud_webrtc_streamer(*args, **kwargs):
    """Use the smallest reliable browser-camera configuration.

    A key detail is that the MediaPipe processor is created on the callback
    thread after the first browser frame arrives. Creating it on Streamlit's
    main script thread and then using it from another thread can destabilize
    the video path on cloud reruns.
    """
    kwargs["rtc_configuration"] = _turn_configuration()

    # Request only a webcam. Avoid device-specific width/height constraints
    # while diagnosing / running across different Chrome camera drivers.
    kwargs["media_stream_constraints"] = {
        "video": True,
        "audio": False,
    }

    factory = kwargs.pop("video_processor_factory", None)
    kwargs.pop("async_processing", None)

    if factory is not None and "video_frame_callback" not in kwargs:
        holder = {"processor": None}
        holder_lock = threading.Lock()

        def _video_frame_callback(frame):
            if holder["processor"] is None:
                with holder_lock:
                    if holder["processor"] is None:
                        holder["processor"] = factory()
            return holder["processor"].recv(frame)

        kwargs["video_frame_callback"] = _video_frame_callback

    kwargs.setdefault("media_toggle_controls", False)

    return _original_webrtc_streamer(*args, **kwargs)


streamlit_webrtc.webrtc_streamer = _cloud_webrtc_streamer


# ============================================================
# RECOGNITION: 2D LANDMARKS ONLY
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
    sklearn.neighbors._signova_original_knn = (
        sklearn.neighbors.KNeighborsClassifier
    )


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
# RUN REAL APP
# ============================================================

runpy.run_path(
    str(Path(__file__).with_name("app.py")),
    run_name="__main__",
)
