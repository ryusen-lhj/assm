"""
streamlit_app.py
-----------------
Streamlit version of the Hand Sign Language Detector.

Streamlit reruns the whole script on every interaction, so it can't just
loop over cv2.VideoCapture like a desktop app. Instead we use
streamlit-webrtc, which opens a real browser<->server video stream and
calls `video_frame_callback` on every incoming frame *in a background
thread*. We do detection/crop/predict/draw inside that callback (same
logic as the desktop app), and hand results back to the main script
through a thread-safe dict + queue, which the main script polls in a
small loop to update the on-screen widgets.

Run:
    streamlit run streamlit_app.py
Requires sign_model.keras and labels.json (from train_model.py) in the
same folder.
"""
import json
import os
import queue
import threading
import time

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, webrtc_streamer

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

MODEL_PATH = "sign_model.keras"
LABELS_PATH = "labels.json"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.65
CONFIRM_FRAMES = 12
BOX_MARGIN = 0.35

LANDMARK_COLOR = (0, 0, 255)
CONNECTION_COLOR = (255, 255, 255)
BORDER_COLOR = (255, 0, 255)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Hand Sign Language Detector", layout="wide")


# ---------------------------------------------------------------------- #
# Cached model loading (runs once per session)
# ---------------------------------------------------------------------- #
@st.cache_resource
def load_model():
    if not TF_AVAILABLE:
        return None, []
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        return None, []
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_names = json.load(f)
    return model, class_names


model, class_names = load_model()
model_ready = model is not None

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ---------------------------------------------------------------------- #
# Thread-safe state shared between the WebRTC callback thread and the
# main Streamlit thread (never call st.* from inside the callback).
# ---------------------------------------------------------------------- #
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.label = "-"
        self.confidence = 0.0
        self.top3 = []
        self.char_queue = queue.Queue()
        # internal debounce counters (only touched by the callback thread)
        self.stable_label = None
        self.stable_count = 0
        self.awaiting_release = False


if "shared_state" not in st.session_state:
    st.session_state.shared_state = SharedState()
if "sentence" not in st.session_state:
    st.session_state.sentence = ""

shared = st.session_state.shared_state
hands_detector = mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5
)


def process_frame(frame_bgr):
    """Detect hand, draw training-style skeleton overlay, return (display, crop)."""
    h, w, _ = frame_bgr.shape
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    display = frame_bgr.copy()
    crop = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        side = max(x_max - x_min, y_max - y_min) * (1 + BOX_MARGIN)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

        x1 = int(max(cx - side / 2, 0))
        y1 = int(max(cy - side / 2, 0))
        x2 = int(min(cx + side / 2, w))
        y2 = int(min(cy + side / 2, h))

        mp_drawing.draw_landmarks(
            display,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=LANDMARK_COLOR, thickness=-1, circle_radius=5),
            mp_drawing.DrawingSpec(color=CONNECTION_COLOR, thickness=2),
        )
        cv2.rectangle(display, (x1, y1), (x2, y2), BORDER_COLOR, 2)

        if x2 > x1 and y2 > y1:
            crop = display[y1:y2, x1:x2].copy()
            crop = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=BORDER_COLOR)

    return display, crop


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    display, crop = process_frame(img)

    if crop is not None and model_ready:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, IMG_SIZE)
        batch = np.expand_dims(rgb.astype("float32"), axis=0)
        probs = model.predict(batch, verbose=0)[0]
        top_idx = np.argsort(probs)[::-1][:3]
        top_label = class_names[top_idx[0]]
        top_conf = float(probs[top_idx[0]])
        top3 = [(class_names[i], float(probs[i])) for i in top_idx]

        with shared.lock:
            shared.label = top_label if top_conf >= CONFIDENCE_THRESHOLD else "-"
            shared.confidence = top_conf
            shared.top3 = top3

        # debounce / auto-confirm logic (only this thread touches these counters)
        if top_conf >= CONFIDENCE_THRESHOLD:
            if top_label == shared.stable_label:
                shared.stable_count += 1
            else:
                shared.stable_label = top_label
                shared.stable_count = 1

            if shared.stable_count == CONFIRM_FRAMES and not shared.awaiting_release:
                shared.char_queue.put(top_label)
                shared.awaiting_release = True
        else:
            shared.stable_count = 0
            shared.awaiting_release = False
    else:
        with shared.lock:
            shared.label = "-"
            shared.confidence = 0.0
            shared.top3 = []
        shared.stable_count = 0
        shared.awaiting_release = False

    return av.VideoFrame.from_ndarray(display, format="bgr24")


def _render_sentence(placeholder):
    # Plain markdown, not a stateful widget - safe to call every loop tick
    # (a text_area re-created with the same key on every tick would raise
    # a DuplicateWidgetID error).
    text = st.session_state.sentence if st.session_state.sentence else "&nbsp;"
    placeholder.markdown(
        f"<div style='min-height:80px;padding:10px;border:1px solid #444;"
        f"border-radius:8px;font-size:20px;white-space:pre-wrap'>{text}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------- #
# Layout
# ---------------------------------------------------------------------- #
st.title("🖐️ Hand Sign Language Detector")

if not model_ready:
    st.warning(
        "Model not found. Run `python train_model.py` first to create "
        "`sign_model.keras` and `labels.json` in this folder."
    )

col_video, col_result = st.columns([3, 2])

with col_video:
    ctx = webrtc_streamer(
        key="sign-detect",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

with col_result:
    st.subheader("Detected Sign")
    letter_ph = st.empty()
    conf_ph = st.empty()
    top3_ph = st.empty()

    st.subheader("Sentence Builder")
    sentence_ph = st.empty()

    auto_add = st.checkbox("Auto-add held sign", value=True)
    b1, b2, b3 = st.columns(3)
    if b1.button("Space"):
        st.session_state.sentence += " "
    if b2.button("⌫ Backspace"):
        st.session_state.sentence = st.session_state.sentence[:-1]
    if b3.button("Clear"):
        st.session_state.sentence = ""

    status_ph = st.empty()

# ---------------------------------------------------------------------- #
# Polling loop: pull the latest results from the shared state and update
# the placeholders above, without doing a full Streamlit rerun.
# ---------------------------------------------------------------------- #
if ctx.state.playing:
    while ctx.state.playing:
        with shared.lock:
            label = shared.label
            confidence = shared.confidence
            top3 = list(shared.top3)

        letter_ph.markdown(f"<h1 style='font-size:80px'>{label}</h1>", unsafe_allow_html=True)
        conf_ph.progress(min(confidence, 1.0), text=f"Confidence: {confidence * 100:.0f}%")
        if top3:
            top3_text = "  \n".join(f"**{c}** — {p * 100:.0f}%" for c, p in top3)
            top3_ph.markdown(top3_text)
        else:
            top3_ph.markdown("-")

        if auto_add:
            try:
                while True:
                    ch = shared.char_queue.get_nowait()
                    st.session_state.sentence += ch
                    status_ph.info(f"Added '{ch}'")
            except queue.Empty:
                pass
        else:
            # drain the queue so it doesn't build up while auto-add is off
            try:
                while True:
                    shared.char_queue.get_nowait()
            except queue.Empty:
                pass

        _render_sentence(sentence_ph)
        time.sleep(0.15)
else:
    _render_sentence(sentence_ph)
    st.info("Click **START** above and allow camera access to begin.")
