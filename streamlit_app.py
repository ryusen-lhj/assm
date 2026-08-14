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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from streamlit_webrtc import webrtc_streamer
from twilio.rest import Client

st.set_page_config(page_title="SIGNOVA", page_icon="🤟", layout="wide")

ZIP_NAMES = ["archive.zip", "archive(1).zip"]
EXTRACT_DIR = "signova_dataset"
CLASSES = [str(i) for i in range(10)] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
TOTAL_IMAGES = 900
IMAGES_PER_CLASS = 25

CONFIDENCE_THRESHOLD = 0.55
STABLE_FRAMES = 7
NO_HAND_REARM_FRAMES = 4
RECORD_COOLDOWN = 0.6
TRAJECTORY_LENGTH = 18

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

mp_hands = mp.solutions.hands

st.markdown("""
<style>
.stApp {
    background:
      radial-gradient(circle at 10% 5%, rgba(124,58,237,.16), transparent 30%),
      radial-gradient(circle at 90% 10%, rgba(14,165,233,.10), transparent 30%),
      #070a12;
    color: #f8fafc;
}
[data-testid="stSidebar"] {
    background: #0b0f1a;
    border-right: 1px solid rgba(255,255,255,.07);
}
.hero {font-size:50px;font-weight:950;letter-spacing:-2px;}
.sub {color:#94a3b8;line-height:1.7;max-width:850px;}
.card {
    background:rgba(15,23,42,.78);
    border:1px solid rgba(255,255,255,.08);
    border-radius:18px;
    padding:18px;
}
.good {color:#4ade80;font-weight:900;}
.warn {color:#facc15;font-weight:800;}
.muted {color:#94a3b8;}
</style>
""", unsafe_allow_html=True)

def hero(title, subtitle):
    st.markdown(f'<div class="hero">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub">{subtitle}</div>', unsafe_allow_html=True)


def _secret(name):
    try:
        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


def _urls_from_ice_server(server):
    if not isinstance(server, dict):
        return []
    urls = server.get("urls", server.get("url", []))
    if isinstance(urls, str):
        return [urls]
    if isinstance(urls, (list, tuple)):
        return [str(url) for url in urls]
    return []


@st.cache_data(ttl=1800, show_spinner=False)
def get_turn_config(account_sid, auth_token):
    token = Client(account_sid, auth_token).tokens.create()
    ice_servers = token.ice_servers

    # Streamlit Community Cloud and this client network have already shown that
    # direct/STUN-only ICE paths fail. Force the browser to use relay candidates
    # only, so it must connect through Twilio TURN instead of retrying blocked
    # host/server-reflexive candidates.
    return {
        "iceServers": ice_servers,
        "iceTransportPolicy": "relay",
    }


def require_turn_config():
    sid = _secret("TWILIO_ACCOUNT_SID")
    auth = _secret("TWILIO_AUTH_TOKEN")

    if not sid or not auth:
        st.error(
            "Twilio TURN is not configured. Add TWILIO_ACCOUNT_SID and "
            "TWILIO_AUTH_TOKEN in Streamlit → Manage app → Secrets, then reboot."
        )
        st.stop()

    try:
        config = get_turn_config(sid, auth)
    except Exception as exc:
        st.error(
            "Twilio TURN token creation failed. Check your Streamlit Secrets. "
            f"{type(exc).__name__}: {str(exc)[:240]}"
        )
        st.stop()

    ice_servers = config.get("iceServers") or []
    if not ice_servers:
        st.error("Twilio returned no ICE servers.")
        st.stop()

    turn_urls = []
    for server in ice_servers:
        for url in _urls_from_ice_server(server):
            if url.lower().startswith(("turn:", "turns:")):
                turn_urls.append(url)

    if not turn_urls:
        st.error(
            "Twilio credentials are valid, but the token contained no TURN relay "
            "endpoints. WebRTC cannot start on this network."
        )
        st.stop()

    st.caption(
        f"TURN relay ready: {len(turn_urls)} relay endpoint(s). "
        "WebRTC is forced to relay-only mode."
    )
    return config


def find_zip():
    for name in ZIP_NAMES:
        if os.path.exists(name):
            return name
    return None


def locate_dataset():
    direct = os.path.join(EXTRACT_DIR, "DATASET")
    if os.path.isdir(direct):
        return direct
    if not os.path.isdir(EXTRACT_DIR):
        return None
    for root, _, _ in os.walk(EXTRACT_DIR):
        if os.path.basename(root).upper() == "DATASET":
            return root
    return None


def prepare_dataset():
    existing = locate_dataset()
    if existing:
        return existing

    zip_name = find_zip()
    if not zip_name:
        st.error("archive.zip is missing from the repository.")
        st.stop()

    os.makedirs(EXTRACT_DIR, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_name, "r") as archive:
            archive.extractall(EXTRACT_DIR)
    except zipfile.BadZipFile:
        st.error("archive.zip is invalid.")
        st.stop()

    dataset = locate_dataset()
    if not dataset:
        st.error("Could not find DATASET/<class>/... inside archive.zip.")
        st.stop()
    return dataset


def features_from_landmarks(landmarks):
    pts = np.asarray([[p.x, p.y] for p in landmarks], dtype=np.float32)
    pts -= pts[0].copy()

    scale = float(np.max(np.linalg.norm(pts, axis=1)))
    if scale > 1e-6:
        pts /= scale

    if pts[5, 0] > pts[17, 0]:
        pts[:, 0] *= -1.0

    axis = pts[9]
    angle = np.arctan2(axis[1], axis[0]) - (-np.pi / 2.0)
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    pts = pts @ rot.T

    vectors = []
    lengths = []
    for a, b in HAND_CONNECTIONS:
        v = pts[b] - pts[a]
        vectors.extend(v.tolist())
        lengths.append(float(np.linalg.norm(v)))

    angle_triples = [
        (1,2,3),(2,3,4),
        (5,6,7),(6,7,8),
        (9,10,11),(10,11,12),
        (13,14,15),(14,15,16),
        (17,18,19),(18,19,20),
    ]
    angles = []
    for a, b, cidx in angle_triples:
        u = pts[a] - pts[b]
        v = pts[cidx] - pts[b]
        denom = float(np.linalg.norm(u) * np.linalg.norm(v))
        if denom < 1e-7:
            angles.append(0.0)
        else:
            cosine = float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))
            angles.append(float(np.arccos(cosine) / np.pi))

    tips = [4, 8, 12, 16, 20]
    distances = [float(np.linalg.norm(pts[t])) for t in tips]
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            distances.append(float(np.linalg.norm(pts[tips[i]] - pts[tips[j]])))

    return np.concatenate([
        pts.flatten(),
        np.asarray(vectors, dtype=np.float32),
        np.asarray(lengths, dtype=np.float32),
        np.asarray(angles, dtype=np.float32),
        np.asarray(distances, dtype=np.float32),
    ]).astype(np.float32)


@st.cache_data(show_spinner=False)
def load_dataset_landmarks(dataset_path):
    X, y = [], []
    detected = {label: 0 for label in CLASSES}
    failed = {label: 0 for label in CLASSES}

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.35,
    ) as detector:
        for label in CLASSES:
            folder = os.path.join(dataset_path, label)
            if not os.path.isdir(folder):
                continue

            files = sorted(
                f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )

            for filename in files:
                try:
                    img = Image.open(os.path.join(folder, filename)).convert("RGB")
                    result = detector.process(np.asarray(img))
                    if not result.multi_hand_landmarks:
                        failed[label] += 1
                        continue
                    lm = result.multi_hand_landmarks[0].landmark
                    X.append(features_from_landmarks(lm))
                    y.append(label)
                    detected[label] += 1
                except Exception:
                    failed[label] += 1

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y),
        detected,
        failed,
    )


@st.cache_resource(show_spinner=False)
def train_model(X, y):
    if len(X) == 0:
        raise ValueError("No usable landmark samples.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=8.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    model.fit(X, y)

    unique, counts = np.unique(y, return_counts=True)
    eligible = unique[counts >= 2]
    if len(eligible) < 2:
        return model, None, 0, 0

    rng = np.random.default_rng(42)
    train_idx, test_idx = [], []
    for label in eligible:
        idx = np.where(y == label)[0].copy()
        rng.shuffle(idx)
        n_test = min(max(1, int(round(len(idx) * 0.2))), len(idx) - 1)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    if not train_idx or not test_idx:
        return model, None, 0, 0

    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    eval_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=8.0,
            gamma="scale",
            probability=False,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    eval_model.fit(X[train_idx], y[train_idx])
    pred = eval_model.predict(X[test_idx])
    return model, float(accuracy_score(y[test_idx], pred)), len(train_idx), len(test_idx)


class RecognitionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.sentence = ""
        self.prediction = "-"
        self.confidence = 0.0
        self.hand_detected = False
        self.stable_prediction = None
        self.stable_count = 0
        self.last_recorded = None
        self.last_record_time = 0.0
        self.no_hand_count = 0


if "signova_state" not in st.session_state:
    st.session_state.signova_state = RecognitionState()

state = st.session_state.signova_state


def draw_overlay(frame_rgb, landmarks, prediction, confidence, trajectory, sentence):
    image = Image.fromarray(frame_rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    left = max(min(xs) - 24, 0)
    top = max(min(ys) - 24, 0)
    right = min(max(xs) + 24, width - 1)
    bottom = min(max(ys) + 24, height - 1)

    draw.rectangle([left, top, right, bottom], outline=(0,255,80), width=4)

    for a, b in HAND_CONNECTIONS:
        draw.line([points[a], points[b]], fill=(255,255,255), width=3)

    for x, y in points:
        r = 5
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(255,30,30), outline=(255,255,255))

    history = list(trajectory)
    for i in range(1, len(history)):
        draw.line([history[i-1], history[i]], fill=(0,220,255), width=3)

    draw.rounded_rectangle([12, 12, 320, 120], radius=12,
                           fill=(8,15,30), outline=(0,255,80), width=3)
    draw.text((26, 28), f"SIGN: {prediction}", fill=(255,255,255))
    draw.text((26, 56), f"CONFIDENCE: {confidence*100:.1f}%", fill=(110,255,150))
    shown_sentence = sentence[-22:] if sentence else "-"
    draw.text((26, 84), f"TEXT: {shown_sentence}", fill=(180,190,255))

    return np.asarray(image)


class SignProcessor:
    def __init__(self, classifier, shared_state):
        self.model = classifier
        self.state = shared_state
        self.lock = threading.Lock()
        self.trajectory = deque(maxlen=TRAJECTORY_LENGTH)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )

    def __call__(self, frame):
        with self.lock:
            rgb = frame.to_ndarray(format="rgb24")
            result = self.hands.process(rgb)

            if not result.multi_hand_landmarks:
                self.trajectory.clear()
                with self.state.lock:
                    self.state.hand_detected = False
                    self.state.prediction = "-"
                    self.state.confidence = 0.0
                    self.state.stable_prediction = None
                    self.state.stable_count = 0
                    self.state.no_hand_count += 1
                    if self.state.no_hand_count >= NO_HAND_REARM_FRAMES:
                        self.state.last_recorded = None
                    sentence = self.state.sentence
                image = Image.fromarray(rgb).convert("RGB")
                draw = ImageDraw.Draw(image)
                draw.rounded_rectangle([12,12,320,84], radius=12,
                                       fill=(8,15,30), outline=(100,116,139), width=2)
                draw.text((26,28), "WAITING FOR HAND", fill=(200,210,220))
                draw.text((26,56), f"TEXT: {sentence[-22:] if sentence else '-'}",
                          fill=(180,190,255))
                return av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")

            landmarks = result.multi_hand_landmarks[0].landmark
            feature = features_from_landmarks(landmarks).reshape(1, -1)
            probs = self.model.predict_proba(feature)[0]
            best = int(np.argmax(probs))
            prediction = str(self.model.classes_[best])
            confidence = float(probs[best])

            h, w, _ = rgb.shape
            tip = landmarks[8]
            self.trajectory.append((int(tip.x*w), int(tip.y*h)))

            now = time.time()
            with self.state.lock:
                self.state.hand_detected = True
                self.state.no_hand_count = 0
                self.state.prediction = prediction
                self.state.confidence = confidence

                if self.state.stable_prediction == prediction:
                    self.state.stable_count += 1
                else:
                    self.state.stable_prediction = prediction
                    self.state.stable_count = 1

                should_record = (
                    confidence >= CONFIDENCE_THRESHOLD
                    and self.state.stable_count >= STABLE_FRAMES
                    and self.state.last_recorded != prediction
                    and (now - self.state.last_record_time) >= RECORD_COOLDOWN
                )
                if should_record:
                    self.state.sentence += prediction
                    self.state.last_recorded = prediction
                    self.state.last_record_time = now

                sentence = self.state.sentence

            out = draw_overlay(
                rgb, landmarks, prediction, confidence, self.trajectory, sentence
            )
            return av.VideoFrame.from_ndarray(out, format="rgb24")


dataset_path = prepare_dataset()

with st.spinner("Reading dataset hand landmarks..."):
    X, y, detected_counts, failed_counts = load_dataset_landmarks(dataset_path)

if len(X) == 0:
    st.error("MediaPipe could not detect any usable training samples.")
    st.stop()

with st.spinner("Training SIGNOVA recognition model..."):
    model, accuracy, eval_train, eval_test = train_model(X, y)

available_classes = sorted(np.unique(y).tolist())
extraction_rate = len(X) / TOTAL_IMAGES * 100.0
accuracy_text = "N/A" if accuracy is None else f"{accuracy*100:.1f}%"

with st.sidebar:
    st.markdown("## 🤟 SIGNOVA")
    st.caption("Real-Time Static Hand-Sign Recognition")
    page = st.radio(
        "Navigation",
        ["🎥 Live Translator", "📊 Dataset Lab", "🧠 Model Insights", "ℹ️ About"],
    )
    st.markdown("---")
    st.caption(f"{len(available_classes)} / 36 classes usable")
    st.caption(f"{len(X)} landmark samples")
    st.caption(f"{extraction_rate:.1f}% extraction rate")

if page == "🎥 Live Translator":
    hero(
        "SIGNOVA",
        "Live static hand-sign recognition with 21 MediaPipe landmarks, "
        "2D geometric features, RBF-SVM classification, and sentence construction.",
    )

    with state.lock:
        current_sentence = state.sentence

    st.markdown(
        f'<div class="card"><b>Detected sequence:</b><br><span style="font-size:30px">'
        f'{current_sentence if current_sentence else "—"}</span></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, _ = st.columns([1,1,1,3])
    with c1:
        if st.button("🗑 Clear", use_container_width=True):
            with state.lock:
                state.sentence = ""
                state.last_recorded = None
            st.rerun()
    with c2:
        if st.button("⌫ Delete", use_container_width=True):
            with state.lock:
                state.sentence = state.sentence[:-1]
                state.last_recorded = None
            st.rerun()
    with c3:
        if st.button("␣ Space", use_container_width=True):
            with state.lock:
                if state.sentence and not state.sentence.endswith(" "):
                    state.sentence += " "
                state.last_recorded = None
            st.rerun()

    st.markdown("### Live camera")
    st.caption(
        "Click START. Your native camera test already passed, so this connection "
        "is forced through Twilio TURN relay-only mode."
    )

    rtc_config = require_turn_config()
    processor = SignProcessor(model, state)

    ctx = webrtc_streamer(
        key="signova-live",
        video_frame_callback=processor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        media_toggle_controls=True,
    )

    if ctx.state.playing:
        st.success("Camera stream is connected through WebRTC.")
    else:
        st.info(
            "Native camera access is confirmed. If START still cannot connect, "
            "the remaining failure is the TURN relay path rather than camera permission."
        )

elif page == "📊 Dataset Lab":
    hero("Dataset Lab", "MediaPipe landmark extraction results for your 900 images.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Images", TOTAL_IMAGES)
    c2.metric("Classes", 36)
    c3.metric("Usable Samples", len(X))
    c4.metric("Extraction Rate", f"{extraction_rate:.1f}%")

    rows = []
    for label in CLASSES:
        detected = int(detected_counts.get(label, 0))
        failed = int(failed_counts.get(label, 0))
        rows.append({
            "Sign": label,
            "Images": IMAGES_PER_CLASS,
            "Detected": detected,
            "Failed": failed,
            "Detection Rate": f"{detected/IMAGES_PER_CLASS*100:.1f}%",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

elif page == "🧠 Model Insights":
    hero("Model Insights", "The live classifier uses normalized 2D hand geometry.")
    a, b, c, d = st.columns(4)
    a.metric("Classifier", "RBF-SVM")
    b.metric("Usable Classes", len(available_classes))
    c.metric("Evaluation Samples", eval_test)
    d.metric("Accuracy", accuracy_text)
    st.markdown(
        """
        <div class="card">
        <b>Features</b><br><br>
        • 21 X/Y hand landmarks<br>
        • wrist-relative position<br>
        • scale normalization<br>
        • mirror normalization<br>
        • palm-axis rotation normalization<br>
        • joint vectors and bone lengths<br>
        • finger bend angles<br>
        • fingertip distances
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    hero(
        "About SIGNOVA",
        "A real-time static hand-sign recognition and sentence construction system.",
    )
    st.markdown(
        """
        <div class="card">
        SIGNOVA detects 21 MediaPipe hand landmarks from the live webcam,
        converts them into normalized geometric features, classifies the hand
        shape with an RBF-SVM trained from the supplied 36-class dataset, and
        records stable predictions into a text sequence.
        <br><br>
        This is static hand-sign recognition, not full motion- and grammar-aware
        natural-language sign-language translation.
        </div>
        """,
        unsafe_allow_html=True,
    )
