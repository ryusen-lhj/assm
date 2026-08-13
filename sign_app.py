import json
import os
import time
import tkinter as tk
from collections import deque, Counter
from datetime import datetime
 
import cv2
import customtkinter as ctk
import mediapipe as mp
import numpy as np
from PIL import Image
 
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
 
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
 
MODEL_PATH = "sign_model.keras"
LABELS_PATH = "labels.json"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.65
CONFIRM_FRAMES = 12          # consecutive stable frames needed before auto-adding a char
HISTORY_LEN = 15             # frames used for majority-vote smoothing
BOX_MARGIN = 0.35            # extra margin around the hand bounding box
 
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
 
LANDMARK_COLOR = (0, 0, 255)     # red dots (BGR)
CONNECTION_COLOR = (255, 255, 255)  # white lines
BORDER_COLOR = (255, 0, 255)     # magenta border, matches dataset style
 
 
class SignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hand Sign Language Detector")
        self.geometry("1180x680")
        self.minsize(1000, 620)
 
        self.cap = None
        self.camera_running = False
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
 
        self.model = None
        self.class_names = []
        self.model_ready = False
        self._load_model()
 
        self.pred_history = deque(maxlen=HISTORY_LEN)
        self.stable_label = None
        self.stable_count = 0
        self.awaiting_release = False
        self.sentence = ""
        self.auto_add = tk.BooleanVar(value=True)
 
        self.tts_engine = pyttsx3.init() if TTS_AVAILABLE else None
 
        self._build_layout()
        os.makedirs("screenshots", exist_ok=True)
 
    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _load_model(self):
        if not TF_AVAILABLE:
            print("TensorFlow not installed - prediction disabled.")
            return
        if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
            print(f"Model files not found ({MODEL_PATH}, {LABELS_PATH}). "
                  f"Run train_model.py first - prediction disabled.")
            return
        self.model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH) as f:
            self.class_names = json.load(f)
        self.model_ready = True
        print(f"Loaded model with {len(self.class_names)} classes.")
 
    # ------------------------------------------------------------------ #
    # UI layout
    # ------------------------------------------------------------------ #
    def _build_layout(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
 
        # ---------------- Left: video panel ----------------
        left = ctk.CTkFrame(self, corner_radius=12)
        left.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="nsew")
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
 
        self.video_label = ctk.CTkLabel(left, text="Camera is off", corner_radius=12)
        self.video_label.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
 
        controls = ctk.CTkFrame(left, fg_color="transparent")
        controls.grid(row=1, column=0, pady=(0, 12))
        self.start_btn = ctk.CTkButton(controls, text="Start Camera", command=self.toggle_camera, width=140)
        self.start_btn.grid(row=0, column=0, padx=6)
        self.screenshot_btn = ctk.CTkButton(controls, text="Screenshot", command=self.save_screenshot, width=120)
        self.screenshot_btn.grid(row=0, column=1, padx=6)
        self.fps_label = ctk.CTkLabel(controls, text="FPS: --")
        self.fps_label.grid(row=0, column=2, padx=16)
 
        if not self.model_ready:
            warn = ctk.CTkLabel(
                left,
                text="⚠ Model not loaded - run train_model.py first",
                text_color="#e07b39",
            )
            warn.grid(row=2, column=0, pady=(0, 10))
 
        # ---------------- Right: prediction + sentence panel ----------------
        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
 
        ctk.CTkLabel(right, text="Detected Sign", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, pady=(18, 4), sticky="w", padx=18
        )
        self.big_letter = ctk.CTkLabel(
            right, text="-", font=ctk.CTkFont(size=90, weight="bold"), height=120
        )
        self.big_letter.grid(row=1, column=0, pady=4)
 
        self.confidence_bar = ctk.CTkProgressBar(right, width=280)
        self.confidence_bar.set(0)
        self.confidence_bar.grid(row=2, column=0, pady=(6, 2))
        self.confidence_label = ctk.CTkLabel(right, text="Confidence: 0%")
        self.confidence_label.grid(row=3, column=0, pady=(0, 12))
 
        ctk.CTkLabel(right, text="Top matches", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=4, column=0, sticky="w", padx=18
        )
        self.top3_frame = ctk.CTkFrame(right, fg_color="transparent")
        self.top3_frame.grid(row=5, column=0, pady=(4, 16), padx=18, sticky="ew")
        self.top3_labels = []
        for i in range(3):
            lbl = ctk.CTkLabel(self.top3_frame, text="-", anchor="w")
            lbl.grid(row=i, column=0, sticky="ew", pady=2)
            self.top3_labels.append(lbl)
 
        sep = ctk.CTkFrame(right, height=2, fg_color="gray30")
        sep.grid(row=6, column=0, sticky="ew", padx=18, pady=8)
 
        ctk.CTkLabel(right, text="Sentence Builder", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=7, column=0, sticky="w", padx=18, pady=(4, 4)
        )
        self.sentence_box = ctk.CTkTextbox(right, height=90, wrap="word")
        self.sentence_box.grid(row=8, column=0, sticky="ew", padx=18)
 
        self.auto_switch = ctk.CTkSwitch(right, text="Auto-add held sign", variable=self.auto_add)
        self.auto_switch.grid(row=9, column=0, sticky="w", padx=18, pady=(8, 4))
 
        btn_row = ctk.CTkFrame(right, fg_color="transparent")
        btn_row.grid(row=10, column=0, pady=8, padx=18, sticky="ew")
        ctk.CTkButton(btn_row, text="Space", width=70, command=self.add_space).grid(row=0, column=0, padx=4)
        ctk.CTkButton(btn_row, text="⌫", width=50, command=self.backspace).grid(row=0, column=1, padx=4)
        ctk.CTkButton(btn_row, text="Clear", width=70, command=self.clear_sentence).grid(row=0, column=2, padx=4)
        speak_state = "normal" if TTS_AVAILABLE else "disabled"
        ctk.CTkButton(btn_row, text="🔊 Speak", width=90, command=self.speak_sentence,
                      state=speak_state).grid(row=0, column=3, padx=4)
 
        self.status_label = ctk.CTkLabel(right, text="Show a hand sign to the camera", text_color="gray70")
        self.status_label.grid(row=11, column=0, pady=(10, 10))
 
    # ------------------------------------------------------------------ #
    # Camera control
    # ------------------------------------------------------------------ #
    def toggle_camera(self):
        if self.camera_running:
            self.camera_running = False
            self.start_btn.configure(text="Start Camera")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.video_label.configure(image=None, text="Camera is off")
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_label.configure(text="Could not open webcam.")
                return
            self.camera_running = True
            self.start_btn.configure(text="Stop Camera")
            self._last_time = time.time()
            self._frame_loop()
 
    # ------------------------------------------------------------------ #
    # Main per-frame loop
    # ------------------------------------------------------------------ #
    def _frame_loop(self):
        if not self.camera_running or self.cap is None:
            return
 
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            display_frame, crop = self._process_frame(frame)
 
            if crop is not None and self.model_ready:
                self._predict(crop)
            else:
                self._decay_prediction()
 
            self._render(display_frame)
 
        now = time.time()
        fps = 1.0 / max(now - self._last_time, 1e-6)
        self._last_time = now
        self.fps_label.configure(text=f"FPS: {fps:.0f}")
 
        self.after(15, self._frame_loop)
 
    def _process_frame(self, frame):
        """Detect the hand, draw the training-style skeleton, return
        (frame_to_display, cropped_hand_image_for_model)."""
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
 
        display = frame.copy()
        crop = None
 
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
 
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            box_w, box_h = x_max - x_min, y_max - y_min
            side = max(box_w, box_h) * (1 + BOX_MARGIN)
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
 
            x1 = int(max(cx - side / 2, 0))
            y1 = int(max(cy - side / 2, 0))
            x2 = int(min(cx + side / 2, w))
            y2 = int(min(cy + side / 2, h))
 
            # Draw skeleton overlay to match training-image style
            self.mp_drawing.draw_landmarks(
                display,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=LANDMARK_COLOR, thickness=-1, circle_radius=5),
                self.mp_drawing.DrawingSpec(color=CONNECTION_COLOR, thickness=2),
            )
 
            cv2.rectangle(display, (x1, y1), (x2, y2), BORDER_COLOR, 2)
 
            if x2 > x1 and y2 > y1:
                crop = display[y1:y2, x1:x2].copy()
                crop = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=BORDER_COLOR)
 
        return display, crop
 
    # ------------------------------------------------------------------ #
    # Prediction + smoothing
    # ------------------------------------------------------------------ #
    def _predict(self, crop_bgr):
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        batch = np.expand_dims(img.astype("float32"), axis=0)
 
        probs = self.model.predict(batch, verbose=0)[0]
        top_idx = np.argsort(probs)[::-1][:3]
        top_label = self.class_names[top_idx[0]]
        top_conf = float(probs[top_idx[0]])
 
        self.big_letter.configure(text=top_label if top_conf >= CONFIDENCE_THRESHOLD else "-")
        self.confidence_bar.set(top_conf)
        self.confidence_label.configure(text=f"Confidence: {top_conf * 100:.0f}%")
        for i, idx in enumerate(top_idx):
            self.top3_labels[i].configure(text=f"{self.class_names[idx]}   {probs[idx] * 100:.0f}%")
 
        self._update_sentence_logic(top_label, top_conf)
 
    def _decay_prediction(self):
        self.stable_count = 0
        self.awaiting_release = False
        self.big_letter.configure(text="-")
        self.confidence_bar.set(0)
        self.confidence_label.configure(text="Confidence: 0%")
        for lbl in self.top3_labels:
            lbl.configure(text="-")
 
    def _update_sentence_logic(self, label, confidence):
        if confidence < CONFIDENCE_THRESHOLD:
            self.stable_count = 0
            self.awaiting_release = False
            return
 
        if label == self.stable_label:
            self.stable_count += 1
        else:
            self.stable_label = label
            self.stable_count = 1
 
        if not self.auto_add.get():
            return
 
        if self.stable_count == CONFIRM_FRAMES and not self.awaiting_release:
            self._append_char(label)
            self.awaiting_release = True  # require a new hold before adding the same char again
            self.status_label.configure(text=f"Added '{label}' - move hand away/change sign to add another")
 
    def _append_char(self, char):
        self.sentence += char
        self.sentence_box.delete("1.0", "end")
        self.sentence_box.insert("1.0", self.sentence)
 
    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        w, h = pil_img.size
        target_w = 720
        target_h = int(h * target_w / w)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))
        self.video_label.configure(image=ctk_img, text="")
        self.video_label.image = ctk_img
 
    # ------------------------------------------------------------------ #
    # Buttons
    # ------------------------------------------------------------------ #
    def add_space(self):
        self.sentence += " "
        self.sentence_box.delete("1.0", "end")
        self.sentence_box.insert("1.0", self.sentence)
 
    def backspace(self):
        self.sentence = self.sentence[:-1]
        self.sentence_box.delete("1.0", "end")
        self.sentence_box.insert("1.0", self.sentence)
 
    def clear_sentence(self):
        self.sentence = ""
        self.sentence_box.delete("1.0", "end")
 
    def speak_sentence(self):
        if not TTS_AVAILABLE or not self.sentence.strip():
            return
        self.tts_engine.say(self.sentence)
        self.tts_engine.runAndWait()
 
    def save_screenshot(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            display, _ = self._process_frame(frame)
            fname = f"screenshots/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(fname, display)
            self.status_label.configure(text=f"Saved {fname}")
 
    def on_close(self):
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
        self.hands.close()
        self.destroy()
 
 
if __name__ == "__main__":
    app = SignLanguageApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
 
