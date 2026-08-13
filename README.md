# Hand Sign Language Detector

Real-time hand sign recognition (digits 0-9 and letters A-Z) with a modern
desktop GUI, built on your `DATASET` folder.

## About your dataset

Your `archive.zip` contains `DATASET/<label>/*.jpg` with **36 classes**
(`0-9`, `A-Z`), **25 images each** (900 total), 300x300 RGB. Every image
already has a MediaPipe hand-landmark skeleton drawn on top of the hand
(red dots + white connections, magenta border) - so it was captured the
same way this app will process the live webcam feed. That's why the app
re-draws that exact same overlay on the webcam crop before predicting:
the model sees the same kind of image at inference time as it saw in
training.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Unzip your dataset so you have a `DATASET/` folder next to these scripts:

```bash
unzip archive.zip        # creates DATASET/0 ... DATASET/Z
```

## 2. Train the model

```bash
python train_model.py --data DATASET --epochs 15 --fine_tune_epochs 10
```

This trains a MobileNetV2 transfer-learning model (frozen base first,
then fine-tunes the last 30 layers) with rotation/zoom/brightness
augmentation - important since you only have 25 images/class. It saves:

- `sign_model.keras` - the trained model
- `labels.json` - class order used by the model
- `training_curves.png` - accuracy/loss plot to sanity-check training

With this dataset size, training is fast even on CPU (a few minutes).
Expect validation accuracy to vary since 25 images/class is small -
if it's low, collect more images per class (50-100+ is much more robust)
using the same capture style before retraining.

## 3. Run the app

```bash
python sign_app.py
```

- Click **Start Camera**.
- Show a hand sign - the app detects your hand, overlays the same
  skeleton style as the training data, crops it, and classifies it.
- The big letter/number, a confidence bar, and the top-3 matches update
  live.
- **Sentence Builder**: hold a sign steady for about half a second and
  it's auto-added to the sentence box (toggle this off if you'd rather
  add manually). Use **Space**, **⌫**, **Clear**, and **🔊 Speak**
  (text-to-speech) to manage the sentence.
- **Screenshot** saves the current annotated frame into `screenshots/`.

## Notes / tuning

- `CONFIDENCE_THRESHOLD` (default 0.65) and `CONFIRM_FRAMES` (default 12
  frames ≈ 0.4-0.6s) in `sign_app.py` control how sure/stable a sign must
  be before it's shown or auto-added - raise them if you get false
  positives, lower them if it feels unresponsive.
- The app deliberately does **not** mirror-flip signs during training
  augmentation, since a mirrored hand sign is a different (or wrong) sign.
- If your webcam isn't index `0`, change `cv2.VideoCapture(0)` in
  `sign_app.py`.
- Text-to-speech (`pyttsx3`) and the model are both optional at import
  time - the app still runs and tells you what's missing instead of
  crashing.
