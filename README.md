Hand Sign Language Detector

Real-time hand sign recognition (digits 0-9 and letters A-Z) with a modern desktop GUI, built on your DATASET folder.

About your dataset

Your archive.zip contains DATASET/<label>/*.jpg with 36 classes (0-9, A-Z), 25 images each (900 total), 300x300 RGB. Every image already has a MediaPipe hand-landmark skeleton drawn on top of the hand (red dots + white connections, magenta border) - so it was captured the same way this app will process the live webcam feed. That's why the app re-draws that exact same overlay on the webcam crop before predicting: the model sees the same kind of image at inference time as it saw in training.

1. Setup
bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

Unzip your dataset so you have a DATASET/ folder next to these scripts:

bash
unzip archive.zip        # creates DATASET/0 ... DATASET/Z
2. Train the model
bash
python train_model.py --data DATASET --epochs 15 --fine_tune_epochs 10

This trains a MobileNetV2 transfer-learning model (frozen base first, then fine-tunes the last 30 layers) with rotation/zoom/brightness augmentation - important since you only have 25 images/class. It saves:

sign_model.keras - the trained model
labels.json - class order used by the model
training_curves.png - accuracy/loss plot to sanity-check training

With this dataset size, training is fast even on CPU (a few minutes). Expect validation accuracy to vary since 25 images/class is small - if it's low, collect more images per class (50-100+ is much more robust) using the same capture style before retraining.

3. Run the app
bash
python sign_app.py
Click Start Camera.
Show a hand sign - the app detects your hand, overlays the same skeleton style as the training data, crops it, and classifies it.
The big letter/number, a confidence bar, and the top-3 matches update live.
Sentence Builder: hold a sign steady for about half a second and it's auto-added to the sentence box (toggle this off if you'd rather add manually). Use Space, ⌫, Clear, and 🔊 Speak (text-to-speech) to manage the sentence.
Screenshot saves the current annotated frame into screenshots/.
3b. Run the Streamlit version instead

If you'd rather have a browser-based app (easy to share a link, deployable to Streamlit Community Cloud) instead of the desktop window, use streamlit_app.py. It reuses the exact same detection/crop/predict logic, but streams video through your browser using streamlit-webrtc instead of OpenCV's own window, since Streamlit reruns the whole script on every interaction and can't drive a cv2.VideoCapture loop directly.

Install the extra Streamlit dependencies (already listed in requirements.txt):
bash
   pip install -r requirements.txt
Make sure you've trained the model first (same as above) - you need sign_model.keras and labels.json sitting next to streamlit_app.py.
Launch it:
bash
   streamlit run streamlit_app.py

This opens http://localhost:8501 in your browser automatically. 4. Click START under the video panel and allow camera access when your browser prompts you. 5. Show a sign - the video panel shows the live skeleton overlay, and the right-hand column shows the detected letter/number, a confidence bar, and the top-3 matches, updating a few times a second. 6. Sentence Builder: with "Auto-add held sign" checked, holding a sign steady auto-appends it (same debounce logic as the desktop app). Use the Space / ⌫ Backspace / Clear buttons underneath to edit it. Note: unlike the desktop version there's no built-in text-to-speech button here (browser apps can't easily drive your system's speakers) - you can copy the sentence text out instead. 7. To stop, click STOP under the video, or just close the tab.

A few Streamlit-specific things worth knowing:

The webcam feed runs through a background thread (video_frame_callback) that streamlit-webrtc manages - it's normal for the first frame to take a second or two to connect.
If the camera never connects, it's almost always a browser permission issue - check the camera icon in your address bar. Camera access also requires localhost or HTTPS, which is why local dev with streamlit run works but a plain HTTP deployment won't.
If you deploy this to Streamlit Community Cloud, the STUN server config already in streamlit_app.py (RTC_CONFIGURATION) usually gets you connected; some restrictive corporate/school networks may still need a TURN server, which is a streamlit-webrtc / networking concern, not something in this app's code.
CONFIDENCE_THRESHOLD and CONFIRM_FRAMES at the top of streamlit_app.py control the same debounce behavior described below, independently from the desktop app's copy of those constants.
Notes / tuning
CONFIDENCE_THRESHOLD (default 0.65) and CONFIRM_FRAMES (default 12 frames ≈ 0.4-0.6s) in sign_app.py control how sure/stable a sign must be before it's shown or auto-added - raise them if you get false positives, lower them if it feels unresponsive.
The app deliberately does not mirror-flip signs during training augmentation, since a mirrored hand sign is a different (or wrong) sign.
If your webcam isn't index 0, change cv2.VideoCapture(0) in sign_app.py.
Text-to-speech (pyttsx3) and the model are both optional at import time - the app still runs and tells you what's missing instead of crashing.
