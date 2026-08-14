import json
import os
import zipfile

import mediapipe as mp
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.set_page_config(page_title="SIGNOVA", page_icon="🤟", layout="wide")

ZIP_NAMES = ["archive.zip", "archive(1).zip"]
EXTRACT_DIR = "signova_dataset"
CLASSES = [str(i) for i in range(10)] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
TOTAL_IMAGES = 900
IMAGES_PER_CLASS = 25

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
.sub {color:#94a3b8;line-height:1.7;max-width:900px;}
.card {
    background:rgba(15,23,42,.78);
    border:1px solid rgba(255,255,255,.08);
    border-radius:18px;
    padding:18px;
}
</style>
""", unsafe_allow_html=True)

def hero(title, subtitle):
    st.markdown(f'<div class="hero">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub">{subtitle}</div>', unsafe_allow_html=True)

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
        (1,2,3),(2,3,4),(5,6,7),(6,7,8),(9,10,11),
        (10,11,12),(13,14,15),(14,15,16),(17,18,19),(18,19,20),
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
    return np.asarray(X, dtype=np.float32), np.asarray(y), detected, failed

@st.cache_resource(show_spinner=False)
def train_eval_model(X, y):
    if len(X) == 0:
        raise ValueError("No usable landmark samples.")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=8.0, gamma="scale", probability=True,
                    class_weight="balanced", random_state=42)),
    ])
    model.fit(X, y)
    unique, counts = np.unique(y, return_counts=True)
    eligible = unique[counts >= 2]
    if len(eligible) < 2:
        return model, None, 0
    rng = np.random.default_rng(42)
    train_idx, test_idx = [], []
    for label in eligible:
        idx = np.where(y == label)[0].copy()
        rng.shuffle(idx)
        n_test = min(max(1, int(round(len(idx) * 0.2))), len(idx) - 1)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    if not train_idx or not test_idx:
        return model, None, 0
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    eval_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=8.0, gamma="scale", probability=False,
                    class_weight="balanced", random_state=42)),
    ])
    eval_model.fit(X[train_idx], y[train_idx])
    pred = eval_model.predict(X[test_idx])
    return model, float(accuracy_score(y[test_idx], pred)), len(test_idx)

@st.cache_data(show_spinner=False)
def browser_training_payload(X, y):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Z = (X - mean) / std
    return {
        "mean": np.round(mean, 6).tolist(),
        "std": np.round(std, 6).tolist(),
        "x": np.round(Z, 5).tolist(),
        "y": y.tolist(),
    }

def browser_camera_component(payload):
    training_json = json.dumps(payload, separators=(",", ":"))
    connections_json = json.dumps(HAND_CONNECTIONS, separators=(",", ":"))
    html = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { box-sizing:border-box; }
body { margin:0; padding:0; background:#070a12; color:#f8fafc; font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
.shell { border:1px solid rgba(255,255,255,.09); border-radius:18px; background:rgba(15,23,42,.84); padding:16px; }
.top { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }
button { border:0; border-radius:10px; padding:10px 16px; font-weight:800; background:#4f46e5; color:#fff; cursor:pointer; }
button.secondary { background:#1e293b; }
button.danger { background:#7f1d1d; }
.status { color:#94a3b8; font-size:13px; }
.sequence { margin:10px 0 14px; padding:14px 16px; border-radius:14px; background:#0b1220; border:1px solid rgba(129,140,248,.25); }
.sequence .label { color:#818cf8; font-size:10px; letter-spacing:1.5px; font-weight:900; }
.sequence .text { font-size:28px; font-weight:900; letter-spacing:3px; min-height:38px; }
.wrap { position:relative; width:100%; max-width:760px; margin:auto; }
video { display:none; }
canvas { width:100%; border-radius:14px; background:#020617; display:block; }
.metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:12px; }
.metric { background:#0b1220; border-radius:12px; padding:12px; border:1px solid rgba(255,255,255,.06); }
.metric b { font-size:22px; display:block; }
.metric span { color:#94a3b8; font-size:11px; }
.note { margin-top:10px; color:#94a3b8; font-size:12px; line-height:1.5; }
.good { color:#4ade80; }
.bad { color:#fb7185; }
</style>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
</head>
<body>
<div class="shell">
  <div class="top">
    <button id="startBtn">▶ START CAMERA</button>
    <button id="clearBtn" class="danger">Clear</button>
    <button id="deleteBtn" class="secondary">Delete</button>
    <button id="spaceBtn" class="secondary">Space</button>
    <span id="status" class="status">Camera stopped.</span>
  </div>
  <div class="sequence"><div class="label">DETECTED SEQUENCE</div><div id="sequence" class="text">—</div></div>
  <div class="wrap"><video id="video" playsinline muted></video><canvas id="canvas" width="640" height="480"></canvas></div>
  <div class="metrics">
    <div class="metric"><b id="pred">-</b><span>Prediction</span></div>
    <div class="metric"><b id="conf">0%</b><span>Confidence</span></div>
    <div class="metric"><b id="stable">0/8</b><span>Stability</span></div>
  </div>
  <div class="note">This camera runs directly in your browser. No WebRTC, STUN, TURN, or aiortc is used.</div>
</div>
<script>
const TRAIN = __TRAINING__;
const CONNECTIONS = __CONNECTIONS__;
const video = document.getElementById("video"), canvas = document.getElementById("canvas"), ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status"), predEl = document.getElementById("pred"), confEl = document.getElementById("conf"), stableEl = document.getElementById("stable"), seqEl = document.getElementById("sequence"), startBtn = document.getElementById("startBtn");
let stream=null, running=false, busy=false, sentence="", stableLabel=null, stableCount=0, lastRecorded=null, noHandFrames=0, trajectory=[];
const STABLE_FRAMES=8, MIN_CONF=0.48;
function updateSequence(){ seqEl.textContent=sentence||"—"; }
document.getElementById("clearBtn").onclick=()=>{sentence="";lastRecorded=null;stableLabel=null;stableCount=0;updateSequence();};
document.getElementById("deleteBtn").onclick=()=>{sentence=sentence.slice(0,-1);lastRecorded=null;updateSequence();};
document.getElementById("spaceBtn").onclick=()=>{if(sentence&&!sentence.endsWith(" "))sentence+=" ";lastRecorded=null;updateSequence();};
function features(lm){
  let pts=lm.map(p=>[p.x,p.y]); const wx=pts[0][0],wy=pts[0][1]; pts=pts.map(([x,y])=>[x-wx,y-wy]);
  let scale=0; for(const [x,y] of pts)scale=Math.max(scale,Math.hypot(x,y)); if(scale>1e-6)pts=pts.map(([x,y])=>[x/scale,y/scale]);
  if(pts[5][0]>pts[17][0])pts=pts.map(([x,y])=>[-x,y]);
  const axis=pts[9],angle=Math.atan2(axis[1],axis[0])-(-Math.PI/2),c=Math.cos(-angle),s=Math.sin(-angle); pts=pts.map(([x,y])=>[x*c-y*s,x*s+y*c]);
  const out=[]; for(const p of pts)out.push(p[0],p[1]); const lengths=[];
  for(const [a,b] of CONNECTIONS){const vx=pts[b][0]-pts[a][0],vy=pts[b][1]-pts[a][1];out.push(vx,vy);lengths.push(Math.hypot(vx,vy));} out.push(...lengths);
  const triples=[[1,2,3],[2,3,4],[5,6,7],[6,7,8],[9,10,11],[10,11,12],[13,14,15],[14,15,16],[17,18,19],[18,19,20]];
  for(const [a,b,cidx] of triples){const ux=pts[a][0]-pts[b][0],uy=pts[a][1]-pts[b][1],vx=pts[cidx][0]-pts[b][0],vy=pts[cidx][1]-pts[b][1],denom=Math.hypot(ux,uy)*Math.hypot(vx,vy);if(denom<1e-7)out.push(0);else{let cos=(ux*vx+uy*vy)/denom;cos=Math.max(-1,Math.min(1,cos));out.push(Math.acos(cos)/Math.PI);}}
  const tips=[4,8,12,16,20]; for(const t of tips)out.push(Math.hypot(pts[t][0],pts[t][1]));
  for(let i=0;i<tips.length;i++)for(let j=i+1;j<tips.length;j++){const a=tips[i],b=tips[j];out.push(Math.hypot(pts[a][0]-pts[b][0],pts[a][1]-pts[b][1]));}
  return out;
}
function classify(raw){
  if(raw.length!==TRAIN.mean.length)return["-",0]; const z=raw.map((v,i)=>(v-TRAIN.mean[i])/TRAIN.std[i]); const best=[];
  for(let r=0;r<TRAIN.x.length;r++){const row=TRAIN.x[r];let d=0;for(let i=0;i<z.length;i++){const diff=z[i]-row[i];d+=diff*diff;}if(best.length<7){best.push([d,r]);best.sort((a,b)=>a[0]-b[0]);}else if(d<best[6][0]){best[6]=[d,r];best.sort((a,b)=>a[0]-b[0]);}}
  const scores={};let total=0;for(const [d,idx] of best){const w=1/(Math.sqrt(d)+0.2),label=TRAIN.y[idx];scores[label]=(scores[label]||0)+w;total+=w;}let label="-",score=0;for(const[k,v]of Object.entries(scores)){if(v>score){score=v;label=k;}}return[label,total>0?score/total:0];
}
function drawLandmarks(lm,label,conf){
  ctx.drawImage(video,0,0,canvas.width,canvas.height);const pts=lm.map(p=>[p.x*canvas.width,p.y*canvas.height]),xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
  const left=Math.max(Math.min(...xs)-24,0),top=Math.max(Math.min(...ys)-24,0),right=Math.min(Math.max(...xs)+24,canvas.width),bottom=Math.min(Math.max(...ys)+24,canvas.height);
  ctx.strokeStyle="#00ff50";ctx.lineWidth=4;ctx.strokeRect(left,top,right-left,bottom-top);ctx.strokeStyle="#ffffff";ctx.lineWidth=3;
  for(const[a,b]of CONNECTIONS){ctx.beginPath();ctx.moveTo(...pts[a]);ctx.lineTo(...pts[b]);ctx.stroke();}
  for(const[x,y]of pts){ctx.beginPath();ctx.fillStyle="#ff2020";ctx.arc(x,y,5,0,Math.PI*2);ctx.fill();}
  trajectory.push(pts[8]);if(trajectory.length>18)trajectory.shift();ctx.strokeStyle="#00dcff";ctx.lineWidth=3;if(trajectory.length>1){ctx.beginPath();ctx.moveTo(...trajectory[0]);for(let i=1;i<trajectory.length;i++)ctx.lineTo(...trajectory[i]);ctx.stroke();}
  ctx.fillStyle="rgba(8,15,30,.9)";ctx.fillRect(12,12,290,92);ctx.strokeStyle="#00ff50";ctx.lineWidth=2;ctx.strokeRect(12,12,290,92);ctx.fillStyle="#fff";ctx.font="bold 22px sans-serif";ctx.fillText("SIGN: "+label,26,42);ctx.fillStyle="#70ff96";ctx.font="16px sans-serif";ctx.fillText("CONFIDENCE: "+(conf*100).toFixed(1)+"%",26,70);ctx.fillStyle="#b8beff";ctx.fillText("TEXT: "+(sentence.slice(-20)||"-"),26,94);
}
function drawNoHand(){ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.fillStyle="rgba(8,15,30,.9)";ctx.fillRect(12,12,260,64);ctx.fillStyle="#d0d8e0";ctx.font="bold 20px sans-serif";ctx.fillText("WAITING FOR HAND",26,50);}
const hands=new Hands({locateFile:(file)=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
hands.setOptions({maxNumHands:1,modelComplexity:1,minDetectionConfidence:0.55,minTrackingConfidence:0.55});
hands.onResults((results)=>{busy=false;if(!running)return;if(!results.multiHandLandmarks||!results.multiHandLandmarks.length){noHandFrames++;trajectory=[];stableLabel=null;stableCount=0;predEl.textContent="-";confEl.textContent="0%";stableEl.textContent="0/8";if(noHandFrames>=5)lastRecorded=null;drawNoHand();return;}noHandFrames=0;const lm=results.multiHandLandmarks[0];const[label,conf]=classify(features(lm));if(stableLabel===label)stableCount++;else{stableLabel=label;stableCount=1;}if(conf>=MIN_CONF&&stableCount>=STABLE_FRAMES&&lastRecorded!==label){sentence+=label;lastRecorded=label;updateSequence();}predEl.textContent=label;confEl.textContent=(conf*100).toFixed(1)+"%";stableEl.textContent=Math.min(stableCount,STABLE_FRAMES)+"/"+STABLE_FRAMES;drawLandmarks(lm,label,conf);});
async function loop(){if(!running)return;if(!busy&&video.readyState>=2){busy=true;try{await hands.send({image:video});}catch(e){busy=false;statusEl.textContent="MediaPipe error: "+e.message;}}requestAnimationFrame(loop);}
async function startCamera(){if(running)return;try{statusEl.textContent="Requesting camera...";stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},facingMode:"user"},audio:false});video.srcObject=stream;await video.play();running=true;startBtn.textContent="■ STOP CAMERA";startBtn.onclick=stopCamera;statusEl.innerHTML='<span class="good">Camera connected directly in browser.</span>';requestAnimationFrame(loop);}catch(e){statusEl.innerHTML='<span class="bad">Camera error: '+e.name+' — '+e.message+'</span>';}}
function stopCamera(){running=false;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;startBtn.textContent="▶ START CAMERA";startBtn.onclick=startCamera;statusEl.textContent="Camera stopped.";ctx.clearRect(0,0,canvas.width,canvas.height);}
startBtn.onclick=startCamera;updateSequence();
</script>
</body>
</html>
"""
    html = html.replace("__TRAINING__", training_json)
    html = html.replace("__CONNECTIONS__", connections_json)
    components.html(html, height=790, scrolling=False)

dataset_path = prepare_dataset()
with st.spinner("Reading dataset hand landmarks..."):
    X, y, detected_counts, failed_counts = load_dataset_landmarks(dataset_path)
if len(X) == 0:
    st.error("MediaPipe could not detect any usable training samples.")
    st.stop()
with st.spinner("Preparing SIGNOVA model..."):
    model, accuracy, eval_test = train_eval_model(X, y)
    browser_payload = browser_training_payload(X, y)
available_classes = sorted(np.unique(y).tolist())
extraction_rate = len(X) / TOTAL_IMAGES * 100.0
accuracy_text = "N/A" if accuracy is None else f"{accuracy*100:.1f}%"

with st.sidebar:
    st.markdown("## 🤟 SIGNOVA")
    st.caption("Real-Time Static Hand-Sign Recognition")
    page = st.radio("Navigation", ["🎥 Live Translator", "📊 Dataset Lab", "🧠 Model Insights", "ℹ️ About"])
    st.markdown("---")
    st.caption(f"{len(available_classes)} / 36 classes usable")
    st.caption(f"{len(X)} landmark samples")
    st.caption(f"{extraction_rate:.1f}% extraction rate")

if page == "🎥 Live Translator":
    hero("SIGNOVA", "Real-time static hand-sign recognition running directly in your browser. This version bypasses WebRTC, STUN, TURN and aiortc completely.")
    st.success("Direct-browser camera mode is active. Click START CAMERA below. Your video stays in the browser; only the precomputed training features are embedded.")
    browser_camera_component(browser_payload)
elif page == "📊 Dataset Lab":
    hero("Dataset Lab", "MediaPipe landmark extraction results for your 900 images.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Images", TOTAL_IMAGES); c2.metric("Classes", 36); c3.metric("Usable Samples", len(X)); c4.metric("Extraction Rate", f"{extraction_rate:.1f}%")
    rows = []
    for label in CLASSES:
        detected = int(detected_counts.get(label, 0)); failed = int(failed_counts.get(label, 0))
        rows.append({"Sign": label, "Images": IMAGES_PER_CLASS, "Detected": detected, "Failed": failed, "Detection Rate": f"{detected/IMAGES_PER_CLASS*100:.1f}%"})
    st.dataframe(rows, use_container_width=True, hide_index=True)
elif page == "🧠 Model Insights":
    hero("Model Insights", "Server-side evaluation uses an RBF-SVM. Live browser recognition uses 7-nearest-neighbour voting on the same normalized 2D feature space.")
    a, b, c, d = st.columns(4)
    a.metric("Live Classifier", "7-NN"); b.metric("Usable Classes", len(available_classes)); c.metric("Evaluation Samples", eval_test); d.metric("SVM Eval Accuracy", accuracy_text)
    st.markdown("""<div class="card"><b>Features</b><br><br>• 21 X/Y hand landmarks<br>• wrist-relative position<br>• scale normalization<br>• mirror normalization<br>• palm-axis rotation normalization<br>• joint vectors and bone lengths<br>• finger bend angles<br>• fingertip distances</div>""", unsafe_allow_html=True)
else:
    hero("About SIGNOVA", "A real-time static hand-sign recognition and sentence construction system.")
    st.markdown("""<div class="card">SIGNOVA detects 21 hand landmarks, converts them into normalized geometric features, compares them with the supplied 36-class dataset, and records stable predictions into a text sequence.<br><br>The Live Translator now performs camera capture and MediaPipe hand tracking directly in the browser. This avoids the network relay requirements of server-side WebRTC and keeps the webcam stream local to the user's browser.<br><br>This is static hand-sign recognition, not full motion- and grammar-aware natural-language sign-language translation.</div>""", unsafe_allow_html=True)
