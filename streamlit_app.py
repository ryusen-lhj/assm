import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="SIGNOVA", page_icon="🤟", layout="wide")

DATASET_IMAGES = 36000
DATASET_CLASSES = 36
TRAIN_IMAGES = 28800
TEST_IMAGES = 7200
LANDMARK_SAMPLES = 30962
VALIDATION_SAMPLES = 6193
VALIDATION_ACCURACY = 100.0

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 10% 5%,rgba(124,58,237,.16),transparent 30%),radial-gradient(circle at 90% 10%,rgba(14,165,233,.10),transparent 30%),#070a12;color:#f8fafc}
[data-testid="stSidebar"]{background:#0b0f1a;border-right:1px solid rgba(255,255,255,.07)}
.hero{font-size:50px;font-weight:950;letter-spacing:-2px}.sub{color:#94a3b8;line-height:1.7;max-width:920px}.card{background:rgba(15,23,42,.78);border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:18px}
</style>
""", unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub">{subtitle}</div>', unsafe_allow_html=True)


def camera_component():
    connections = str(HAND_CONNECTIONS).replace("(", "[").replace(")", "]")
    html = r'''
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
*{box-sizing:border-box}body{margin:0;background:#070a12;color:#f8fafc;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}.shell{border:1px solid rgba(255,255,255,.09);border-radius:18px;background:rgba(15,23,42,.84);padding:16px}.top{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}button{border:0;border-radius:10px;padding:10px 16px;font-weight:800;background:#4f46e5;color:#fff;cursor:pointer}button.secondary{background:#1e293b}button.danger{background:#7f1d1d}.status{color:#94a3b8;font-size:13px}.good{color:#4ade80}.bad{color:#fb7185}.sequence{margin:10px 0 14px;padding:14px 16px;border-radius:14px;background:#0b1220;border:1px solid rgba(129,140,248,.25)}.sequence .label{color:#818cf8;font-size:10px;letter-spacing:1.5px;font-weight:900}.sequence .text{font-size:28px;font-weight:900;letter-spacing:3px;min-height:38px}.wrap{position:relative;width:100%;max-width:760px;margin:auto}video{display:none}canvas{width:100%;border-radius:14px;background:#020617;display:block}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}.metric{background:#0b1220;border-radius:12px;padding:12px;border:1px solid rgba(255,255,255,.06)}.metric b{font-size:21px;display:block}.metric span{color:#94a3b8;font-size:11px}.note{margin-top:10px;color:#94a3b8;font-size:12px;line-height:1.5}@media(max-width:650px){.metrics{grid-template-columns:repeat(2,1fr)}}
</style>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js" crossorigin="anonymous"></script>
</head>
<body>
<div class="shell">
 <div class="top">
  <button id="startBtn">▶ START CAMERA</button>
  <button id="clearBtn" class="danger">Clear</button>
  <button id="deleteBtn" class="secondary">Delete</button>
  <button id="spaceBtn" class="secondary">Space</button>
  <span id="status" class="status">Loading landmark model…</span>
 </div>
 <div class="sequence"><div class="label">DETECTED SEQUENCE</div><div id="sequence" class="text">—</div></div>
 <div class="wrap"><video id="video" playsinline muted></video><canvas id="canvas" width="640" height="480"></canvas></div>
 <div class="metrics">
  <div class="metric"><b id="pred">-</b><span>Predicted sign</span></div>
  <div class="metric"><b id="conf">0%</b><span>MLP confidence</span></div>
  <div class="metric"><b id="stable">0/8</b><span>Stability</span></div>
  <div class="metric"><b id="top3">-</b><span>Top alternatives</span></div>
 </div>
 <div class="note"><b>Actual comparison path:</b> the 21 red MediaPipe points provide 63 values (x, y, z). They are wrist-centred and scale-normalized, then sent directly into the 36-class landmark MLP. The white skeleton and green box are visual overlays only.</div>
</div>
<script>
const CONNECTIONS=__CONNECTIONS__;
const CLASSES=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'];
const MODEL_URL='https://huggingface.co/nocontextdoruk/asl-landmark-mlp/resolve/main/mlp_asl.onnx?download=true';
const video=document.getElementById('video'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
const statusEl=document.getElementById('status'),predEl=document.getElementById('pred'),confEl=document.getElementById('conf'),stableEl=document.getElementById('stable'),top3El=document.getElementById('top3'),seqEl=document.getElementById('sequence'),startBtn=document.getElementById('startBtn');
let session=null,stream=null,running=false,busy=false,sentence='',stableLabel=null,stableCount=0,lastRecorded=null,noHandFrames=0,trajectory=[];
const STABLE_FRAMES=8,MIN_CONF=0.62,REARM_FRAMES=5;

function updateSequence(){seqEl.textContent=sentence||'—'}
document.getElementById('clearBtn').onclick=()=>{sentence='';lastRecorded=null;stableLabel=null;stableCount=0;updateSequence()};
document.getElementById('deleteBtn').onclick=()=>{sentence=sentence.slice(0,-1);lastRecorded=null;updateSequence()};
document.getElementById('spaceBtn').onclick=()=>{if(sentence&&!sentence.endsWith(' '))sentence+=' ';lastRecorded=null;updateSequence()};

async function loadModel(){
 try{
   ort.env.wasm.numThreads=1;
   session=await ort.InferenceSession.create(MODEL_URL,{executionProviders:['wasm']});
   statusEl.innerHTML='<span class="good">Landmark MLP ready. Click START CAMERA.</span>';
 }catch(e){statusEl.innerHTML='<span class="bad">Model load failed: '+e.message+'</span>';}
}

function normalizeLandmarks(lm){
 const wx=lm[0].x,wy=lm[0].y,wz=lm[0].z||0;
 const mx=lm[9].x-wx,my=lm[9].y-wy,mz=(lm[9].z||0)-wz;
 let scale=Math.hypot(mx,my,mz);if(!Number.isFinite(scale)||scale<1e-7)scale=1;
 const out=new Float32Array(63);let k=0;
 for(const p of lm){out[k++]=(p.x-wx)/scale;out[k++]=(p.y-wy)/scale;out[k++]=((p.z||0)-wz)/scale;}
 return out;
}
function softmax(logits){let m=-Infinity;for(const v of logits)if(v>m)m=v;const p=new Float32Array(logits.length);let s=0;for(let i=0;i<logits.length;i++){p[i]=Math.exp(logits[i]-m);s+=p[i]}for(let i=0;i<p.length;i++)p[i]/=s;return p;}
async function classify(lm){
 if(!session)return null;
 const input=new ort.Tensor('float32',normalizeLandmarks(lm),[1,63]);
 const result=await session.run({input});
 const outputName=session.outputNames[0];const logits=result[outputName].data;const probs=softmax(logits);
 const order=Array.from(probs.keys()).sort((a,b)=>probs[b]-probs[a]);const best=order[0];
 return{label:CLASSES[best],confidence:probs[best],top:order.slice(0,3).map(i=>[CLASSES[i],probs[i]])};
}
function boxFor(lm){const xs=lm.map(p=>p.x*canvas.width),ys=lm.map(p=>p.y*canvas.height);const pad=24;return{x:Math.max(0,Math.min(...xs)-pad),y:Math.max(0,Math.min(...ys)-pad),r:Math.min(canvas.width,Math.max(...xs)+pad),b:Math.min(canvas.height,Math.max(...ys)+pad)};}
function drawOverlay(lm,r){
 ctx.drawImage(video,0,0,canvas.width,canvas.height);const pts=lm.map(p=>[p.x*canvas.width,p.y*canvas.height]),box=boxFor(lm);
 ctx.strokeStyle='#00ff50';ctx.lineWidth=4;ctx.strokeRect(box.x,box.y,box.r-box.x,box.b-box.y);
 ctx.strokeStyle='#fff';ctx.lineWidth=3;for(const[a,b]of CONNECTIONS){ctx.beginPath();ctx.moveTo(...pts[a]);ctx.lineTo(...pts[b]);ctx.stroke()}
 for(const[x,y]of pts){ctx.beginPath();ctx.fillStyle='#ff2020';ctx.arc(x,y,5,0,Math.PI*2);ctx.fill()}
 trajectory.push(pts[8]);if(trajectory.length>18)trajectory.shift();if(trajectory.length>1){ctx.strokeStyle='#00dcff';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(...trajectory[0]);for(let i=1;i<trajectory.length;i++)ctx.lineTo(...trajectory[i]);ctx.stroke()}
 ctx.fillStyle='rgba(8,15,30,.90)';ctx.fillRect(12,12,315,96);ctx.strokeStyle='#00ff50';ctx.lineWidth=2;ctx.strokeRect(12,12,315,96);ctx.fillStyle='#fff';ctx.font='bold 22px sans-serif';ctx.fillText('SIGN: '+r.label,26,42);ctx.fillStyle='#70ff96';ctx.font='16px sans-serif';ctx.fillText('CONFIDENCE: '+(r.confidence*100).toFixed(1)+'%',26,70);ctx.fillStyle='#b8beff';ctx.fillText('TEXT: '+(sentence.slice(-20)||'-'),26,96);
}
function drawNoHand(){ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.fillStyle='rgba(8,15,30,.9)';ctx.fillRect(12,12,260,64);ctx.fillStyle='#d0d8e0';ctx.font='bold 20px sans-serif';ctx.fillText('WAITING FOR HAND',26,50)}

const hands=new Hands({locateFile:file=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
hands.setOptions({maxNumHands:1,modelComplexity:1,minDetectionConfidence:.55,minTrackingConfidence:.55});
hands.onResults(async results=>{
 if(!running){busy=false;return}
 if(!results.multiHandLandmarks||!results.multiHandLandmarks.length){busy=false;noHandFrames++;trajectory=[];stableLabel=null;stableCount=0;predEl.textContent='-';confEl.textContent='0%';stableEl.textContent='0/8';top3El.textContent='-';if(noHandFrames>=REARM_FRAMES)lastRecorded=null;drawNoHand();return}
 noHandFrames=0;const lm=results.multiHandLandmarks[0];
 try{
   const r=await classify(lm);busy=false;if(!r)return;
   if(stableLabel===r.label)stableCount++;else{stableLabel=r.label;stableCount=1}
   if(r.confidence>=MIN_CONF&&stableCount>=STABLE_FRAMES&&lastRecorded!==r.label){sentence+=r.label;lastRecorded=r.label;updateSequence()}
   predEl.textContent=r.label;confEl.textContent=(r.confidence*100).toFixed(1)+'%';stableEl.textContent=Math.min(stableCount,STABLE_FRAMES)+'/'+STABLE_FRAMES;top3El.textContent=r.top.map(x=>x[0]).join(' · ');drawOverlay(lm,r);
 }catch(e){busy=false;statusEl.textContent='Inference error: '+e.message;}
});
async function loop(){if(!running)return;if(!busy&&video.readyState>=2){busy=true;try{await hands.send({image:video})}catch(e){busy=false;statusEl.textContent='MediaPipe error: '+e.message}}requestAnimationFrame(loop)}
async function startCamera(){if(running)return;if(!session){statusEl.textContent='Model is still loading…';return}try{statusEl.textContent='Requesting camera…';stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},facingMode:'user'},audio:false});video.srcObject=stream;await video.play();running=true;startBtn.textContent='■ STOP CAMERA';startBtn.onclick=stopCamera;statusEl.innerHTML='<span class="good">Camera connected. Landmark comparison active.</span>';requestAnimationFrame(loop)}catch(e){statusEl.innerHTML='<span class="bad">Camera error: '+e.name+' — '+e.message+'</span>'}}
function stopCamera(){running=false;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;startBtn.textContent='▶ START CAMERA';startBtn.onclick=startCamera;statusEl.textContent='Camera stopped.';ctx.clearRect(0,0,canvas.width,canvas.height)}
startBtn.onclick=startCamera;updateSequence();loadModel();
</script>
</body>
</html>
'''.replace('__CONNECTIONS__', connections)
    components.html(html, height=800, scrolling=False)


with st.sidebar:
    st.markdown("## 🤟 SIGNOVA")
    st.caption("Real-Time Static Hand-Sign Recognition")
    page = st.radio("Navigation", ["🎥 Live Translator", "📊 Dataset Lab", "🧠 Model Insights", "💻 Computer Vision", "ℹ️ About"])
    st.markdown("---")
    st.caption("Dataset: ASL-HG")
    st.caption("36 classes · 36,000 images")
    st.caption("Landmark MLP · 30,962 samples")

if page == "🎥 Live Translator":
    hero("SIGNOVA", "Real-time static hand-sign recognition using 21 MediaPipe landmarks and a 36-class landmark MLP trained on the ASL-HG dataset.")
    st.success("Landmark-comparison mode is active. The red points now provide the actual classifier input, not just a visual overlay.")
    camera_component()
elif page == "📊 Dataset Lab":
    hero("Dataset Lab", "ASL-HG contains balanced A–Z and 0–9 gesture images from 10 participants.")
    a,b,c,d=st.columns(4);a.metric("Images", f"{DATASET_IMAGES:,}");b.metric("Classes", DATASET_CLASSES);c.metric("Training", f"{TRAIN_IMAGES:,}");d.metric("Testing", f"{TEST_IMAGES:,}")
    st.markdown("""<div class="card">The processed dataset uses an 80/20 train–test split. It also distinguishes letter <b>O</b> from digit <b>0</b> by using a separate zero gesture, reducing the ambiguity that affected the earlier dataset.</div>""", unsafe_allow_html=True)
elif page == "🧠 Model Insights":
    hero("Model Insights", "The live model classifies normalized 3D hand landmarks, not raw image pixels.")
    a,b,c,d=st.columns(4);a.metric("Classifier", "MLP");b.metric("Input", "21 × 3");c.metric("Landmark Samples", f"{LANDMARK_SAMPLES:,}");d.metric("Published Validation", f"{VALIDATION_ACCURACY:.0f}%")
    st.markdown("""<div class="card"><b>Model pipeline</b><br><br>1. MediaPipe detects 21 hand landmarks.<br>2. Wrist coordinates are subtracted from every landmark.<br>3. Coordinates are divided by wrist→middle-MCP distance for scale invariance.<br>4. The resulting 63 values are passed to a multilayer perceptron.<br>5. Softmax converts the 36 output logits to confidence values.</div>""", unsafe_allow_html=True)
elif page == "💻 Computer Vision":
    hero("Computer Vision", "What each overlay means in the live camera.")
    st.markdown("""<div class="card"><b>Red dots</b> — the 21 MediaPipe hand landmarks. These coordinates are the actual recognition input.<br><br><b>White lines</b> — skeletal connections between landmarks; useful for visualising finger geometry.<br><br><b>Green box</b> — the detected hand region for visual feedback.<br><br><b>Cyan trail</b> — recent index-fingertip trajectory; visual only.</div>""", unsafe_allow_html=True)
else:
    hero("About SIGNOVA", "A real-time static hand-sign recognition and sentence construction system.")
    st.markdown("""<div class="card">SIGNOVA uses MediaPipe hand tracking in the browser and a landmark MLP trained on the ASL-HG 36-class dataset. The live video remains in the browser. The recognition model consumes normalized 21-point hand geometry and predicts A–Z or 0–9 with confidence.<br><br>This is static hand-sign recognition; it does not model full sign-language grammar or continuous motion semantics.</div>""", unsafe_allow_html=True)
    st.caption("Dataset: ASL-HG (Pranto et al., 2026), CC BY 4.0. Landmark model: nocontextdoruk/asl-landmark-mlp, CC BY-NC 4.0.")
