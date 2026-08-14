import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="SIGNOVA", page_icon="🤟", layout="wide")

MODEL_PARTS = [
    Path("model/hog_model.part1"),
    Path("model/hog_model.part2"),
    Path("model/hog_model.part3"),
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

@st.cache_data(show_spinner=False)
def load_hog_model():
    missing = [str(p) for p in MODEL_PARTS if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))
    payload = "".join(p.read_text(encoding="utf-8") for p in MODEL_PARTS)
    return json.loads(payload)

try:
    MODEL = load_hog_model()
except Exception as exc:
    st.error("SIGNOVA model could not be loaded.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

st.markdown("""
<style>
.stApp {
  background:
    radial-gradient(circle at 10% 5%, rgba(124,58,237,.16), transparent 30%),
    radial-gradient(circle at 90% 10%, rgba(14,165,233,.10), transparent 30%),
    #070a12;
  color:#f8fafc;
}
[data-testid="stSidebar"] { background:#0b0f1a; border-right:1px solid rgba(255,255,255,.07); }
.hero {font-size:50px;font-weight:950;letter-spacing:-2px;}
.sub {color:#94a3b8;line-height:1.7;max-width:900px;}
.card {background:rgba(15,23,42,.78);border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:18px;}
</style>
""", unsafe_allow_html=True)

def hero(title, subtitle):
    st.markdown(f'<div class="hero">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub">{subtitle}</div>', unsafe_allow_html=True)


def browser_camera_component(model):
    model_json = json.dumps(model, separators=(",", ":"))
    connections_json = json.dumps(HAND_CONNECTIONS, separators=(",", ":"))
    html = r'''
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
*{box-sizing:border-box} body{margin:0;background:#070a12;color:#f8fafc;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.shell{border:1px solid rgba(255,255,255,.09);border-radius:18px;background:rgba(15,23,42,.84);padding:16px}
.top{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
button{border:0;border-radius:10px;padding:10px 16px;font-weight:800;background:#4f46e5;color:#fff;cursor:pointer}
button.secondary{background:#1e293b} button.danger{background:#7f1d1d}
.status{color:#94a3b8;font-size:13px}.good{color:#4ade80}.bad{color:#fb7185}
.sequence{margin:10px 0 14px;padding:14px 16px;border-radius:14px;background:#0b1220;border:1px solid rgba(129,140,248,.25)}
.sequence .label{color:#818cf8;font-size:10px;letter-spacing:1.5px;font-weight:900}.sequence .text{font-size:28px;font-weight:900;letter-spacing:3px;min-height:38px}
.wrap{position:relative;width:100%;max-width:760px;margin:auto} video{display:none} canvas{width:100%;border-radius:14px;background:#020617;display:block}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}.metric{background:#0b1220;border-radius:12px;padding:12px;border:1px solid rgba(255,255,255,.06)}
.metric b{font-size:21px;display:block}.metric span{color:#94a3b8;font-size:11px}.note{margin-top:10px;color:#94a3b8;font-size:12px;line-height:1.5}
@media(max-width:650px){.metrics{grid-template-columns:repeat(2,1fr)}}
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
  <div class="metric"><b id="pred">-</b><span>Predicted sign</span></div>
  <div class="metric"><b id="conf">0%</b><span>SVM confidence</span></div>
  <div class="metric"><b id="stable">0/8</b><span>Stability</span></div>
  <div class="metric"><b id="top3">-</b><span>Top alternatives</span></div>
 </div>
 <div class="note">Red dots and white lines are MediaPipe landmarks for hand localization/visualization. Classification uses HOG appearance features from the green hand crop and a linear SVM trained on 28,800 clean images.</div>
</div>
<script>
const MODEL=__MODEL__;
const CONNECTIONS=__CONNECTIONS__;
const video=document.getElementById('video'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d',{willReadFrequently:true});
const work=document.createElement('canvas');work.width=64;work.height=64;const wctx=work.getContext('2d',{willReadFrequently:true});
const statusEl=document.getElementById('status'),predEl=document.getElementById('pred'),confEl=document.getElementById('conf'),stableEl=document.getElementById('stable'),top3El=document.getElementById('top3'),seqEl=document.getElementById('sequence'),startBtn=document.getElementById('startBtn');
const raw=atob(MODEL.q_b64);const Q=new Int8Array(raw.length);for(let i=0;i<raw.length;i++){let v=raw.charCodeAt(i);Q[i]=v>127?v-256:v;}
let stream=null,running=false,busy=false,sentence='',stableLabel=null,stableCount=0,lastRecorded=null,noHandFrames=0,trajectory=[];
const STABLE_FRAMES=8,MIN_CONF=0.54;
function updateSequence(){seqEl.textContent=sentence||'—'}
document.getElementById('clearBtn').onclick=()=>{sentence='';lastRecorded=null;stableLabel=null;stableCount=0;updateSequence()};
document.getElementById('deleteBtn').onclick=()=>{sentence=sentence.slice(0,-1);lastRecorded=null;updateSequence()};
document.getElementById('spaceBtn').onclick=()=>{if(sentence&&!sentence.endsWith(' '))sentence+=' ';lastRecorded=null;updateSequence()};

function squareBox(lm){
 const xs=lm.map(p=>p.x*canvas.width),ys=lm.map(p=>p.y*canvas.height);let minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);
 let side=Math.max(maxx-minx,maxy-miny)*1.65;side=Math.max(side,120);let cx=(minx+maxx)/2,cy=(miny+maxy)/2+side*0.04;
 let sx=Math.max(0,cx-side/2),sy=Math.max(0,cy-side/2);side=Math.min(side,canvas.width-sx,canvas.height-sy);return{sx,sy,sw:side,sh:side};
}
function equalize(gray){
 const hist=new Uint32Array(256);for(const v of gray)hist[v]++;const cdf=new Uint32Array(256);let run=0,cmin=0;for(let i=0;i<256;i++){run+=hist[i];cdf[i]=run;if(!cmin&&run)cmin=run}const den=gray.length-cmin;if(den<=0)return gray;
 const out=new Uint8Array(gray.length);for(let i=0;i<gray.length;i++)out[i]=Math.max(0,Math.min(255,Math.round((cdf[gray[i]]-cmin)*255/den)));return out;
}
function hogFromBox(box){
 wctx.clearRect(0,0,64,64);wctx.drawImage(video,box.sx,box.sy,box.sw,box.sh,0,0,64,64);const rgba=wctx.getImageData(0,0,64,64).data;let gray=new Uint8Array(4096);
 for(let i=0,j=0;i<rgba.length;i+=4,j++)gray[j]=Math.round(.299*rgba[i]+.587*rgba[i+1]+.114*rgba[i+2]);gray=equalize(gray);
 const feats=new Float32Array(576);let fk=0;
 const px=(x,y)=>gray[Math.max(0,Math.min(63,y))*64+Math.max(0,Math.min(63,x))];
 for(let cy=0;cy<64;cy+=8){for(let cx=0;cx<64;cx+=8){const hist=new Float32Array(9);
   for(let y=cy;y<cy+8;y++){for(let x=cx;x<cx+8;x++){
    const gx=(-px(x-1,y-1)+px(x+1,y-1)-2*px(x-1,y)+2*px(x+1,y)-px(x-1,y+1)+px(x+1,y+1));
    const gy=(-px(x-1,y-1)-2*px(x,y-1)-px(x+1,y-1)+px(x-1,y+1)+2*px(x,y+1)+px(x+1,y+1));
    const mag=Math.hypot(gx,gy);let ang=Math.atan2(gy,gx)*180/Math.PI;if(ang<0)ang+=180;if(ang>=180)ang-=180;const bin=Math.min(8,Math.floor(ang/20));hist[bin]+=mag;
   }}let n=1e-6;for(let b=0;b<9;b++)n+=hist[b]*hist[b];n=Math.sqrt(n);for(let b=0;b<9;b++)feats[fk++]=hist[b]/n;
 }}let gn=1e-6;for(const v of feats)gn+=v*v;gn=Math.sqrt(gn);for(let i=0;i<feats.length;i++)feats[i]/=gn;return feats;
}
function classify(feat){
 const scores=new Float32Array(MODEL.classes.length);let best=-Infinity,bestI=0;
 for(let c=0;c<MODEL.classes.length;c++){let dot=0,off=c*576;for(let i=0;i<576;i++)dot+=Q[off+i]*feat[i];const s=MODEL.intercept[c]+MODEL.scales[c]*dot;scores[c]=s;if(s>best){best=s;bestI=c}}
 let den=0;const probs=new Float32Array(scores.length);for(let i=0;i<scores.length;i++){probs[i]=Math.exp(scores[i]-best);den+=probs[i]}for(let i=0;i<probs.length;i++)probs[i]/=den;
 const order=Array.from(scores.keys()).sort((a,b)=>scores[b]-scores[a]);return{label:MODEL.classes[bestI],confidence:probs[bestI],top:order.slice(0,3).map(i=>[MODEL.classes[i],probs[i]])};
}
function drawOverlay(lm,box,label,conf){
 ctx.drawImage(video,0,0,canvas.width,canvas.height);const pts=lm.map(p=>[p.x*canvas.width,p.y*canvas.height]);ctx.strokeStyle='#00ff50';ctx.lineWidth=4;ctx.strokeRect(box.sx,box.sy,box.sw,box.sh);
 ctx.strokeStyle='#fff';ctx.lineWidth=3;for(const[a,b]of CONNECTIONS){ctx.beginPath();ctx.moveTo(...pts[a]);ctx.lineTo(...pts[b]);ctx.stroke()}
 for(const[x,y]of pts){ctx.beginPath();ctx.fillStyle='#ff2020';ctx.arc(x,y,5,0,Math.PI*2);ctx.fill()}
 trajectory.push(pts[8]);if(trajectory.length>18)trajectory.shift();if(trajectory.length>1){ctx.strokeStyle='#00dcff';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(...trajectory[0]);for(let i=1;i<trajectory.length;i++)ctx.lineTo(...trajectory[i]);ctx.stroke()}
 ctx.fillStyle='rgba(8,15,30,.90)';ctx.fillRect(12,12,315,96);ctx.strokeStyle='#00ff50';ctx.lineWidth=2;ctx.strokeRect(12,12,315,96);ctx.fillStyle='#fff';ctx.font='bold 22px sans-serif';ctx.fillText('SIGN: '+label,26,42);ctx.fillStyle='#70ff96';ctx.font='16px sans-serif';ctx.fillText('CONFIDENCE: '+(conf*100).toFixed(1)+'%',26,70);ctx.fillStyle='#b8beff';ctx.fillText('TEXT: '+(sentence.slice(-20)||'-'),26,96);
}
function drawNoHand(){ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.fillStyle='rgba(8,15,30,.9)';ctx.fillRect(12,12,260,64);ctx.fillStyle='#d0d8e0';ctx.font='bold 20px sans-serif';ctx.fillText('WAITING FOR HAND',26,50)}

const hands=new Hands({locateFile:file=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
hands.setOptions({maxNumHands:1,modelComplexity:1,minDetectionConfidence:.55,minTrackingConfidence:.55});
hands.onResults(results=>{busy=false;if(!running)return;if(!results.multiHandLandmarks||!results.multiHandLandmarks.length){noHandFrames++;trajectory=[];stableLabel=null;stableCount=0;predEl.textContent='-';confEl.textContent='0%';stableEl.textContent='0/8';top3El.textContent='-';if(noHandFrames>=5)lastRecorded=null;drawNoHand();return}
 noHandFrames=0;const lm=results.multiHandLandmarks[0],box=squareBox(lm),r=classify(hogFromBox(box));if(stableLabel===r.label)stableCount++;else{stableLabel=r.label;stableCount=1}
 if(r.confidence>=MIN_CONF&&stableCount>=STABLE_FRAMES&&lastRecorded!==r.label){sentence+=r.label;lastRecorded=r.label;updateSequence()}
 predEl.textContent=r.label;confEl.textContent=(r.confidence*100).toFixed(1)+'%';stableEl.textContent=Math.min(stableCount,STABLE_FRAMES)+'/'+STABLE_FRAMES;top3El.textContent=r.top.map(x=>x[0]).join(' · ');drawOverlay(lm,box,r.label,r.confidence);
});
async function loop(){if(!running)return;if(!busy&&video.readyState>=2){busy=true;try{await hands.send({image:video})}catch(e){busy=false;statusEl.textContent='MediaPipe error: '+e.message}}requestAnimationFrame(loop)}
async function startCamera(){if(running)return;try{statusEl.textContent='Requesting camera...';stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},facingMode:'user'},audio:false});video.srcObject=stream;await video.play();running=true;startBtn.textContent='■ STOP CAMERA';startBtn.onclick=stopCamera;statusEl.innerHTML='<span class="good">Camera connected. HOG-SVM model ready.</span>';requestAnimationFrame(loop)}catch(e){statusEl.innerHTML='<span class="bad">Camera error: '+e.name+' — '+e.message+'</span>'}}
function stopCamera(){running=false;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;startBtn.textContent='▶ START CAMERA';startBtn.onclick=startCamera;statusEl.textContent='Camera stopped.';ctx.clearRect(0,0,canvas.width,canvas.height)}
startBtn.onclick=startCamera;updateSequence();
</script>
</body></html>
'''
    html = html.replace("__MODEL__", model_json)
    html = html.replace("__CONNECTIONS__", connections_json)
    components.html(html, height=800, scrolling=False)

with st.sidebar:
    st.markdown("## 🤟 SIGNOVA")
    st.caption("Real-Time Static Hand-Sign Recognition")
    page = st.radio("Navigation", ["🎥 Live Translator", "📊 Dataset Lab", "🧠 Model Insights", "🖥️ Computer Vision", "ℹ️ About"])
    st.markdown("---")
    st.caption("36 / 36 classes")
    st.caption("28,800 training images")
    st.caption("7,200 test images")

if page == "🎥 Live Translator":
    hero("SIGNOVA", "Real-time static hand-sign recognition using browser MediaPipe localization plus a HOG linear SVM trained on your new 36,000-image ASL dataset.")
    st.success("New dataset model active: 36 classes, 28,800 training images, 7,200 held-out test images.")
    browser_camera_component(MODEL)
elif page == "📊 Dataset Lab":
    hero("Dataset Lab", "Summary of the clean ASL_Processed_Images dataset you supplied.")
    a,b,c,d=st.columns(4);a.metric("Total Images","36,000");b.metric("Classes","36");c.metric("Train","28,800");d.metric("Test","7,200")
    st.markdown("""<div class="card"><b>Class balance</b><br><br>Every class 0–9 and A–Z contains 1,000 raw hand images: 800 training images and 200 held-out test images. The images do not contain pre-drawn landmark annotations, so the classifier learns the actual hand appearance instead of annotation artifacts.</div>""",unsafe_allow_html=True)
elif page == "🧠 Model Insights":
    hero("Model Insights", "The production classifier is a compact HOG + linear SVM trained offline on the full clean dataset.")
    a,b,c,d=st.columns(4);a.metric("Classifier","Linear SVM");b.metric("Feature","HOG (576)");c.metric("Test Images","7,200");d.metric("Held-out Accuracy",f"{MODEL.get('test_accuracy',0)*100:.1f}%")
    st.markdown("""<div class="card"><b>Pipeline</b><br><br>1. MediaPipe finds 21 landmarks and the hand region.<br>2. The green square is cropped and resized to 64×64.<br>3. Grayscale histogram equalization reduces lighting variation.<br>4. HOG measures edge directions in 8×8 cells (576 values).<br>5. A 36-class linear SVM scores A–Z and 0–9.<br>6. Stable high-confidence predictions are appended to the sentence.</div>""",unsafe_allow_html=True)
elif page == "🖥️ Computer Vision":
    hero("Computer Vision", "What the live overlay and classifier are doing.")
    st.markdown("""<div class="card"><b>Red dots</b> — the 21 MediaPipe hand landmarks.<br><br><b>White lines</b> — the hand skeleton connecting anatomical landmarks.<br><br><b>Green box</b> — the region actually cropped for HOG classification.<br><br><b>Cyan trace</b> — recent index-fingertip motion, visualization only.<br><br><b>Important:</b> the red dots are not matched as pixels against red dots in the dataset. MediaPipe is used to locate the hand; HOG + SVM compares the hand's edge/shape appearance learned from the clean dataset.</div>""",unsafe_allow_html=True)
else:
    hero("About SIGNOVA", "A real-time static hand-sign recognition and sentence construction system.")
    st.markdown("""<div class="card">SIGNOVA recognizes static A–Z and 0–9 hand signs. The current version uses the clean 36,000-image dataset supplied for this project, MediaPipe Hands for real-time hand localization, HOG for image-shape features, and a linear SVM for classification.<br><br>The webcam remains in the browser; there is no WebRTC/STUN/TURN video relay.<br><br>This is static hand-sign recognition, not full motion- and grammar-aware natural-language sign-language translation.</div>""",unsafe_allow_html=True)
