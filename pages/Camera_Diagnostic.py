import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Camera Diagnostic", page_icon="📷", layout="wide")

st.title("📷 SIGNOVA Camera Diagnostic")
st.write(
    "This diagnostic now tests the same direct-browser camera path used by SIGNOVA. "
    "It no longer uses WebRTC, STUN, TURN, Twilio, aiortc, or streamlit-webrtc."
)

st.subheader("1. Streamlit native camera test")
st.caption(
    "This confirms that the browser and operating system can access a camera. "
    "Take one test photo below."
)

picture = st.camera_input("Open camera and take a test picture", key="native-camera-test")
if picture is not None:
    st.success("Native camera access works.")
    st.image(picture, caption="Native camera test succeeded", use_container_width=True)
else:
    st.info("If this cannot open, check browser and operating-system camera permission first.")

st.divider()
st.subheader("2. Direct-browser live camera + MediaPipe test")
st.caption(
    "This uses navigator.mediaDevices.getUserMedia(), the same capture method as the Live Translator. "
    "It also loads MediaPipe Hands and reports whether 21 landmarks are detected."
)

html = r'''
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
*{box-sizing:border-box}
body{margin:0;background:#070a12;color:#f8fafc;font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.shell{background:#0f172a;border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:16px}
.top{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
button{border:0;border-radius:10px;padding:10px 16px;font-weight:800;background:#4f46e5;color:#fff;cursor:pointer}
button.secondary{background:#334155}
.status{color:#cbd5e1;font-size:13px}.good{color:#4ade80}.bad{color:#fb7185}.warn{color:#facc15}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}
.card{background:#0b1220;border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:12px}
.card b{font-size:17px;display:block}.card span{color:#94a3b8;font-size:11px}
video{display:none}canvas{display:block;width:100%;max-width:760px;margin:auto;border-radius:14px;background:#020617}
.details{margin-top:12px;background:#0b1220;border-radius:12px;padding:12px;font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;color:#cbd5e1;white-space:pre-wrap}
@media(max-width:700px){.grid{grid-template-columns:repeat(2,1fr)}}
</style>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
</head>
<body>
<div class="shell">
 <div class="top">
   <button id="startBtn">▶ START DIAGNOSTIC CAMERA</button>
   <button id="refreshBtn" class="secondary">↻ Refresh Devices</button>
   <span id="status" class="status">Ready.</span>
 </div>
 <div class="grid">
   <div class="card"><b id="secure">-</b><span>Secure context</span></div>
   <div class="card"><b id="api">-</b><span>getUserMedia API</span></div>
   <div class="card"><b id="devices">0</b><span>Video devices</span></div>
   <div class="card"><b id="hands">0 / 21</b><span>Hand landmarks</span></div>
 </div>
 <video id="video" playsinline muted></video>
 <canvas id="canvas" width="640" height="480"></canvas>
 <div id="details" class="details">No camera started yet.</div>
</div>
<script>
const video=document.getElementById('video'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
const startBtn=document.getElementById('startBtn'),refreshBtn=document.getElementById('refreshBtn'),statusEl=document.getElementById('status'),detailsEl=document.getElementById('details');
const secureEl=document.getElementById('secure'),apiEl=document.getElementById('api'),devicesEl=document.getElementById('devices'),handsEl=document.getElementById('hands');
let stream=null,running=false,busy=false;
secureEl.textContent=window.isSecureContext?'YES':'NO';
secureEl.className=window.isSecureContext?'good':'bad';
const apiOK=!!(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia);
apiEl.textContent=apiOK?'YES':'NO';apiEl.className=apiOK?'good':'bad';

async function refreshDevices(){
 try{
   if(!navigator.mediaDevices||!navigator.mediaDevices.enumerateDevices){devicesEl.textContent='N/A';return}
   const all=await navigator.mediaDevices.enumerateDevices();
   const cams=all.filter(d=>d.kind==='videoinput');devicesEl.textContent=String(cams.length);
   if(!running){detailsEl.textContent='Detected camera devices: '+cams.length+'\n'+cams.map((d,i)=>(i+1)+'. '+(d.label||'Camera label hidden until permission is granted')).join('\n')}
 }catch(e){detailsEl.textContent='Device enumeration failed: '+e.name+' — '+e.message}
}

const hands=new Hands({locateFile:file=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`});
hands.setOptions({maxNumHands:1,modelComplexity:1,minDetectionConfidence:.55,minTrackingConfidence:.55});
hands.onResults(results=>{
 busy=false;if(!running)return;
 ctx.drawImage(video,0,0,canvas.width,canvas.height);
 if(!results.multiHandLandmarks||!results.multiHandLandmarks.length){handsEl.textContent='0 / 21';return}
 const lm=results.multiHandLandmarks[0];handsEl.textContent='21 / 21';handsEl.className='good';
 const pts=lm.map(p=>[p.x*canvas.width,p.y*canvas.height]);
 const connections=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]];
 ctx.strokeStyle='#fff';ctx.lineWidth=3;for(const[a,b]of connections){ctx.beginPath();ctx.moveTo(...pts[a]);ctx.lineTo(...pts[b]);ctx.stroke()}
 for(const[x,y]of pts){ctx.beginPath();ctx.fillStyle='#ff2020';ctx.arc(x,y,5,0,Math.PI*2);ctx.fill()}
});

async function loop(){
 if(!running)return;
 if(!busy&&video.readyState>=2){busy=true;try{await hands.send({image:video})}catch(e){busy=false;statusEl.textContent='MediaPipe error: '+e.message}}
 requestAnimationFrame(loop);
}

async function startCamera(){
 if(running){stopCamera();return}
 if(!apiOK){statusEl.innerHTML='<span class="bad">getUserMedia is unavailable. Open the app using HTTPS and a supported browser.</span>';return}
 try{
   statusEl.textContent='Requesting camera permission...';
   stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},facingMode:'user'},audio:false});
   video.srcObject=stream;await video.play();running=true;startBtn.textContent='■ STOP CAMERA';
   const track=stream.getVideoTracks()[0];const s=track.getSettings?track.getSettings():{};
   detailsEl.textContent='Camera: '+(track.label||'Camera')+'\nResolution: '+(s.width||'?')+' × '+(s.height||'?')+'\nFrame rate: '+(s.frameRate||'?')+'\nDevice ID available: '+(s.deviceId?'YES':'NO')+'\nMediaPipe: running; show your hand to test 21 landmarks.';
   statusEl.innerHTML='<span class="good">Direct-browser camera connected.</span>';await refreshDevices();requestAnimationFrame(loop);
 }catch(e){statusEl.innerHTML='<span class="bad">Camera failed: '+e.name+' — '+e.message+'</span>';detailsEl.textContent='Common causes:\n• Camera permission denied\n• Camera already in use\n• Browser/OS camera access disabled\n• Page not running in a secure context'}
}
function stopCamera(){running=false;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;startBtn.textContent='▶ START DIAGNOSTIC CAMERA';statusEl.textContent='Camera stopped.';handsEl.textContent='0 / 21';ctx.clearRect(0,0,canvas.width,canvas.height)}
startBtn.onclick=startCamera;refreshBtn.onclick=refreshDevices;refreshDevices();
</script>
</body>
</html>
'''

components.html(html, height=770, scrolling=False)

st.divider()
st.subheader("How to read the result")
st.markdown(
    """
- **Native camera works + direct-browser camera works + 21/21 landmarks:** the camera and MediaPipe path are healthy.
- **Native camera works but direct-browser test fails:** check the error shown in the diagnostic panel; this normally means browser iframe permission or device availability.
- **Direct camera works but landmarks stay 0/21:** camera access is fine; improve lighting, keep one hand fully visible, and move it into the center of the frame.
- **Both camera tests fail:** check browser and operating-system camera permissions and confirm the webcam is not being used exclusively by another app.
"""
)

st.info(
    "SIGNOVA now uses direct browser camera access. The old WebRTC/TURN diagnostic was removed because it no longer represents the live application architecture."
)
