import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(page_title="Camera Diagnostic", page_icon="📷", layout="centered")

st.title("📷 Camera Diagnostic")
st.write(
    "This page separates browser/Windows camera access from the WebRTC connection used by SIGNOVA."
)

st.subheader("1. Native browser camera test")
st.caption(
    "This uses Streamlit's built-in camera widget and does NOT use WebRTC, TURN, MediaPipe, or the SIGNOVA model."
)

picture = st.camera_input("Open camera and take a test picture", key="native-camera-test")

if picture is not None:
    st.success(
        "Native camera access works. Your browser and Windows can access the webcam. "
        "If SIGNOVA still fails, the remaining problem is WebRTC/TURN rather than the camera device."
    )
    st.image(picture, caption="Native camera test succeeded", use_container_width=True)
else:
    st.info(
        "If this widget cannot open the camera at all, fix Chrome/Windows camera permission first. "
        "WebRTC cannot work until this native test can access the webcam."
    )

st.divider()
st.subheader("2. Minimal WebRTC camera test")
st.caption(
    "This is the smallest streamlit-webrtc camera test: video only, no MediaPipe and no classifier."
)

rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
}

try:
    from twilio.rest import Client

    sid = st.secrets.get("TWILIO_ACCOUNT_SID")
    auth = st.secrets.get("TWILIO_AUTH_TOKEN")
    if sid and auth:
        token = Client(str(sid), str(auth)).tokens.create(ttl=3600)
        if token.ice_servers:
            rtc_configuration = {"iceServers": token.ice_servers}
            st.success("Twilio TURN credentials loaded for this test.")
    else:
        st.warning("Twilio TURN secrets are not present; this test is using STUN only.")
except Exception as exc:
    st.warning(f"TURN setup failed for this diagnostic: {type(exc).__name__}: {str(exc)[:200]}")

ctx = webrtc_streamer(
    key="camera-diagnostic-webrtc",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
    media_toggle_controls=True,
)

if ctx.state.playing:
    st.success("WebRTC camera stream is connected.")
else:
    st.caption("Press START above. If the native test works but this never connects, the issue is WebRTC/ICE/TURN.")

st.divider()
st.subheader("How to interpret the result")
st.markdown(
    """
- **Native camera fails:** Chrome or Windows is blocking/not seeing the webcam.
- **Native camera works, WebRTC fails:** camera hardware is fine; the remaining problem is WebRTC/TURN/network traversal.
- **Both work:** the camera stack is healthy and the remaining problem is inside the SIGNOVA processing path.
"""
)
