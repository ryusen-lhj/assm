"""Streamlit Cloud entry point for SIGNOVA.

This wrapper injects a remote-host WebRTC ICE configuration before importing
SIGNOVA's existing app.py. The main application can continue using
`webrtc_streamer(...)` normally.
"""

import streamlit_webrtc

_original_webrtc_streamer = streamlit_webrtc.webrtc_streamer


def _cloud_webrtc_streamer(*args, **kwargs):
    # streamlit-webrtc requires STUN/TURN configuration on remote hosts.
    # Google STUN is the library's documented basic cloud configuration.
    kwargs.setdefault(
        "rtc_configuration",
        {
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302",
                        "stun:stun1.l.google.com:19302",
                    ]
                }
            ]
        },
    )
    return _original_webrtc_streamer(*args, **kwargs)


streamlit_webrtc.webrtc_streamer = _cloud_webrtc_streamer

# Importing app runs the existing SIGNOVA Streamlit application.
import app  # noqa: E402,F401
