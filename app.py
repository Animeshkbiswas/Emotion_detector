import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, AudioProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import requests
import tempfile
import soundfile as sf
import time

st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("🎥🧠 Real-time Emotion Detection (Video + Audio)")

# --- Display placeholders ---
video_placeholder = st.empty()
audio_placeholder = st.empty()

# --- Video Processor ---
class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_emotion = "Detecting..."

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_img = image[y:y+h, x:x+w]
            _, img_encoded = cv2.imencode('.jpg', face_img)
            try:
                res = requests.post(
                    "https://animeshakb-emotion.hf.space/predict_image",
                    files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
                )
                if res.status_code == 200:
                    data = res.json()
                    label = data["emotion"]
                    conf = data["confidence"]
                    self.current_emotion = f"{label} ({conf:.2f})"
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(image, self.current_emotion, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                print("Video API error:", e)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- Audio Processor ---
class EmotionAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_chunks = []
        self.current_emotion = "Listening..."

    def recv_queued(self, frames):
        for frame in frames:
            self.audio_chunks.append(frame.to_ndarray())

        if len(self.audio_chunks) >= 20:
            audio_data = np.concatenate(self.audio_chunks, axis=1).flatten()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio_data, frames[0].sample_rate)
                try:
                    with open(f.name, "rb") as audio_file:
                        res = requests.post(
                            "https://animeshakb-emotion.hf.space/predict_audio",
                            files={"file": audio_file}
                        )
                        if res.status_code == 200:
                            result = res.json()
                            self.current_emotion = f'{result["emotion"]} ({result["confidence"]:.2f})'
                except Exception as e:
                    print("Audio API error:", e)

            self.audio_chunks.clear()

        return frames[-1]

# --- Start Stream ---
webrtc_ctx = webrtc_streamer(
    key="emotion-stream",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EmotionVideoProcessor,
    audio_processor_factory=EmotionAudioProcessor,
    media_stream_constraints={"video": True, "audio": True},
    async_processing=True
)

# --- Use st.session_state to simulate updates ---
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

if webrtc_ctx.video_processor or webrtc_ctx.audio_processor:
    # update every 0.5 sec
    if time.time() - st.session_state.last_update > 0.5:
        video_emotion = webrtc_ctx.video_processor.current_emotion if webrtc_ctx.video_processor else "Detecting..."
        audio_emotion = webrtc_ctx.audio_processor.current_emotion if webrtc_ctx.audio_processor else "Listening..."

        video_placeholder.markdown(f"📸 **Video Emotion:** {video_emotion}")
        audio_placeholder.markdown(f"🎙️ **Audio Emotion:** {audio_emotion}")

        st.session_state.last_update = time.time()

    st.experimental_rerun = lambda: None  # Prevents error from `st.rerun()` not existing
    st.button("Refresh", on_click=lambda: None)  # Dummy button to allow rerender
