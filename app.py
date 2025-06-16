import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, AudioProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import requests
import tempfile
import soundfile as sf
import time
import threading
import logging

# Suppress WebRTC noise
logging.getLogger("aioice").setLevel(logging.ERROR)
logging.getLogger("aiortc").setLevel(logging.ERROR)

st.set_page_config(page_title="Emotion Detection", layout="centered")
st.title("🎥🧠 Real-time Emotion Detection (Video + Audio)")

video_placeholder = st.empty()
audio_placeholder = st.empty()

# --- Video Transformer ---
class EmotionVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.current_emotion = "Detecting..."
        self.last_infer_time = 0
        print("📸 Video Transformer started")

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0 and time.time() - self.last_infer_time > 1:
                x, y, w, h = faces[0]
                face_img = image[y:y+h, x:x+w]
                _, img_encoded = cv2.imencode('.jpg', face_img)

                def infer():
                    try:
                        res = requests.post(
                            "https://animeshakb-emotion.hf.space/predict_image",
                            files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")},
                            timeout=4
                        )
                        if res.status_code == 200:
                            result = res.json()
                            label = result["emotion"]
                            conf = result["confidence"]
                            self.current_emotion = f"{label} ({conf:.2f})"
                    except Exception as e:
                        print("Image API error:", e)

                threading.Thread(target=infer).start()
                self.last_infer_time = time.time()

                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, self.current_emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        except Exception as e:
            print("Video transform error:", e)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def __del__(self):
        print("❌ Video Transformer stopped")

# --- Audio Processor ---
class EmotionAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_chunks = []
        self.current_emotion = "Listening..."
        print("🎙️ Audio Processor started")

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
                            files={"file": audio_file},
                            timeout=5
                        )
                        if res.status_code == 200:
                            result = res.json()
                            self.current_emotion = f'{result["emotion"]} ({result["confidence"]:.2f})'
                except Exception as e:
                    print("Audio API error:", e)

            self.audio_chunks.clear()

        return frames[-1]

    def __del__(self):
        print("❌ Audio Processor stopped")

# --- Start WebRTC Stream ---
ctx = webrtc_streamer(
    key="emotion-stream",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=EmotionVideoTransformer,
    audio_processor_factory=EmotionAudioProcessor,
    media_stream_constraints={"video": True, "audio": True},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
)

# --- Update Emotion Display ---
if ctx.state.playing:
    while True:
        if ctx.video_transformer:
            video_placeholder.markdown(f"📸 **Video Emotion:** {ctx.video_transformer.current_emotion}")
        if ctx.audio_processor:
            audio_placeholder.markdown(f"🎙️ **Audio Emotion:** {ctx.audio_processor.current_emotion}")
        time.sleep(1)
