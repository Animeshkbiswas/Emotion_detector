
# Multimodal Emotion Detection (Video + Audio)

A real-time web application that detects human emotions using both **facial expressions (video)** and **voice tone (audio)**. This project integrates a **Streamlit-based frontend** with a **FastAPI backend hosted on Hugging Face Spaces**, delivering an end-to-end multimodal emotion recognition experience.

---

## 🚀 Live Demos

- 🌐 **Frontend App (Streamlit Cloud):**  
  👉 [https://emotiondetector-multimodal.streamlit.app/](https://emotiondetector-multimodal.streamlit.app/)

- ⚙️ **Backend API (Hugging Face Space):**  
  👉 [https://animeshakb-emotion.hf.space](https://animeshakb-emotion.hf.space)

---

## 🧠 How It Works

- **Frontend:** Captures real-time webcam and microphone input using `streamlit-webrtc`.
- **Backend:** Sends image/audio to Hugging Face API endpoints for inference.
- **Response:** Receives emotion predictions with confidence scores and displays them live.

---

## 🤖 Models Used

| Modality | Model | Link | Base Model |
|----------|-------|------|------------|
| 🖼️ Video | `animeshakb/emotion_video` | [View Model](https://huggingface.co/animeshakb/emotion_video) | [`Alpiyildo/vit-Facial-Expression-Recognition`](https://huggingface.co/Alpiyildo/vit-Facial-Expression-Recognition) |
| 🎙️ Audio | `animeshakb/emotion_audio` | [View Model](https://huggingface.co/animeshakb/emotion_audio) | [`superb/wav2vec2-base-superb-er`](https://huggingface.co/superb/wav2vec2-base-superb-er) |

---

## 🛠️ Tech Stack

| Layer         | Tools Used |
|---------------|-------------|
| Frontend      | Streamlit, streamlit-webrtc |
| Video Emotion | OpenCV, ViT (Hugging Face Transformers) |
| Audio Emotion | Wav2Vec2 (Hugging Face Transformers) |
| Backend API   | FastAPI, Hugging Face Spaces |
| Deployment    | Streamlit Cloud + Hugging Face Hub |
