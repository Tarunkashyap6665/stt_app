import gradio as gr
import requests

# Backend ASR API URL
API_URL = "http://localhost:8000/transcribe"  

def transcribe_audio(file_path):
    if file_path is None:
        return "Please upload a .wav file."
    
    if not file_path.endswith(".wav"):
        return "Only .wav files are supported."

    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "audio/wav")}
            response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            return response.json().get("transcription", "No transcription found.")
        else:
            return f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

interface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(sources="upload", type="filepath", label="Upload a WAV file (max 10 sec)"),
    outputs=gr.Textbox(label="Transcription"),
    title="Speech-to-Text Transcriber",
    description="Upload a .wav audio file (max 10 seconds) and get the transcription using the FastAPI + ONNX backend.",
    flagging_mode="never"
)

if __name__ == "__main__":
    interface.launch(server_port=7860, server_name="0.0.0.0")