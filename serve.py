from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import os
import io
from inference import InferenceONNX 

app = FastAPI()

MAX_DURATION_SECONDS = 10

async def transcribe_audio(audio_path) -> str:

    # Initialize the ONNX inference session
    inference_onnx = InferenceONNX()
    
    # Run inference
    transcript = inference_onnx.run_inference(audio_path)
    
    return transcript

def get_audio_duration(audio_data: bytes) -> float:
    audio = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
    return len(audio) / 1000.0  # Duration in seconds

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(..., description="Upload a .wav file (max 10 seconds)")):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    contents = await file.read()
    # Convert bytes to a BinaryIO (BytesIO acts like a file)
    audio_binary_io = io.BytesIO(contents)
    try:
        duration = get_audio_duration(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file. {str(e)}")

    if duration > MAX_DURATION_SECONDS:
        raise HTTPException(status_code=400, detail="Audio file is too long. Max 10 seconds allowed.")
    
    transcript = await transcribe_audio(audio_path=audio_binary_io)

    return JSONResponse(content={"transcription": transcript})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,port=8000)