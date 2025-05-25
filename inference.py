
import json
import os
import tempfile
import onnxruntime
import torch

import numpy as np

from nemo.collections.asr.metrics.wer import WER
from preprocess_audio import AudioPreprocessor
from dotenv import load_dotenv
load_dotenv()


class InferenceONNX:
    def __init__(self, model_path='./models/stt_hi_conformer_model.onnx'):
        # Load the ONNX model from environment variable or default path  
        model_path = os.getenv('MODEL_PATH', model_path)
        print(f"Loading ONNX model from: {model_path}")
        self.session = onnxruntime.InferenceSession(model_path)
        self.audio_preprocessor = AudioPreprocessor(sample_rate=16000)

    def run_inference(self, inputs):
        if type(inputs) is list:
            inputs=inputs[0]

        ort_inputs = self.audio_preprocessor.preprocess(inputs) 
        ologits = self.session.run(None, ort_inputs)
        alogits = np.asarray(ologits)
        logits = torch.from_numpy(alogits[0])
        greedy_predictions = logits.argmax(dim=-1, keepdim=False)
        wer = WER(decoding=self.audio_preprocessor.getDecoding(), use_cer=False)
        hypotheses= wer.decoding.ctc_decoder_predictions_tensor(greedy_predictions)
      
        return hypotheses[0].text
    

if __name__ == "__main__":

    # Initialize the ONNX inference session
    inference_onnx = InferenceONNX()

    # Preprocess the audio file
    audio_path = './sample_audio/sample_audio.wav'
    

    # Run inference
    transcript = inference_onnx.run_inference(audio_path)
    
    print(transcript)
