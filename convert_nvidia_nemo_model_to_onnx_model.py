import nemo.collections.asr as nemo_asr
import os
import argparse

def convert_nvidia_nemo_model_to_onnx_model(output_file="stt_hi_conformer_ctc_medium.onnx"):
    print("Converting NVIDIA NeMo model to ONNX format...")

    if not output_file.endswith('.onnx'):
        raise ValueError("Output file must have a .onnx extension")
        
    
    # Load the pre-trained model
    stt_hi_conformer_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained('stt_hi_conformer_ctc_medium')

    # Set the model to evaluation mode
    stt_hi_conformer_model.eval()

    # Ensure the model directory exists
    model_dir="./models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Define the output path for the ONNX model
    output_path=os.path.join(model_dir, output_file)

    # Convert the model to ONNX format
    stt_hi_conformer_model.export(output_path)
    print(f"Model converted and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NVIDIA NeMo model to ONNX")
    parser.add_argument(
        "--output_path",
        type=str,
        default="stt_hi_conformer_ctc_medium.onnx",
        help="Output ONNX file name (default: stt_hi_conformer_ctc_medium.onnx)"
    )

    args = parser.parse_args()
    output_path = args.output_path

    if output_path.startswith("./models"):
        output_path = output_path.replace("./models/", "")

    convert_nvidia_nemo_model_to_onnx_model(output_path)