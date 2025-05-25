# Dockerfile for deploying a FastAPI application with Gradio UI using NVIDIA NeMo models converted to ONNX format
# Use an official python:3.10-slim base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y build-essential \
    git wget libsndfile1 ffmpeg \
    && apt-get clean

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=./models/stt_hi_conformer_model.onnx

#  Copy the requirements file into the container
COPY ./requirements.txt ./requirements.txt

# Upgrade pip and install Python dependencies from requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Clean up pip cache to reduce image size
RUN pip cache purge

# Expose ports for the application 8000 for FastAPI and 7860 for Gradio UI
EXPOSE 8000 7860

# Copy the application code into the container
COPY . .

# Run the command below to convert the NVIDIA NeMo model to ONNX format only if the model is not already available in the 'models' directory. If it is present, comment out the command.
RUN python3 convert_nvidia_nemo_model_to_onnx_model.py --output_path ${MODEL_PATH}

# Run the FastAPI server and Gradio web UI
CMD python3 serve.py & \
    python3 web-ui.py