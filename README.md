# Speech-to-Text Model Deployment

This project demonstrates the deployment of a Hindi Speech-to-Text model using NVIDIA NeMo and ONNX Runtime. The model is converted from NVIDIA NeMo format to ONNX format for efficient inference and deployed using both FastAPI and a web interface. The application provides two ways to interact with the model:
1. A user-friendly web interface for easy testing and demonstration
2. A FastAPI server for production deployment and API integration

The project includes Docker support for easy deployment and can run both services simultaneously in a single container.

## Project Structure

```
.
├── models/                      # Directory for storing ONNX models
├── resources/                   # Directory containing demo videos
├── sample_audio/               # Directory containing sample audio files for testing
├── convert_nvidia_nemo_model_to_onnx_model.py  # Script to convert NeMo model to ONNX format
├── inference.py                # ONNX model inference implementation and prediction logic
├── preprocess_audio.py         # Audio preprocessing utilities for model input
├── serve.py                    # FastAPI server implementation for API endpoints
├── web-ui.py                   # Gradio web interface for user interaction
├── utils.py                    # Common utility functions and helpers
├── Description.md             # Documentation of implemented features, challenges, limitations, and future improvements
├── Dockerfile                  # Docker configuration for containerized deployment
└── .dockerignore              # Docker build context exclusions
```

## Prerequisites

- Python 3.10+
- NVIDIA NeMo
- ONNX Runtime
- FastAPI
- PyTorch
- Other dependencies listed in requirements.txt

## Model Conversion Process

1. **Convert NeMo Model to ONNX**

   The project includes a script to convert the NVIDIA NeMo model to ONNX format:

   ```bash
   python convert_nvidia_nemo_model_to_onnx_model.py --output_path stt_hi_conformer_ctc_medium.onnx
   ```

   This script:
   - Loads the pre-trained Hindi Conformer CTC model
   - Converts it to ONNX format
   - Saves the converted model in the `models/` directory

## Running the Application

1. **Start the FastAPI Server**

   ```bash
   python serve.py
   ```
   The FastAPI server will be accessible at http://localhost:8000.

2. **Access the Web Interface**

   ```bash
   python web-ui.py
   ```
   This will launch the Gradio web interface accessible at http://localhost:7860 for easy interaction with the model.

## API Usage

The server exposes a `/transcribe` endpoint that accepts WAV audio files for Hindi speech transcription.

### Request
```bash
curl -X POST "http://localhost:8000/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@./sample_audio/sample_audio.wav"
```

### Response
```json
{
    "transcription": "आर्टिफिशियल इंटेलिजेंस इस ट्रांसफॉर्मिंग द वे इंटरेक्ट विद टेक्नॉलॉजी फ्रॉम वर्चुअल असिस्टेंट टू सेल्फ ड्राइविंग काड"
}
```

**Note**: The audio file must be:
- In WAV format
- Maximum duration of 10 seconds

## Docker Deployment

The project includes video demonstrations for Docker deployment in the `resources/` folder. Follow these steps to deploy the application:

### 1. Building the Docker Image

First, build the Docker image from the project directory:

```bash
docker image build -t speech-to-text-app .
```

📹 Watch the build process: [resources/docker-build-demo.mp4](resources/docker-build-demo.mp4)

### 2. Running the Web UI Container

To run the web interface version of the application, bind the host port 3000 to the container port 7860:

```bash
docker container run -p 3000:7860 speech-to-text-app
```

📹 Watch the web UI demo: [resources/webui-container-demo.mp4](resources/webui-container-demo.mp4)

### 3. Running the FastAPI Container

To run the FastAPI server version of the application, bind the host port 8000 to the container port 8000:

```bash
docker container run -p 8000:8000 speech-to-text-app
```

📹 Watch the FastAPI demo: [resources/fastapi-container-demo.mp4](resources/fastapi-container-demo.mp4)

### 4. Running Both Web UI and FastAPI Server

To run both the Web UI and FastAPI server in a single container, use multiple port mappings:

```bash
docker container run -d -p 8000:8000 -p 3000:7860 speech-to-text-app
```

Now you can access:
- Web UI at: http://localhost:3000
- FastAPI server at: http://localhost:8000

**Note**: 
- The web UI and FastAPI server cannot run simultaneously on the same port
- Choose one deployment method based on your needs
- The web UI provides a user-friendly interface for testing
- The FastAPI server is suitable for production deployment and API integration
- When running both services, ensure you have sufficient system resources

## Model Details

- Base Model: NVIDIA NeMo Hindi Conformer CTC Medium
- Input: Audio (WAV format, 16kHz sample rate)
- Output: Hindi text transcription
- Maximum audio duration: 10 seconds

## Environment Variables

- `MODEL_PATH`: Path to the ONNX model file (default: './models/stt_hi_conformer_model.onnx')

## Notes

- The model is optimized for Hindi speech recognition
- Audio preprocessing includes resampling to 16kHz and feature extraction
- The system uses ONNX Runtime for efficient inference

## CI/CD Deployment

The project includes GitHub Actions workflow for continuous integration and deployment to DigitalOcean. The workflow automatically:

1. Builds the Docker image on every push to main branch
2. Pushes the image to DockerHub
3. Deploys the application to DigitalOcean

### Required Secrets

Set up the following secrets in your GitHub repository:

- `DOCKERHUB_USERNAME`: Your DockerHub username
- `DOCKERHUB_TOKEN`: Your DockerHub access token
- `DIGITALOCEAN_HOST`: Your DigitalOcean droplet IP
- `DIGITALOCEAN_USERNAME`: SSH username for DigitalOcean (usually 'root' for new droplets)
- `DIGITALOCEAN_PASSWORD`: Your DigitalOcean droplet root password

#### Setting up DigitalOcean Credentials

1. **Create a DigitalOcean Droplet**:
   - Log in to DigitalOcean
   - Click "Create" → "Droplets"
   - Choose Ubuntu 22.04
   - Select a plan (Basic is sufficient)
   - Choose a datacenter region
   - Select "Password" authentication
   - Set a strong root password
   - Click "Create Droplet"

2. **Get DIGITALOCEAN_HOST**:
   - After droplet creation, copy the IP address
   - This is your `DIGITALOCEAN_HOST`

3. **Get DIGITALOCEAN_USERNAME**:
   - For new droplets, the default username is `root`
   - You can verify this in the droplet's console

4. **Get DIGITALOCEAN_PASSWORD**:
   - Use the root password you set during droplet creation
   - Make sure to use a strong password

5. **Add Secrets to GitHub**:
   - Go to your GitHub repository
   - Click "Settings" → "Secrets and variables" → "Actions"
   - Click "New repository secret"
   - Add each secret with its corresponding value

### Deployment Process

1. Push your code to the main branch
2. GitHub Actions will automatically:
   - Build the Docker image
   - Push it to DockerHub
   - Deploy to DigitalOcean
3. The application will be available at:
   - FastAPI: http://your-droplet-ip:8000
   - Web UI: http://your-droplet-ip:7860
