# Project Description: Hindi Speech-to-Text Model Deployment

## Successfully Implemented Features

### 1. Model Preparation and Optimization
- Successfully integrated the NVIDIA NeMo Hindi Conformer CTC Medium model
- Implemented model conversion to ONNX format for optimized inference
- Created preprocessing pipeline for 16kHz audio input
- Handles audio files of 5-10 seconds duration

### 2. FastAPI Implementation
- Created a robust FastAPI server with `/transcribe` endpoint
- Implemented input validation for:
  - File type (.wav format)
  - Audio duration (max 10 seconds)
  - Sample rate (16kHz)
- Returns transcribed text in JSON format
- Added comprehensive error handling

### 3. Containerization
- Created a Dockerfile for containerized deployment
- Used Python 3.10 slim base image for optimized container size
- Configured for both FastAPI and web interface deployment
- Set up proper port mappings (8000 for API, 3000 for web UI)

### 4. Additional Feature: Web Interface
- Developed a user-friendly Gradio web interface as an enhancement to the core FastAPI implementation
- Provides easy audio file upload and transcription through a graphical interface
- Real-time feedback on transcription process with progress indicators
- Visual display of results with formatted output
- Complements the API endpoint for better user experience


## Development Issues and Solutions

### 1. Environment Setup Challenges
**Issue**: Complex dependency management during NVIDIA NeMo framework installation
- Repeated dependency conflicts with texterrors module
- Compatibility issues with different g++ versions

**Solution**:
- Used specific g++ 11 version to resolve texterrors module installation
- Created a controlled environment with exact version specifications

### 2. Audio Processing Challenges
**Issue**: Model compatibility with audio formats
- Model only supports mono audio, not stereo

**Solution**:
- Implemented audio preprocessing pipeline to:
  - Convert stereo to mono audio
  - Validate audio format before processing

### 3. ONNX Model Conversion Issues
**Issue**: Input format mismatch after ONNX conversion
- Direct audio input not working with ONNX model
- Missing preprocessing steps in converted model

**Solution**:
- Utilized NVIDIA NeMo ASR model preprocessor component for audio preprocessing
- Created data loader for proper input formatting
- Utilized NVIDIA NeMo ASR model decoder component to convert model logits to text

## Limitations and Technical Constraints

### 1. Model Dependencies
- Heavy reliance on NVIDIA NeMo framework
- Required components:
  - Preprocessor from NeMo ASR model
  - Decoder from NeMo ASR model
- Large environment size due to dependencies:
  - PyTorch
  - CUDA
  - Other NeMo dependencies

### 2. Audio Processing Limitations
- Restricted to mono audio format
- Fixed sample rate requirement (16kHz)

### 3. Deployment Constraints
- Large container size due to NeMo dependencies
- Limited concurrent request handling

## Future Improvements

### 1. Dependency Optimization
- Remove dependency on NeMo's preprocessor and decoder
- Implement custom preprocessing and decoding
- Reduce container size by removing NVIDIA NeMo framework dependencies

### 2. Performance Enhancements
- Implement async processing for better concurrency
- Add request queuing system

### 3. Feature Extensions
- Support for stereo audio processing
- Flexible sample rate handling
- Batch processing capability

## Conclusion

The project successfully implements the core requirements while facing and overcoming several technical challenges. The main limitation stems from the heavy dependency on NVIDIA NeMo framework, which we plan to address in future iterations by implementing custom preprocessing and decoding components. This will significantly reduce the application size and improve deployment efficiency.
