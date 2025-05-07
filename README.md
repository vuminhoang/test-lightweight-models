# Test Lightweight Models

This repository contains quick tests and demos for several lightweight machine learning models across different tasks.

## Features

- **Speech-to-Text (English)**  
  Convert English audio into text using lightweight ASR (Automatic Speech Recognition) models.

- **Object Detection and Counting**  
  Detect and count the number of objects in an image.

- **Face and Hand Landmark Detection**  
  Identify facial keypoints and hand joints using MediaPipe or similar lightweight libraries.

- **Famous Brand Recognition**  
  Recognize logos and famous brand names from images.

## Repository Structure

- `requirements.txt` – List of required Python packages.
- `test_detect_count.py` – Script for object detection and counting.
- `test_mediapipe.py` – Script for facial and hand landmark detection.
- `test_whisper.py` – Script for speech-to-text conversion.
- `volo11n.onnx` – Lightweight ONNX model file, possibly used for brand/logo detection.

## Installation

```bash
pip install -r requirements.txt
