---
title: ASL Classification
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
---


# ASL Classification (Streamlit Demo)

This is an interactive ASL (American Sign Language) classifier demo built with PyTorch and Streamlit.

🔗 Live Demo  
https://huggingface.co/spaces/sohui/ASL-Classification

## How it works
- Model: MLP (Linear + BatchNorm + Dropout)
- Input: 28x28 grayscale images
- Output: 24 classes (A-Y, excluding J and Z)
- Dataset: Sign Language MNIST

## Features
- Upload an image or use sample test images
- Top-3 prediction with probability visualization
- Runs directly on Hugging Face Spaces

## Folder Structure
- `app.py`: Streamlit frontend & inference logic
- `model/`: Trained PyTorch model file
- `requirements.txt`: Python dependencies
- `Dockerfile`: (Optional) Space containerization
