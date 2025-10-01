🧠 Brain Tumor Analysis & Segmentation App

<p align="center">
<img src="https://www.google.com/search?q=https://placehold.co/600x300/1E40AF/FFFFFF%3Ftext%3DBrain%2BTumor%2BAnalysis%2BApp" alt="Application Screenshot">
</p>
Overview 

This professional Streamlit application is designed to analyze Magnetic Resonance Imaging (MRI) scans for the detection and segmentation of brain tumors. It leverages a robust dual-model deep learning architecture to provide both qualitative (classification) and quantitative (segmentation) results.
Key Features 

Feature
	

Model
	

Description

Classification
	

VGG-16
	

Accurately determines if a tumor is present (Tumor Detected or No Tumor).

Segmentation 
	

UNet
	

Provides precise localization and boundary mapping of the tumor tissue on the MRI scan.

User Interface
	

Streamlit
	

A clean, modern, and interactive web interface for easy file uploads and real-time analysis.

Statistics
	

Pandas/Matplotlib
	

A dedicated tab for viewing simulated performance metrics and model statistics.
Technical Achievements 

The deployment of this application required overcoming critical challenges related to deep learning model compatibility:

    Model Loading Resolution: Successfully resolved persistent structural errors (specifically the "Shape mismatch" issue) encountered during the loading of the VGG model weights, ensuring full functionality.

    Robust Architecture: Implemented a resilient loading strategy that manually reconstructs the VGG-16 model architecture before loading custom weights, maximizing cross-environment compatibility.

    Dual-Model Integration: Seamless integration of two distinct models (VGG for classification, UNet for pixel-level segmentation) within a single Streamlit environment.

Prerequisites 

To run this application locally, you need the following:

    Python 3.8+

    TensorFlow/Keras

    Streamlit

    OpenCV (cv2)

    NumPy

    Pandas

Setup and Installation 

    Clone the repository:

    git clone [Your Repository Link Here]
    cd brain-tumor-analysis-app


    Install dependencies:

    pip install -r requirements.txt


    (Note: You will need to create a requirements.txt file containing all necessary libraries.)

    Place Model Files:
    Ensure the two required model files are in the root directory:

        vgg_model.h5 (Classification Model)

        unet_model.h5 (Segmentation Model)

    Run the application:

    streamlit run app.py


Future Enhancements 

    Real Data Integration: Connecting the Statistics tab to live evaluation metrics.

    Model Optimization: Converting models to TensorFlow Lite or ONNX for faster inference.

    Multi-Class Support: Expanding the VGG model to classify different types of brain tumors.
