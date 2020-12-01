# webcam video applications

This repo comes with a few cool projects I've worked with computer vision (with CPU + GPU support), including: 
 - modules to capture, preview, and save videos from webcams and saved videos
 - ability to detect faces
 - track objects and assign IDs based on cosine similarity
 - detect emotion of faces
 
 To get started, would recommend creating a virtual environment to match:
 ```
 conda create --name <env-name> python==3.7
 ```
 And install the required libraries:
 ```
 pip install -r requirements.txt
 ```
 
Explanation of the main files:
 - `tutorial.py`: To get a feel for the basic video capture methods 
 - `webcam_only.py`: To run the pipeline through the webcam without any models and preview video
 - `face_detector.py`: To run pipeline with face detector and feature tracker
 - `face_emo_detector.py`: To run pipeline with face detector, emotion detector, and feature tracker

All trained models are in the `/models` directory, and other modules are under the `/tracking` directory.
