import torch
import tracking as trk
from tracking import Image
from tracking.detect import FaceMTCNN
from tracking.annotate import load_osnet
import time
import os

cam = trk.io.CamReader(width=1280, height=720, mirror=True)
window = trk.io.WindowWriter("Face Detection Window", fps=True)

script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "models")
osnet_path = os.path.join(model_path, "osnet075.pth")

class FaceDetector:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detection = FaceMTCNN(model_path, device=device)
        self.features = load_osnet(osnet_path, arch="x0_75")

    def __call__(self, frame: Image):
        frame = self.detection(frame)
        if len(frame.objects) < 1:
            print("-----------------------NO FACES DETECTED--------------------")
            return frame
        frame = self.features(frame)
        return frame

detector = FaceDetector()

# Visualizer takes the object metadata and draws it to the frame
visualizer = trk.visualize.Visualizer(box_color=(0, 0, 0, 150),)

prev_frame_time = time.time()
for image in cam:
    # Detect faces in camera
    image = detector(image)
    # visualize object metadata
    image = visualizer(image)
    # write to window
    window.write(image)
    prev_frame_time = time.time()
    
window.close_all()
cam.close()
