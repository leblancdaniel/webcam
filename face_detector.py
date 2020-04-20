import torch
import tracking as trk
from tracking import Image
from tracking.detect import FaceMTCNN
from tracking.annotate import load_osnet
import time
import os

cam = trk.io.CamReader(width=1280, height=720, mirror=True)
window = trk.io.WindowWriter("Test Window", fps=True)

script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "models")
osnet_path = os.path.join(model_path, "/osnet075.pth")

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

prev_frame_time = time.time()
for image in cam:
    # Detect faces in camera
    image = detector(image)
    # write to window
    window.write(image)
    prev_frame_time = time.time()
    
window.close_all()
cam.close()
