import torch

import tracking as trk
from tracking.detect import FaceMTCNN
from tracking.annotate import load_osnet
import time

cam = trk.io.CamReader(width=1280, height=720, mirror=True)
window = trk.io.WindowWriter("Test Window", fps=True)

class FaceDetector:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ###########################
        # change filepaths for demo
        ###########################
        self.detection = FaceMTCNN("file/path/to/cameralytics/models", device=device)
        self.features = load_osnet(
            "file/path/to/cameralytics/models/osnet075.pth", arch="x0_75"
        )

    def __call__(self, frame: Image):
        frame = self.detection(frame)
        # No faces detected, return False
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

