from typing import Tuple, List, Union
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

import torch
from torchvision import transforms
from tracking import Image

from .mtcnn import MTCNN
from .utils import expand_box


def load_weights(model: MTCNN, weights_dir: str):
    """ Load MTCNN model weights from a directory. """
    path = Path(weights_dir).expanduser()
    if not path.is_dir():
        raise NotADirectoryError(
            f"Path must point to a directory with the network weight files, got {path}"
        )

    for net_name in ("pnet", "onet", "rnet"):
        net = getattr(model, net_name)
        filepath = path / (net_name + ".pth")
        state_dict = torch.load(str(filepath))
        net.load_state_dict(state_dict)
        print(f"Loaded {filepath}")


class FaceMTCNN:
    """ Detector module for detecting faces and returning bounding boxes. Adds fields
        "box" and "score" to an input image for each detected object.
    
        Arguments
        ---------
        model_path: str, path to directory with model weights. These should be three 
            files for each sub-network in the model: onet.pth, pnet.pth, and rnet.pth
        score_threshold: float, threshold for detection
        landmarks: bool, return facial landmarks along with bounding box and score
        bbox_expand: float, expand predicted boxes by a factor
        device: Device for running inference, "cpu" or "cuda" for example
    """

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.7,
        landmarks: bool = False,
        bbox_expand: float = 1.0,
        device="cpu",
    ):

        self.bbox_expand = bbox_expand
        self.score_threshold = score_threshold
        self.landmarks = landmarks
        self.device = device

        self.model = MTCNN(device=device, keep_all=True)

        load_weights(self.model, model_path)

        self.model.eval()

    def detect(self, image: PILImage.Image) -> Tuple:
        """ Runs the face detection model on a PIL Image """
        with torch.no_grad():
            if self.landmarks:
                boxes, scores, landmarks = self.model.detect(
                    image, landmarks=self.landmarks
                )
                return boxes, scores, landmarks
            else:
                boxes, scores = self.model.detect(image)
                return boxes, scores

    def __call__(self, image: Image) -> Image:

        if self.landmarks:
            boxes, scores, landmarks = self.detect(image.pil())
        else:
            boxes, scores = self.detect(image.pil())

        if boxes is None:
            return image

        height, width, _ = image.shape

        for box, score in zip(boxes, scores):
            if score < self.score_threshold:
                continue

            box = expand_box(box, self.bbox_expand, width, height)

            obj = {
                "box": box,
                "score": score.item(),
            }

            if self.landmarks:
                obj["landmarks"] = landmarks.squeeze()

            image.objects.append(obj)

        return image
