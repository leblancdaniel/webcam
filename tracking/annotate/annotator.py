from typing import Callable, List, Dict, Any, Union, Optional, NoReturn

import numpy as np
from PIL import Image as PILImage
import torch

from tracking import Image, Module


class Annotator(Module):
    def __init__(
        self,
        model: torch.nn.Module,
        transform: Callable[[PILImage.Image], torch.Tensor],
        name: Optional[str] = None,
    ):
        self.model = model
        self.transform = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name

    def to(self, device: Union[torch.device, str]):
        self.model.to(device)
        self.device = device  # type: ignore

    def add_labels(
        self, predictions: torch.Tensor, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # This method should add model predictions to the object data
        raise NotImplementedError

    def forward(self, image: Image) -> Image:
        frame: PILImage = image.pil()
        objects: List[Dict[str, Any]] = image.objects
        tensors: List[torch.Tensor] = []

        cropped_image: PILImage

        # This way we can run a bare image through the annotator and still get output
        if len(objects) == 0:
            width, height = image.shape
            objects = [{"box": (0, 0, width, height)}]

        # TODO: This is probably inefficient. See if I can make it better.
        for track in objects:
            cropped_image = frame.crop(track["box"])
            tensors.append(self.transform(cropped_image).squeeze().to(self.device))

        crops: torch.Tensor
        predictions: torch.Tensor
        if tensors:
            with torch.no_grad():
                crops = torch.stack(tensors)
                predictions = self.model(crops).detach()
            objects = self.add_labels(predictions, objects)

        return Image(np.array(frame), objects)

    def __repr__(self):
        return (
            f"{self.__class__.__name__} Annotator --- model:\n" + self.model.__repr__()
        )


class Classify(Annotator):
    """ This annotator uses a classification model to add category and score labels 
        to detected objects.
        Args:
            model: PyTorch module, the classification model
            transform: torchvision transform, for preprocessing images before the model
            categories: list of strings, mapping of category index to category name
            logp: boolean, True if model returns log-probabilities
        Usage:
            # Load model and define transform
            annotator = Classify(model, transform)
            objects = annotator(image, objects)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        transform: Callable[[PILImage.Image], torch.Tensor],
        name: Optional[str] = None,
        categories: Optional[List[str]] = None,
        logp: bool = True,
        threshold: Optional[float] = None,
    ):
        super().__init__(model, transform, name)
        self.categories = categories
        self.logp = logp
        self.threshold = threshold

    def add_labels(
        self, predictions: torch.Tensor, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """ Add predicted categories and scores to the object data. """
        predicted_cats: torch.Tensor = predictions.argmax(axis=1)  # type: ignore

        cat_name: Union[str, int]
        pred: torch.Tensor
        cat_score: float
        pred_cat: torch.Tensor

        for i, pred_cat in enumerate(predicted_cats):  # type: ignore # Mypy complains about torch.Tensor not being an Iterable
            cat_name = (
                self.categories[pred_cat.item()] if self.categories else pred_cat.item()  # type: ignore
            )
            pred = predictions[i, pred_cat].to("cpu").detach()
            cat_score = torch.exp(pred).item() if self.logp else pred.item()

            if hasattr(self, "name") and self.name is not None:
                name = self.name
                objects[i][name] = {}
                objects[i][name]["label"] = cat_name
                objects[i][name]["score"] = float(cat_score)
            else:
                objects[i]["label"] = cat_name
                objects[i]["score"] = float(cat_score)

        return objects


class Expectation(Annotator):
    """ This annotator uses a model to calculate an expected value of a 
        categorical probability distribution, like you'd get from softmax.
        This adds a field 'expected' to the object data.
        Usage:
            # Load model and define transform
            annotator = Expectation(model, transform)
            objects = annotator(image, objects)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        transform: Callable[[PILImage.Image], torch.Tensor],
        name: Optional[str] = None,
        logp: bool = True,
    ):
        super().__init__(model, transform, name)
        self.logp = logp

    def add_labels(
        self, predictions: torch.Tensor, objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if self.logp:
            predictions = torch.exp(predictions)

        values: torch.Tensor = torch.arange(1, predictions.shape[1] + 1).to(self.device)
        expectations: torch.Tensor = torch.sum(values * predictions, dim=1).to("cpu")

        expected: torch.Tensor

        for i, expected in enumerate(expectations):  # type: ignore # Mypy complains about torch.Tensor not being an Iterable
            if hasattr(self, "name") and self.name is not None:
                name = self.name
                objects[i][name] = {}
                objects[i][name]["value"] = float(expected.item())  # type: ignore # Mypy says '<nothing> has no attribute "item"'
            else:
                objects[i]["value"] = float(expected.item())  # type: ignore # Mypy says '<nothing> has no attribute "item"'

        return objects
