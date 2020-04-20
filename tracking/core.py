""" Module containing core datastructures and classes to use with the library. """

import io

import cv2
import numpy as np
from PIL import Image as PILImage

from typing import Optional, List, Union, IO, Callable, Sequence


class Image:
    def __init__(
        self, data: Optional[np.ndarray] = None, objects: Optional[List] = None
    ):
        self.data: np.ndarray = data
        self.objects: List = [] if objects is None else objects

    @classmethod
    def open(cls, fp: Union[str, IO]) -> "Image":
        """ Open an image from a file object or a filepath string. """
        img = PILImage.open(fp)
        img = img.convert('RGB')

        return cls(np.array(img))

    def save(self, fp: Union[str, IO], format: Optional[str] = None, **params) -> None:
        """ Save an image to a file object or a filepath string. """
        img = PILImage.fromarray(self.data)
        img.save(fp, format=format, **params)

    def flip(self, axis: str, inplace: bool = False) -> "Image":
        """ Flip image along one axis or both, inplace. """
        axis_dict = {"x": 0, "y": 1, "both": -1}
        flip_axis = axis_dict.get(axis, None)
        if flip_axis is None:
            raise ValueError(f"Valid axes are 'x', 'y', or 'both'. You entered {axis}.")

        img_data = cv2.flip(self.data, flip_axis)
        if inplace:
            self.data = img_data
            return self
        else:
            return Image(img_data, self.objects)

    @property
    def shape(self):
        return self.data.shape

    def numpy(self):
        return self.data

    def pil(self):
        return PILImage.fromarray(self.numpy())

    def convert(self, mode: str):
        img = self.pil()
        img = img.convert(mode)
        return Image(np.array(img))

    def __iter__(self):
        return self.objects.__iter__()

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)
        return self

    def __repr__(self):
        return self.data.__repr__()

    def _repr_png_(self):
        """ iPython display hook support
        :returns: png version of the image as bytes
        """
        b = io.BytesIO()
        self.save(b, "PNG")
        return b.getvalue()


class Module:
    def forward(self, image: Image) -> Image:
        raise NotImplementedError
        return image

    def __call__(self, image: Image) -> Image:
        return self.forward(image)


class GenericModule(Module):
    def __init__(self, func: Callable[[Image], Image]):
        self.func = func

    def forward(self, image: Image) -> Image:
        return self.func(image)


class Pipeline(Module):
    def __init__(self, *modules):
        self.modules = modules

    def forward(self, image: Image) -> Image:
        for module in self.modules:
            image = module(image)

        return image


def pipe(*modules: Callable[[Image], Image]) -> Callable[[Image], Image]:
    """ Creates a pipeline that runs an image through multiple modules."""
    pipeline = Pipeline(*modules)
    return pipeline
