from pathlib import Path
from typing import List, Optional, Tuple, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image as PILImage, ImageDraw, ImageFont

from tracking import Image

ColorTuple = Union[Tuple[int, int, int], Tuple[int, int, int, int]]


def default_font() -> str:
    mpl_path = Path(plt.__file__).parent
    font_path = str(mpl_path / "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf")
    return font_path


def gather_text(obj: dict, fields: list) -> str:
    text_lines: List[str] = []
    for field in fields:
        value = obj.get(field, None)
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, float):
            text_lines.append(f"{value:.3f}")
        else:
            text_lines.append(f"{value}")

    text = "\n".join(text_lines)

    return text


def draw_text(
    image: PILImage,
    text: str,
    location: Tuple[int, int] = (0, 0),
    color: ColorTuple = (0, 0, 0),
    font: Optional[Union[str, ImageFont.ImageFont]] = None,
    size: int = 20,
    bg_color: Optional[ColorTuple] = None,
    padding: Tuple[int, int] = (5, 2),
    line_spacing: int = 2,
) -> PILImage:

    draw = ImageDraw.Draw(image)

    if font is None:
        font = ImageFont.truetype(default_font(), size)
    elif isinstance(font, str):
        font = ImageFont.truetype(font, size)

    x = location[0] + padding[0]
    y = location[1] + padding[1]

    # Draw a box behind the text
    if bg_color is not None:
        text_width, text_height = ImageDraw.ImageDraw.multiline_textsize(
            text, font=font, spacing=line_spacing
        )
        draw.rectangle(
            [x, y, x + text_width + padding[0], y + text_height * 1.2], fill=bg_color,
        )

    draw.multiline_text(
        (x, y), text, font=font, fill=color, spacing=line_spacing,
    )

    return image


def draw_box(image: PILImage, obj: dict, color: ColorTuple, width: int = 3) -> PILImage:
    draw = ImageDraw.Draw(image)
    box: Tuple[int, int, int, int] = obj["box"]
    draw.rectangle(
        list(box), outline=color, width=width,
    )

    return image


def draw_mask(
    image: PILImage,
    mask: np.ndarray,
    color: ColorTuple = (0, 255, 0),
    alpha: float = 1.0,
) -> PILImage:
    mask = mask.astype(np.int8) * 127
    bitmap: PILImage = PILImage.fromarray(mask, mode="L")
    mask_color = color + (int(alpha * 255),)
    image.paste(mask_color, None, bitmap)

    return image


class Visualizer:
    def __init__(
        self,
        font: Optional[str] = None,
        font_size: int = 24,
        text_fields: Optional[List[str]] = None,
        text_color: ColorTuple = (0, 0, 0),
        text_line_spacing: int = -2,
        text_padding: Tuple[int, int] = (5, 0),
        box_color: ColorTuple = (0, 0, 0),
        box_width: int = 8,
        color_cycle: Optional[Sequence[ColorTuple]] = None,
    ):
        """ Create a Visualizer to draw object data to an image.
        
            Args:
                font: path to font you want to use
                font_size: int, font size
                text_fields: list of text fields (category, score, etc) to draw
                text_color: (R, G, B) color of text
                text_line_spacing: int, spacing between multiple lines of text
                text_padding: (x, y) padding for text placement from top left corner of bounding box
                box_color: (R, G, B) color of bounding box
                box_width: int, width of bouding box
                color_cycle: sequence of color tuples (RGB) for coloring boxes and such
        """

        if font is None:
            font = default_font()
        self.font = font
        self.font_size = font_size

        self._font = ImageFont.truetype(font, font_size)

        if text_fields is None:
            self.text_fields = ["category", "score"]
        else:
            self.text_fields = text_fields

        self.text_color = text_color
        self.text_padding = text_padding
        self.text_line_spacing = text_line_spacing
        self.box_color = box_color
        self.box_width = box_width
        self.color_cycle = color_cycle

    def __call__(self, image: Image, show_image: Optional[bool] = False):
        """ Draw object metadata to an image.
        
            Args:
                image: Numpy array, image to draw on
                objects: list of dictionaries, object metadata
                show_image: display image with Matplotlib after drawing
        """
        base: PILImage = PILImage.fromarray(image.data).convert("RGBA")
        overlay: PILImage = PILImage.new("RGBA", base.size, (255, 255, 255, 0))

        if len(image.objects) == 0:
            frame: PILImage = image.pil()

        obj_idx: int
        color_idx: int
        for obj_idx, obj in enumerate(image.objects):
            box_color: ColorTuple
            if self.color_cycle is not None:
                if "id" in obj:
                    color_idx = obj["id"] % len(self.color_cycle)
                else:
                    color_idx = obj_idx
                box_color = self.color_cycle[color_idx]
            else:
                box_color = self.box_color

            if "mask" in obj:
                overlay = draw_mask(overlay, obj["mask"], box_color)

            overlay = draw_box(overlay, obj, color=box_color, width=self.box_width)

            text = gather_text(obj, self.text_fields)

            overlay = draw_text(
                overlay,
                text,
                font=self.font,
                location=obj["box"][:2],
                color=self.text_color,
                bg_color=box_color,
                line_spacing=self.text_line_spacing,
                padding=self.text_padding,
            )

            frame = PILImage.alpha_composite(base, overlay)

        if show_image == True:
            frame = np.asarray(frame)
            plt.figure(figsize=(7, 14))
            plt.imshow(frame)
            plt.show()

        image.data = np.array(frame)
        return image

    def __repr__(self):
        return "Visualizer({self.text_fields})"
