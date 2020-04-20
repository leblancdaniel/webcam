import json
from pathlib import Path
import subprocess
from subprocess import PIPE
import time
from threading import Thread

import cv2
from tracking import Image


def get_metadata(filepath):
    """ Return the metadata for a video as a dictionary. 
        Requires ffprobe which is installed along with ffmpeg.
    """

    video_path = Path(filepath)
    if not video_path.exists():
        raise FileNotFoundError(f"Video {filepath} doesn't exist")

    command = f"ffprobe -show_streams {filepath} -select_streams v -v 0 -print_format json".split(
        " "
    )
    result = subprocess.run(command, stdout=PIPE, stderr=PIPE)

    metadata = json.loads(result.stdout)["streams"][0]

    return metadata


class FPS:
    def __init__(self, interval=15):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._numFrames = 0
        self.interval = interval
        self.fps = None

    def start(self):
        """ Start the timer """
        self._start = time.time()
        self._numFrames = 0
        return self

    def update(self):
        """ Increment the number of frames in this interval and return the FPS """
        self._numFrames += 1

        if self._numFrames == self.interval:
            self.fps = int(self._numFrames / (time.time() - self._start))
            self.start()

        return self.fps


class CamReader:
    def __init__(
        self, cam_id=0, width=1920, height=1080, fps=30, buffer_size=3, mirror=True
    ):
        """ Read webcam frames as Images in RGB color format.
            Usage:
                cam = CamReader()
                image = cam.read()
                # do something with image
                # Also as an iterator/generator
                cam = CamReader()
                for image in cam:
                    # do something with image
        """

        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # This reduces lag in the webcam feed.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

        self.mirror = mirror

        _ = self.cap.read()

    def read(self):
        _, frame = self.cap.read()
        frame = Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.mirror:
            frame.flip("y", inplace=True)

        return frame

    def close(self):
        self.cap.release()

    def __iter__(self):

        try:
            cv2.waitKey(1)
            use_waitKey = True
        except:
            # the above isn't implemented, so just have people close with Ctrl + C
            use_waitKey = False

        while True:
            # Pressing Escape key stops reading from camera
            if use_waitKey and cv2.waitKey(1) == 27:
                break

            yield self.read()


class WindowWriter:
    """ Write frames to a window.
            Args:
                window_name: name of window
            Usage:
                writer = io.WindowWriter('View Window')
                for frame in something:
                    writer(frame)
                writer.close()
    """

    def __init__(self, window_name, fps=False):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)

        if fps:
            self.fps = FPS()
            self.fps.start()
        else:
            self.fps = None

    def write(self, image):
        """ Write image to the window. """

        frame = cv2.cvtColor(image.data, cv2.COLOR_RGB2BGR)

        if self.fps is not None:
            fps = self.fps.update()
            cv2.putText(
                frame,
                f"FPS: {fps}",
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (200, 255, 155),
                thickness=3,
            )

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def close(self):
        """ Close this window. """
        cv2.destroyWindow(self.window_name)

    def close_all(self):
        """ Close all windows. """
        cv2.destroyAllWindows()

    def __call__(self, image):
        self.write(image)
        return image


class VideoReader:
    def __init__(self, filepath):
        """ Yield video frames as numpy arrays in RGB color format.
    
        Usage:
            video_reader = VideoReader('path/to/video')
            for frame in video_reader:
                # do something with frame
        """
        video_path = Path(filepath)
        if not video_path.exists():
            raise FileNotFoundError(f"Video {filepath} doesn't exist")

        vid_abs_path = str(video_path.resolve())
        self.metadata = get_metadata(vid_abs_path)
        self.vid = cv2.VideoCapture(vid_abs_path)

    def __iter__(self):
        # TODO make this run faster by parallelizing frame loading
        n_frames = int(self.metadata["nb_frames"])

        for _ in range(n_frames):
            _, frame = self.vid.read()
            # Standardize modules on RGB color format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield Image(frame, [])


class VideoWriter:
    def __init__(self, filepath, fps=30, width=1920, height=1080, fourcc="mp4v"):
        """ Write frames to a video file.
            Args:
                filepath: path for output file
                fps: int, frames per second
                weight: int, width of video
                height: int, height of video
                fourcc: string, FOURCC code for video codec. 
                    See (http://www.fourcc.org/codecs.php) for codes. Note that you 
                    must have the desired codec installed.
            Usage:
                writer = io.VideoWriter('test.mp4')
                for frame in something:
                    writer(frame)
                writer.close()
            Can also be used as a context manager:
                with io.VideoWriter('test.mp4') as writer:
                    writer(frame)
                
        """
        self.filepath = filepath
        self.fps = fps
        self.width = width
        self.height = height
        self.fourcc = fourcc

        self.out = cv2.VideoWriter(
            filepath, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height)
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, image: Image):
        """ Write image to video. image should be a Numpy array. """
        # image should be in RGB format, need to switch it to BGR for OpenCV
        frame = cv2.cvtColor(image.data, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        self.out.write(frame)

    def __call__(self, image: Image):
        """ Write image to video. """
        self.write(image)
        return image

    def close(self):
        self.out.release()
