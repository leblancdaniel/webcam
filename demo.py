import io
import time

cam = io.CamReader(width=1280, height=720, mirror=True)
window = io.WindowWriter("Test Window", fps=True)

for image in cam:
    window.write(image)

window.close_all()
cam.close()
