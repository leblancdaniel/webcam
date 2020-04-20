import tracking as trk
import time

cam = trk.io.CamReader(width=1280, height=720, mirror=True)
window = trk.io.WindowWriter("Test Window", fps=True)

for image in cam:
    window.write(image)

window.close_all()
cam.close()
