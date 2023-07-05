# import the opencv library
import cv2
from imutils.video import FPS
import numpy as np

# define a video capture object
vid = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg,framerate=30/1,width=960, height=540,type=video ! jpegdec ! videoconvert ! video/x-raw ! appsink", cv2.CAP_GSTREAMER)
# vid = cv2.VideoCapture("/home/root/ICAIPose/vid.mp4")

fpsIn = FPS().start()

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    # print(np.shape(frame))
    # frame = cv2.resize(frame, (256, 256))
    fpsIn.update()

    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fpsIn.stop()

print("[INFO] taskDisplay : elapsed FPS: {:.2f}".format(fpsIn.fps()))

# After the loop release the cap object
vid.release()
# Destroy all the windows   
cv2.destroyAllWindows()