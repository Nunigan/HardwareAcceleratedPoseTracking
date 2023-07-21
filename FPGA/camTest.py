# import the opencv library
import cv2
from imutils.video import FPS
import numpy as np

cam_width = 1280
cam_height = 720
video_capture_str = "v4l2src device=/dev/video0 ! videoscale ! videoconvert ! video/x-raw, width={}, height={}, format=NV12, framerate=30/1 ! videoconvert ! appsink".format(cam_width, cam_height)
vid = cv2.VideoCapture(video_capture_str)

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