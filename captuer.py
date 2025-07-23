import cv2 as cv
import matplotlib.pyplot as plt


capture = cv.VideoCapture(0)

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame)

    cv.imshow('Video', frame)
    cv.imshow('Resized', frame_resized)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

# After breaking the loop, capture a frame for edge detection
isTrue, frame = capture.read()
if isTrue:
    edges = cv.Canny(frame, 200, 200,(0,0,0))
    plt.imshow(edges)
    plt.show()

capture.release()
cv.destroyAllWindows()