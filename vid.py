import cv2 as cv
import numpy as np

alpha = 1.0
beta = 0.0
#gamma = 2.2
gamma = 2.2

def get_lut():
    linear = np.arange(0, 256, dtype=np.uint8)
    values = np.zeros_like(linear)
    #np.clip(((linear / 255.0)**(1/gamma)) * 255, 0, 255, values)
    np.clip(((linear / 255.0)**(gamma)) * 255, 0, 255, values)
    #LUT = np.array([linear, values], dtype=np.uint8)
    LUT = np.array(values, dtype=np.uint8)
    return LUT

LUT = get_lut()

def lut_process(frame):

    #mess...

    #linear = np.zeros_like(frame)

    linearized = cv.LUT(frame, LUT)

    blurred = cv.GaussianBlur(linearized, (7, 7), 0)
    out = cv.Canny(blurred, threshold1=90, threshold2=95)

    #gray_float = np.float32(cv.cvtColor(out, cv.COLOR_BGR2GRAY))

    corners = cv.cornerHarris(out, 2, 3, 0.04)

    dilated_corners = cv.dilate(corners, None)

    blurred[dilated_corners > 0.01 * dilated_corners.max()] = [0, 0, 255]

    return blurred

def process(frame):

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    equalized = cv.equalizeHist(gray)
    #out = gray
    out = cv.medianBlur(equalized,9)

    return out


cap = cv.VideoCapture(0)


while True:
    ret, frame = cap.read()
    # not sure what ret is

    if ret:
        processed = process(frame)


        cv.imshow('frame', frame)
        cv.imshow('processed', processed)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

