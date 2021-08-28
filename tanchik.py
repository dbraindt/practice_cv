import numpy as np
import cv2 as cv
#%%

img = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8) * 255

# col, row
# y, x
img2 = np.copy(img)
img2[3,4] = 255
img2[3,5] = 255

# interpolation=cv.INTER_CUBIC)
img = cv.resize(img, (200,200), interpolation=cv.INTER_NEAREST)
img2 = cv.resize(img2, (200,200), interpolation=cv.INTER_NEAREST)


cv.imshow('tanchik', img)
cv.imshow('exploded_tanchik', img2)
cv.waitKey(0)
cv.destroyAllWindows()

