import numpy as np
import cv2

img = cv2.imread('data/batman3.jpg', cv2.IMREAD_COLOR)

#%%
px = img[55,55]
print(px)
#%% md
# ROI == Region of Image
if False:
    roi = img[55:70, 90:100]
    img[55:70, 90:100] = [255,255,255]

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#%%

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)


img_bg = cv2.bitwise_and(gray, gray, mask_inv)

#%%
cv2.imshow('mask', mask)
cv2.imshow('img_bg', img_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()
