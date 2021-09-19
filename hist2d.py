import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def show(*args):
    """nasty hack courtesy interwebs"""
    # could do `import inspect` as well?
    # https://exceptionshub.com/how-to-get-a-variable-name-as-a-string-in-python.html
    passed_in = {id(thing): thing for thing in args}
    global_objects = {id(o): name for name, o in globals().items()}
    print(passed_in)
    print(global_objects)
    bag = {global_objects.get(_id): thing for _id, thing in passed_in.items()}
    print(bag)
    for name, frame in bag.items():
        assert isinstance(frame, np.ndarray), f"{name} <- not an image / ndarray"
        cv.imshow(name, frame)

    cv.waitKey(0)
    cv.destroyAllWindows()

#%%
img = cv.imread('data/rgb.jpg') # BGR

#%%

img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
#%%
# 0, 1, 2
b,g,r = cv.split(img)

#%%

show(img,b,g,r)

#%%

# images, channels,
#hist = cv.calcHist([b,g],[0,0],None, [40,40], [0,255,0,255])

# B vs G
hist = cv.calcHist([img],[0,1],None, [10,10], [0,255,0,255])
plt.imshow(hist, interpolation = 'nearest')
plt.title('B vs G')
plt.show()
#%%
#
hist = cv.calcHist([img],[0,2],None, [10,10], [0,255,0,255])
plt.imshow(hist, interpolation = 'nearest')
plt.title('B vs R')
plt.show()
#%%
hist = cv.calcHist([img],[1,2],None, [10,10], [0,255,0,255])
plt.imshow(hist, interpolation = 'nearest')
plt.title('G vs R')
plt.show()

#%%
_, green_mask = cv.threshold(g, 240, 255, type=cv.THRESH_BINARY)
_, red_mask = cv.threshold(r, 240, 255, type=cv.THRESH_BINARY)
green_red_mask = cv.bitwise_and(green_mask,red_mask)

green_only = cv.bitwise_and(img,img,mask=green_mask)
red_only = cv.bitwise_and(img,img,mask=red_mask)
green_and_red = cv.bitwise_and(img,img,mask=green_red_mask)

#%%

show(green_only, red_only, green_and_red)
