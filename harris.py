import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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

#img = cv.imread('data/ska.jpg')
img = cv.imread('data/batman_again.png')

#%%

#cv.cornerHarris(img)

dilated = cv.dilate(cv.dilate(img, None), None) # artificial bokeh
blurred = cv.GaussianBlur(img, (7,7), 0)

show(img, dilated, blurred)

#%%

# TODO do Canny proper

# no idea how these thresholds come about
edgy = cv.Canny(blurred, threshold1=90, threshold2=95)
edgy2 = cv.Canny(blurred, threshold1=90, threshold2=95)

show(img, blurred, edgy)

edgy2 = cv.Canny(dilated, threshold1=90, threshold2=95)
show(img, dilated, edgy2)

#%%

# TODO do Harris proper

#cv.Sobel() to find Ix Iy image derivatives
# construct M
# find determinant and trace

#img2 = cv.imread('data/ska.jpg')
img2 = cv.imread('data/laptop_ska.jpg')

#gray_float = np.float32((cv.cvtColor(img2, cv.COLOR_BGR2GRAY) - 128) / 255.0)
gray_float = np.float32(cv.cvtColor(img2, cv.COLOR_BGR2GRAY))

corners = cv.cornerHarris(gray_float, 5, 7, 0.04)

dilated_corners = cv.dilate(corners, None)

img2[dilated_corners > 0.01 * dilated_corners.max()] = [0,0,255]
show(img2, gray_float)

#%%

# TODO now blob detection
# https://learnopencv.com/blob-detection-using-opencv-python-c/ - a bit outdated

#img = cv.imread('data/ska.jpg', cv.IMREAD_GRAYSCALE)
img = cv.imread('data/laptop_ska.jpg')
detector = cv.SimpleBlobDetector().create()
keypoints = detector.detect(img)

img_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0,0,255),
                                      cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

show(img_with_keypoints)

# well it does something...
#%%

# TODO histograms

#%%
import matplotlib.pyplot as plt

plt.figure()
plt.title('duh')
plt.xlabel('bins')
plt.ylabel('# of pixels')

#img = cv.imread('data/laptop_ska.jpg')
#img = cv.imread('data/batman4.jpg')
#img = cv.imread('data/batman_again.png')
img = cv.imread('data/spotlight.png')
channels = cv.split(img)
colors = ('b','g','r')
for chan, color in zip(channels, colors):
    hist = cv.calcHist([chan], channels=[0], mask=None, histSize=[256], ranges=[0,256])

    plt.plot(hist, color=color)

plt.show()
# Fourier Transform?
b, g, r =  channels[0],channels[1], channels[2]
show(img, b, g, r)
#%%

plt.figure()
plt.title('duh')
plt.xlabel('bins')
plt.ylabel('# of pixels')
hist = cv.calcHist([g], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
plt.plot(hist, color='g')
plt.show()
#%%

# histogram equalization

img = cv.imread('data/batman_again.png', cv.IMREAD_GRAYSCALE)
eq = cv.equalizeHist(img)

#plt.figure()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
plt.title('duh')
plt.xlabel('bins')
plt.ylabel('# of pixels')
hist1 = cv.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
hist2 = cv.calcHist([eq], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
ax1.plot(hist1)
ax2.plot(hist2)
plt.show()

cv.imshow('equalization', np.hstack([img, eq]))
cv.waitKey(0)
cv.destroyAllWindows()

#%%
def get_lut(gamma):
    linear = np.arange(0, 256, dtype=np.uint8)
    values = np.zeros_like(linear)
    #np.clip(((linear / 255.0)**(1/gamma)) * 255, 0, 255, values)
    np.clip(((linear / 255.0)**(gamma)) * 255, 0, 255, values)
    #LUT = np.array([linear, values], dtype=np.uint8)
    LUT = np.array(values, dtype=np.uint8)
    return LUT

LUT = get_lut(gamma=2.2)

#img = cv.imread('data/batman_again.png', cv.IMREAD_GRAYSCALE)
#img = cv.imread('data/spotlight.jpg', cv.IMREAD_GRAYSCALE)
img = cv.imread('data/batman.jpg', cv.IMREAD_GRAYSCALE)

luuted = cv.LUT(img, LUT)

eq = cv.equalizeHist(luuted)
#eq = cv.equalizeHist(img)




cv.imshow('equalization', np.hstack([img,eq]))
cv.waitKey(0)
cv.destroyAllWindows()



fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)


hist1 = cv.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0,256])
ax1.plot(hist1)

hist2 = cv.calcHist([eq], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.plot(hist2)
plt.show()



#%%
img = cv.imread('data/batman.jpg', cv.IMREAD_GRAYSCALE)

retval, img_mask = cv.threshold(img, 150, 255, cv.THRESH_BINARY)

print(retval)
show(img, cv.bitwise_not(img_mask))

