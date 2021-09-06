import numpy as np
import scipy as sp
import cv2 as cv

img = cv.imread('data/batman4.jpg')
#%%
img.shape
print(img.shape)
_height = img.shape[0] # num_rows, y_span
_width = img.shape[1] # num_cols, x_span
_channels =  img.shape[2] # num channels
print(f'height: {_height}')
print(f'width: {_width}')
print(f'channels: {_channels}')

#%%

assert len(img) == _height
assert len(img[0]) == _width



#%%

small_bat = cv.resize(img, (int(0.5*_width),int(0.5*_height)), interpolation=cv.INTER_LINEAR)

#%%

cv.imshow('bat', img)
cv.imshow('small_bat', small_bat)
cv.waitKey(0)
cv.destroyAllWindows()
#%%

# TODO cv.imwrite()

print(f'img.ndim == {img.ndim}')
print(f'type(img) == {type(img)}') # "nd" as in multidimensional
print(f'img.dtype == {img.dtype}') # uint8 yeah

# assume RGB to lose
teal = np.zeros_like(img)
teal[:,:] = (0,128,128)

olive = np.zeros_like(img)
olive[:,:] = (128,128,0)

cv.imshow('teal?', teal)
cv.imshow('olive?', olive)
cv.waitKey(0)
cv.destroyAllWindows()

print('it is actually BGR')
teal[:,:] = (128,128,0)
olive[:,:] = (0,128,128)


red = np.ones_like(img) * np.array([0, 0, 255], dtype=np.uint8)
green = np.ones_like(img) * np.array([0, 250, 0], dtype=np.uint8)
blue = np.ones_like(img) * np.array([255, 0, 0], dtype=np.uint8)

cv.imshow('teal!=)', teal)
cv.imshow('olive!=)', olive)
cv.imshow('red', red)
cv.imshow('green', green)
cv.imshow('blue', blue)
cv.waitKey(0)
cv.destroyAllWindows()

#%%
# drawing
# TODO
# cv.line
# cv.rectangle
# cv.circle

#canvas = np.zeros((200,300,3), dtype=np.uint8)
canvas = np.ones((200,300,3), dtype=np.uint8) * 255

cv.line(canvas, pt1=(150,50),pt2=(150,100), color=(0,0,0), thickness=1)
cv.line(canvas, pt1=(170,50),pt2=(170,100), color=(0,0,0), thickness=1)
cv.line(canvas, pt1=(120,120),pt2=(190,120), color=(0,0,0), thickness=1)
cv.circle(canvas, (155,90),radius=60,color=(0,0,0), thickness=1)
cv.rectangle(canvas, pt1=(95,30), pt2=(215, 150), color=(0,0,0), thickness=2)
cv.rectangle(canvas, pt1=(20,30), pt2=(80, 50), color=(0,0,0), thickness=-1)

cv.imshow('canvas',canvas)
cv.waitKey(0)
cv.destroyAllWindows()

#%%
#
x_span = 300
y_span = 200
canvas = np.ones((y_span,x_span,3), dtype=np.uint8) * 255

center_xy = np.min([int(x_span/2), int(y_span/2)])
num_colors = 8
for r, c in zip(range(center_xy, 1, -int(center_xy / (num_colors-1))), range(1, num_colors)):
    _color = (np.array([int(bool(c & 4)), int(bool(c & 2)), int(bool(c & 1))]) * 255).tolist()
    print(_color)
    cv.circle(canvas, center=(center_xy,center_xy), radius=r, color=_color, thickness=-1)

gray_canvas = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
gray_canvas = cv.warpAffine(gray_canvas, np.array([[1,0,250],[0,1,150]], dtype=np.float32), dsize=(gray_canvas.shape[1]+250, gray_canvas.shape[0]+150 ))
gray_canvas[0:150,0:250] = 255 # (y,x) here because (row, col)
gray_canvas[150:,0:250] = 255 # lower quadrant
gray_canvas[0:150,250:] = 255 # upper quadrant
gray_canvas = cv.GaussianBlur(gray_canvas, (9,9), 0)
canvas = cv.resize(canvas, (int(1.7 * x_span), int(1.7 * y_span)), interpolation=cv.INTER_CUBIC)
cv.imshow('canvas',canvas)
cv.imshow('gray_canvas',gray_canvas)
cv.waitKey(0)
cv.destroyAllWindows()

#%%
# getting tired of the last three lines of every cell...

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

def fall_on_face():
    spotlight_jpg = cv.imread('data/spotlight.jpg')
    spotlight_png = cv.imread('data/spotlight.png')

    show(spotlight_png, spotlight_jpg)

# TODO fix this, maybe pass in a dict or auto-name
# will do for now
fall_on_face()

#%%

spotlight_jpg = cv.imread('data/spotlight.jpg')
spotlight_png = cv.imread('data/spotlight.png')

show(spotlight_png, spotlight_jpg)
#%%
# TODO
import matplotlib.pyplot as plt
# % matplotlib inline
#cv.cvtColor(spotlight_png,code=cv.COLOR_RGB2BGR)
#cv.colorChange()
show(spotlight_png[:,:,0])

#%%
# following https://numpy.org/numpy-tutorials/content/tutorial-svd.html?highlight=axis now
img_array = img / 255
#common pattern is convert to float
# prefer `img_as_float` utility function from scikit-image
print(img_array.min())
print(img_array.max())
print(img_array.dtype)
#%%
import numpy.linalg as la
# SVD - singular value decomposition
# numpy.linalg
# and
# scipy.linalg
# have it
# u * sigma * vt = A
img_gray = img_array @ [0.2126, 0.7152, 0.0722]
U, s, Vt = la.svd(img_gray)
# interesting...
# assert
# TODO assert

print(f'U - {U.shape}')
print(f's - {s.shape}')
print(f'Vt - {Vt.shape}')

# assert U @ s @ Vt == img_gray, "this would fail s is a vector"

sigma = np.zeros((U.shape[1], Vt.shape[0]))
np.fill_diagonal(sigma, s)

svd_rebuilt = U @ sigma @ Vt



#%%
plt.imshow(img_gray, cmap='gray')
plt.show()
plt.imshow(svd_rebuilt, cmap='gray')
plt.show()
#%%

print("norm difference: ", la.norm(img_gray - svd_rebuilt))
print("allclose: ", np.allclose(img_gray, svd_rebuilt))

#%%
plt.plot(s)
plt.show()
"""
most of the singular values in the sigma matrix are pretty small. 
So it might make sense to use only the information related to the first X singular values 
to build a more economical approximation to our image.
"""

#%%
# todo - think about it...
k = 50
approx = U @ sigma[:, :k] @ Vt[:k, :]

plt.imshow(approx, cmap='gray')
plt.show()
#%%

np.transpose(img_array, (2,0,1))
np.dot()
np.matmul() # @
np.clip()
#%%
# nifty!
row, col = np.ogrid[:10, :20]
mask = (row**2 + col**2)  #> 100
print(mask)

plt.imshow(mask)
plt.show()
#%%

# Image Coordinates
# grayscale (row, col)
# multichannel(eg.RGB) (row, col, ch)
# grayscale (pln, row, col)
# multichannel (pln, row, col, ch)



