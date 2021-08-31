import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
#%%

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
img = cv.imread('data/batman3.jpg')

batslap = cv.resize( img,
           (int(img.shape[1]*.5), int(img.shape[0]*.5)),
           interpolation=cv.INTER_LINEAR)

#show(batslap)
#%%

#alpha = 1.0
#beta = 1.0
alpha = 4.5 # contrast ?
beta = 1.0 # exposure / gain
gamma = 0.5

#bs = cv.convertScaleAbs(batslap, alpha=alpha, beta=beta)
#show(batslap, bs)

#%%


def on_trackbar(pos):
    global bs
    bs = cv.convertScaleAbs(batslap, alpha=(pos / max_pos * 10), beta=beta)
    cv.imshow(window, bs)


window = 'what'
cv.namedWindow(window)
max_pos = 10
cv.createTrackbar('polzunok', window, 1, max_pos, on_trackbar)
cv.imshow(window, bs)
cv.waitKey(0)
cv.destroyAllWindows()

# pointer is unsafe and deprecated.
# Use NULL as value pointer.
# To fetch trackbar value setup callback.
# ...
# whatever that means


#%%

