import cv2 as cv
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

# let's do alpha beta by hand, and then by foot

#img = cv.imread('data/batman4.jpg')
img = cv.imread('data/batman_again.png')

new_image = np.zeros(img.shape, dtype=np.uint8)


alpha = 2.2
beta = 50

#%%

# triple for-loop ^ tm
# i pust' ves mir podozhdet
def triple_for():
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_image[y,x,c] = np.clip(alpha * img[y,x,c] + beta, 0, 255)

triple_for()
#%%

# maybe vectorize
#new_vec_image = np.clip(alpha * img + beta, 0, 255)
# dtype is float for some reason?
# not really the same thing, even visually

# this works however, since clip saves into a uint8 ndarray
new_vec_image = np.zeros(img.shape, dtype=np.uint8)
np.clip(alpha * img + beta, 0, 255, new_vec_image)

assert np.allclose(new_image, new_vec_image)

#%%

new_cv_image = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

assert np.allclose(new_image, new_cv_image)
# looks much closer but also fails

#%%
show(img, new_image, new_vec_image, new_cv_image)

#%%

# now on to gamma

gamma = 2.2
factor = 1/gamma
I = np.arange(0, 256, dtype=np.uint8)
lut = np.zeros(256, dtype=np.uint8)
np.clip((( I / 255.0) ** factor ) * 255.0, 0, 255, out= lut)


from matplotlib import pyplot as plt
plt.plot(I, lut)
plt.show()

#%%
gimg = cv.LUT(img, lut)
#%%

show(img, gimg)

#%%

#img2 = cv.cvtColor(img, cv.COLOR_RGB2XYZ)
img2 = cv.cvtColor(img, cv.COLOR_BGR2XYZ)

show(img, img2)

# not sure what that is.

#%%

# linearization
cv.imdecode(img, cv.COLOR_LRGB2LAB)



