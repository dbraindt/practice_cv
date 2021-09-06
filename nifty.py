import cv2 as cv
import numpy as np

def show(*args):
    """nasty hack courtesy interwebs"""
    """doesn't actually work though"""
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
