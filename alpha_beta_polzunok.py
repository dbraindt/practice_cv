import cv2 as cv
import numpy as np

class Batmage:
    def __init__(self, path: str):

        # image params
        self.orig_img: np.ndarray = cv.imread(path)
        self.orig_num_rows = self.orig_img.shape[0]
        self.orig_num_cols = self.orig_img.shape[1]

        self.img = cv.resize(self.orig_img,
                             (int(self.orig_num_rows * .5), int(self.orig_num_cols * .5)),
                             interpolation=cv.INTER_LINEAR)


        self._alpha: float = 1.0
        self._beta: float = 0.0

        # window params
        self.name = 'polzunok'
        self.max_pos = 10

        cv.namedWindow(self.name)
        cv.createTrackbar('alpha', self.name, 1, self.max_pos, self.on_trackbar_alpha)
        cv.createTrackbar('beta', self.name, 1, self.max_pos, self.on_trackbar_beta)

        # party hard
        cv.waitKey(0)
        cv.destroyAllWindows()

    def on_trackbar_alpha(self, pos):
        self.alpha = pos / self.max_pos * 10

    def on_trackbar_beta(self, pos):
        self.beta = pos / self.max_pos * 100


    def redraw(self):
        # silly, doesn't belong here
        self.bs = cv.convertScaleAbs(self.img, alpha=self.alpha, beta=self.beta)
        cv.imshow(self.name, self.bs)

    @property
    def alpha(self):
        return self._alpha


    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        self.redraw()


    @property
    def beta(self):
        return self._beta


    @beta.setter
    def beta(self, value):
        self._beta = value
        self.redraw()


# lol what
img = Batmage('data/batman3.jpg')



