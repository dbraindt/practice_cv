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
        self._gamma: float = 1.0

        # window params
        self.name = 'polzunok'
        self.max_pos = 10
        self.gamma_max_pos = 10

        cv.namedWindow(self.name)
        cv.createTrackbar('alpha', self.name, 1, self.max_pos, self.on_trackbar_alpha)
        cv.createTrackbar('beta', self.name, 1, self.max_pos, self.on_trackbar_beta)
        cv.createTrackbar('gamma', self.name, 1, self.gamma_max_pos, self.on_trackbar_gamma)

        # party hard
        cv.waitKey(0)
        cv.destroyAllWindows()

    def on_trackbar_alpha(self, pos):
        self.alpha = pos / self.max_pos * 10

    def on_trackbar_beta(self, pos):
        self.beta = pos / self.max_pos * 100

    def on_trackbar_gamma(self, pos):
        self.gamma = 0.1 + (pos / self.gamma_max_pos) * 5


    def _gamma_transform(self):

        factor = 1 / self._gamma
        I = np.arange(0, 256, dtype=np.uint8)
        lut = np.zeros(256, dtype=np.uint8)
        np.clip(((I / 255.0) ** factor) * 255.0, 0, 255, out=lut)
        return cv.LUT(self.img, lut)


    def redraw(self):
        # silly, doesn't belong here
        gimg = self._gamma_transform()
        self.bs = cv.convertScaleAbs(gimg, alpha=self.alpha, beta=self.beta)
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


    @property
    def gamma(self):
        return self._gamma


    @beta.setter
    def gamma(self, value):
        self._gamma = value + 0.1
        self.redraw()

# lol what
img = Batmage('data/batman3.jpg')



