import copy
import cv2

class ImageIterator:
    """
    Bidirectional infinite iterator for OpenCV image matrices.

    Parameters
    ----------
    images : list of str and/or np.ndarray
        Images that the ImageIterator will iterate through. (The list consists
        of paths to images and/or image matrices.)
    fx: float
        Scaling factor for images, along the x-axis.
    fy: float
        Scaling factor for images, along the y-axis.
    """
    def __init__(self, images, fx=0.2, fy=0.2):
        self.fx, self.fy = fx, fy
        self.set_images(images, fx, fy)

    def set_images(self, images, fx=None, fy=None):
        assert type(fx) is float, "fx needs to be float!"
        assert type(fy) is float, "fy needs to be float!"

        if fx:
            self.fx = fx
        if fy:
            self.fy = fy

        self._images = [self._load_image(img) for img in images]
        self._iter_counter = 0
        self._end = len(self._images) - 1

    def _load_image(self, image):
        if type(image) is str:
            image = cv2.imread(image)
        return cv2.resize(image, None, fx=0.2, fy=0.2,
                          interpolation=cv2.INTER_CUBIC)

    def __len__(self):
        return len(self._images)

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter_counter == self._end:
            self._iter_counter = 0
        else:
            self._iter_counter += 1
        return self.get()

    def __reversed__(self):
        gen_copy = copy.deepcopy(self)
        gen_copy._images = list(reversed(gen_copy._images))
        gen_copy._iter_counter = gen_copy._end - gen_copy._iter_counter

        return gen_copy

    def next(self):
        return self.__next__()

    def prev(self):
        if self._iter_counter == 0:
            self._iter_counter = self._end
        else:
            self._iter_counter -= 1
        return self.get()

    def get(self):
        return self._images[self._iter_counter]
