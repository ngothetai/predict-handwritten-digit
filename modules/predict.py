import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('mnist.h5')

def resize_image(img, size=(28,28), border = 0.7857):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w
    dif = int(dif // border)

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.multiply(np.ones((dif, dif), dtype=img.dtype), 255.0)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.multiply(np.ones((dif, dif, c), dtype=img.dtype), 255.0)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


def predict_digit(path_img):
    # Change PIL to CV2
    # pil_image = img.convert('RGB')
    # open_cv_image = numpy.array(pil_image)
    # open_cv_image = open_cv_image[:, :, ::-1].copy()
    # img = open_cv_image
    img = cv2.imread(path_img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # get coordinates of bounding digit box
    _, imgBi = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
    _, contours, hierarchy = cv2.findContours(imgBi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
    x, y, w, h = bounding_boxes[0]


    # Crop image with bounding digit box
    cropImg = img[y:y+h, x:x+w]

    # Resize image (28,28)
    # cropImg = cv2.resize(cropImg, (28, 28), interpolation = cv2.INTER_LINEAR)
    cropImg = resize_image(cropImg, (28, 28))


    img = np.array(cropImg)
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0

    # convert image to black background
    one = np.ones((1, 28, 28, 1), dtype=float)
    img = one - img
    # plt.imshow(img.reshape(28,28), cmap='gray')
    # plt.show()

    # predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

