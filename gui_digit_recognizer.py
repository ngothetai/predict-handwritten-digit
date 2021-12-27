import numpy
from tensorflow import keras
from keras.models import load_model
from keras.datasets import mnist
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# input_shape = (28, 28, 1)
#
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

model = load_model('mnist.h5')
#
# plt.subplot(221)
# plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
# plt.subplot(222)
# plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
# plt.subplot(223)
# plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
# plt.subplot(224)
# plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))
# plt.show()
#
# loss, acc = model.evaluate(x_test, y_test)
# print(loss)
# print(acc)

# def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
#     # initialize the dimensions of the image to be resized and
#     # grab the image size
#     dim = None
#     (h, w) = image.shape[:2]
#
#     # if both the width and height are None, then return the
#     # original image
#     if width is None and height is None:
#         return image
#
#     # check to see if the width is None
#     if width is None:
#         # calculate the ratio of the height and construct the
#         # dimensions
#         r = height / float(h)
#         dim = (int(w * r), height)
#
#     # otherwise, the height is None
#     else:
#         # calculate the ratio of the width and construct the
#         # dimensions
#         r = width / float(w)
#         dim = (width, int(h * r))
#
#     # resize the image
#     resized = cv2.resize(image, dim, interpolation = inter)
#
#     # return the resized image
#     return resized

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


def predict_digit(img):
    # Change PIL to CV2
    pil_image = img.convert('RGB')
    open_cv_image = numpy.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    img = open_cv_image

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
    one = numpy.ones((1, 28, 28, 1), dtype=float)
    img = one - img
    plt.imshow(img.reshape(28,28), cmap='gray')
    plt.show()

    # predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command =         self.classify_handwriting)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND) # get the coordinate of the canvas
        im = ImageGrab.grab(rect)

        digit, acc = predict_digit(im)
        self.label.configure(text= 'num: ' + str(digit)+', '+ str(int(acc*100))+'%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

app = App()
mainloop()