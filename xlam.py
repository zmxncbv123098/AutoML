import os
import sys
import uuid
import cv2
import numpy as np
import string
import random
# import msgpack
import requests
import base64
import shutil
import glob
import logging
import time
import traceback
import multiprocessing
import signal
from logging.handlers import RotatingFileHandler

"""
                    __________________________________________________
                    ___________________________¶¶¶____________________
                    _______________________¶¶¶¶__¶¶¶__________________
                    ______________________¶¶_______¶__________________
                    _____________________¶¶_______¶¶__________________
                    ____________________¶¶_______¶¶_______________¶¶__
                    ____________________¶________¶¶¶_____________¶¶¶__
                    ____________________¶_________¶¶____________¶¶¶___
                    ___________________¶¶_________¶¶___________¶¶_____
                    ____________________¶_________¶¶__________¶¶______
                    ____________________¶¶________¶¶_________¶¶_______
                    ____________________¶¶________¶¶________¶¶________
                    _____________________¶¶¶_______¶¶_______¶_________
                    _____________________¶¶_________¶¶¶____¶¶_________
                    _____________________¶¶___________¶¶__¶¶__________
                    _____________________¶¶____________¶¶¶¶___________
                    _____________¶¶¶¶¶¶¶¶¶______________¶¶____________
                    __________¶¶¶¶_¶¶¶¶_¶¶______________¶¶¶___________
                    _______¶¶¶¶__¶¶¶___¶¶_______________¶¶¶___________
                    ______¶¶¶____¶¶___¶¶_________________¶¶___________
                    ____¶¶¶_¶¶¶¶¶¶¶_________¶¶¶__________¶¶________¶¶¶
                    ___¶¶¶_______¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶________¶¶______¶¶¶¶_
                    __¶¶____________¶¶¶¶_____¶¶¶¶¶_______¶______¶¶¶__¶
                    __¶¶____________________¶¶__¶¶______¶¶____¶¶¶__¶¶¶
                    __¶¶¶___________________¶¶__¶¶__________¶¶¶__¶¶¶__
                    ____¶¶¶_________________¶¶¶¶¶¶¶_________¶¶__¶¶¶___
                    ___¶¶¶¶¶¶______________¶¶¶___¶¶¶_______¶¶_¶¶¶_____
                    __¶¶_¶¶¶¶¶¶¶______¶¶__¶¶___¶¶_¶¶_______¶_¶¶_______
                    ___¶___¶¶¶¶¶¶¶¶¶__¶¶¶¶¶____¶¶_¶¶_____¶¶__¶________
                    ___¶¶¶¶_____¶¶¶¶¶¶¶¶¶¶¶_____¶¶¶¶_____¶¶¶¶¶________
                    _____¶¶¶¶¶¶¶¶_______________¶¶¶¶¶_____¶¶__________
                    _____¶¶_________________¶¶¶¶¶¶¶¶¶_____¶___________
                    ______¶¶¶¶______________¶¶¶__¶¶¶¶____¶¶___________
                    ________¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶____¶¶_____¶¶___________
                    _________¶¶____¶¶¶¶¶¶¶¶______¶¶¶____¶¶____________
                    __________¶¶_______________¶¶¶¶_¶¶¶¶¶_____________
                    __________¶¶¶¶¶__________¶¶¶¶_¶¶¶¶¶_______________
                    ____________¶¶¶¶¶¶¶¶¶¶¶¶¶¶¶__¶¶¶__________________
                    ______________¶¶¶___________¶¶____________________
                    ________________¶¶¶¶¶¶_¶¶¶¶¶¶_____________________
                    __________________¶¶¶¶¶¶¶¶________________________
                    __________________________________________________
                    
"""


# collapse/expand all funcs --> 'ctrl+shift+NumPad -/+' for PyCharm


# ----------------  Utils

class Logger:

    def __init__(self, filename=None):

        self.format = '%(levelname)s -- %(asctime)s -- %(message)s'

        self.to_std = self.init_std_logger()
        self.to_std.info("Log method to std initialised.")

        if filename is not None and filename != "":
            self.filename = filename
            self.to_file = self.init_file_logger()
            self.to_std.info("Log method to {} initialised.".format(filename))

        self.printer = self.init_print()

    def init_std_logger(self):

        my_handler = logging.StreamHandler()
        my_handler.setFormatter(logging.Formatter(self.format, datefmt='%Y-%m-%d %H:%M:%S'))

        logger = logging.getLogger("std")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(my_handler)

        return logger

    def init_print(self):

        my_handler = logging.StreamHandler()
        my_handler.setFormatter(logging.Formatter('%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

        logger = logging.getLogger("printer")
        logger.setLevel(logging.INFO)
        logger.addHandler(my_handler)

        return logger

    def print(self, msg):
        self.printer.info(msg)

    def init_file_logger(self):

        my_handler = RotatingFileHandler(self.filename, mode='a', maxBytes=5 * 1024 * 1024,
                                         backupCount=0, encoding=None, delay=0)
        my_handler.setFormatter(logging.Formatter(self.format, datefmt='%Y-%m-%d %H:%M:%S'))

        logger = logging.getLogger("file_logger")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(my_handler)

        return logger


def get_random_name(name_len=22):
    """generate random name string without extension"""
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(name_len))


def get_filelist(directory, ext):
    """
    get files list with required extensions list
    """

    ret_list = []
    for folder, subs, files in os.walk(directory):

        for filename in files:

            if filename.split(".")[-1] in ext:
                ret_list.append(os.path.join(folder, filename))

    return ret_list


def get_random_uuid():
    return str(uuid.uuid4())


def get_mac():
    return str(uuid.getnode())


def get_format_date(date_format="%d-%m-%Y_%H:%M:%S"):
    return time.strftime(date_format)


def find_free_port(p_from, p_to):
    """
    get random web socket service port

    p_from  -  int
    p_to    -  int
    """

    def try_port(port):
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except:
            return False
        finally:
            sock.close()

    try:
        if not len(range(int(p_from), int(p_to))) > 0:
            raise Exception("Port range is wrong - from {} to {}".format(int(p_from), int(p_to)))
        for i in range(int(p_from), int(p_to)):
            if try_port(i):
                return i
        return -1
    except Exception as e:
        print(e)


# ----------------  Image/Bbox utils

def xyxy_to_xcycwh(imh, imw, bbox):
    """
    convert [xmin, ymin, xmax, ymax] to relative coordinates [x_center, y_center, width, height]

    imh   -  image height
    imw   -  image width
    bbox  -  absolute coords [xmin, ymin, xmax, ymax]
    """

    dw = 1. / imw
    dh = 1. / imh
    x = (bbox[0] + bbox[2]) / 2.0
    y = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = x * dw
    if x < 0:
        x = 0.0001
    if x > imw:
        x = imw * 0.9999

    y = y * dh
    if y < 0:
        y = 0.0001
    if y > imh:
        y = imh * 0.9999

    w = w * dw
    if w < 0:
        w = 0.0001
    if w > imw:
        w = imw * 0.9999

    h = h * dh
    if h < 0:
        h = 0.0001
    if h > imh:
        h = imh * 0.9999

    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]


def xyxy_to_xywh(imh, imw, bbox):
    """
    convert [xmin, ymin, xmax, ymax] to relative coordinates [x_top, y_top, width, height]

    imh   -  image height
    imw   -  image width
    bbox  -  absolute coords [xmin, ymin, xmax, ymax]
    """
    return [round(int(bbox[0]) / imw, 4), round(int(bbox[1]) / imh, 4),
            round((int(bbox[2]) - int(bbox[0])) / imw, 4), round((int(bbox[3]) - int(bbox[1])) / imh, 4)]


def xywh_to_xyxy(imh, imw, bbox):
    """
    convert relative coordinates [x_top, y_top, width, height] to absolute [xmin, ymin, xmax, ymax]

    imh   -  image height
    imw   -  image width
    bbox  -  [x,y,w,h]
    """
    return [int(bbox[0] * imw), int(bbox[1] * imh),
            int(bbox[0] * imw) + int(bbox[2] * imw), int(bbox[1] * imh) + int(bbox[3] * imh)]


def xcycwh_to_xyxy(imh, imw, bbox):
    """
    convert relative coordinates [x_center, y_center, width, height] to absolute [xmin, ymin, xmax, ymax]

    imh   -  image height
    imw   -  image width
    bbox  -  [xc,yc,w,h]
    """
    return [int((float(bbox[0]) - float(bbox[2]) / 2) * imw), int((float(bbox[1]) - float(bbox[3]) / 2) * imh),
            int((float(bbox[0]) + float(bbox[2]) / 2) * imw), int((float(bbox[1]) + float(bbox[3]) / 2) * imh)]


def draw_bbox(img, bbox, label="", bbox_color=(255, 25, 25), text_color=(225, 255, 255)):
    """
    draw bbox on image

    img   -  np.array
    bbox  -  absolute coords [xmin, ymin, xmax, ymax]
    label -  str, detected class and probability for example "{} {:.4f}".format(cls, prob)
    """
    c1, c2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    cv2.rectangle(img, c1, c2, bbox_color, thickness=tl)
    if label != "":
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0]+ t_size[0] + 2, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, bbox_color, -1)  # filled
        cv2.putText(img, label, (c1[0] + 2, c1[1] + t_size[1]), 0, tl / 3, text_color, thickness=tf,
                    lineType=cv2.LINE_AA)
    return img


def get_iou(bb1, bb2):
    """
    calculate iou of two bboxes with absolute coordinates

    bb1  -  [xmin, ymin, xmax, ymax]
    bb2  -  [xmin, ymin, xmax, ymax]
    """

    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]
    # if bb1[0] < bb1[2] or bb1[1] < bb1[3] or bb2[0] < bb2[2] or bb2[1] < bb2[3]:
    #     return 0

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_overlap(bbox1, bbox2):
    """
    calculate how many bbox1 percents inside of bbox2

    bb1  -  [xmin, ymin, xmax, ymax]
    bb2  -  [xmin, ymin, xmax, ymax]
    """

    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])
    if xB < xA or yB < yA:
        return 0

    inter_area = (xB - xA) * (yB - yA)
    bbox_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])

    if bbox_area == 0.:
        return 0

    res = inter_area / bbox_area
    return res


def cut_bbox(img, bbox, x_expand=.0, y_expand=.0):

    """
    return new image == bbox + expand coefficients

    img  -  np.array
    bbox  -  [xmin, ymin, xmax, ymax]
    x_expand=2 means increase new image width twice

    also return new image coordinates inside original image
    """

    if x_expand > 1.:
        x_exp = x_expand % 1
    else:
        x_exp = 1 - x_expand

    if y_expand > 1.:
        y_exp = y_expand % 1
    else:
        y_exp = 1 - y_expand

    yK = + int(((bbox[3] - bbox[1]) * y_exp) / 2)
    xK = + int(((bbox[2] - bbox[0]) * x_exp) / 2)

    new_bbox = [bbox[0] - xK if x_expand > 1. else bbox[0] + xK, bbox[1] - yK if y_expand > 1. else bbox[1] + yK,
                bbox[2] + xK if x_expand > 1. else bbox[2] - xK, bbox[3] + yK if y_expand > 1. else bbox[3] - yK]

    if new_bbox[0] < 0:
        new_bbox[0] = 1
    if new_bbox[1] < 0:
        new_bbox[1] = 1
    if new_bbox[2] > img.shape[1]:
        new_bbox[2] = img.shape[1] - 1
    if new_bbox[3] > img.shape[0]:
        new_bbox[3] = img.shape[0] - 1

    return img[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]], new_bbox


def show_image(img, win_name="show", delay=0):

    """easy imshow"""

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(delay=delay)


def resize_image(img, height=None, width=None, letterbox=False, lb_color=(128, 128, 128), inter=cv2.INTER_AREA):

    """
    img        -  np.array
    height     -  new H
    width      -  new W
    letterbox  -  add areas to lowest side for escaping image distortion
    lb_color   -  color of areas, default is gray
    """

    if letterbox:
        h, w = img.shape[:2]
        if h > w:
            border_y = 0
            border_x = round((h - w + .5) / 2.)
        else:
            border_x = 0
            border_y = round((w - h + .5) / 2.)
        img = cv2.copyMakeBorder(img, top=border_y, bottom=border_y, left=border_x, right=border_x,
                                 borderType=cv2.BORDER_CONSTANT, value=lb_color)

    dim = (width, height)
    (h, w) = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    if height is None:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(img, dim, interpolation=inter)


def rotate_image(img, ang):

    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, ang, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)


def warp_image(img, pts):
    """
    create rotated image using four points [top_left, top_right, bot_right, bot_left] absolute coords.

    Usage:

    roi = [(x_min, ymin), (x_max, y_max)]
    warped_img = warp_image(frame, np.array(eval(str([(roi[0][0], roi[0][1]), (roi[1][0], roi[0][1]),
                                                      (roi[1][0], roi[1][1]), (roi[0][0], roi[1][1])])),
                                            dtype="float32"))

    """
    # pts = np.array(eval(str(pts)), dtype="float32")
    tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]

    # tl, tr, br, bl = self.pts[0], self.pts[1], self.pts[2], self.pts[3]

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)

    # return the warped image
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))


def stack_images_for_show(imgs_list, stack_shape, row_max=0, messages=None):

    """
    create image of stacked images for show classifier/detector/reID

    imgs_list    -  list of images path
    #TODO add list of np.arrays

    stack_shape  -  all images should to be with the same shape
    row_max      - max images number in row
    messages     - list of strings for each image accordingly
    """

    if type(imgs_list[0]) is str:
        imgs_list = [resize_image(cv2.imread(x), height=stack_shape, width=stack_shape) for x in imgs_list]

    # get rows images
    imgs_lists = [imgs_list[i:i + row_max] for i in range(0, len(imgs_list), row_max)]
    if messages is not None:
        messages = [messages[i:i + row_max] for i in range(0, len(messages), row_max)]

    # add images to last batch
    while len(imgs_lists[-1]) != len(imgs_lists[0]):
        imgs_lists[-1].append(np.full(tuple(imgs_list[0].shape), 255, dtype='uint8'))

    # create line
    bl_l_shape = list(imgs_list[0].shape)
    bl_l_shape[0] = 20
    bl_l_shape[1] = stack_shape
    black_line_w = np.full(tuple(bl_l_shape), 255, dtype='uint8')
    bl_l_shape[0] = stack_shape + 20
    bl_l_shape[1] = 4
    black_line_h = np.full(tuple(bl_l_shape), 255, dtype='uint8')

    stacked_img = None
    for row_n, imgs_list in enumerate(imgs_lists):

        # stack row images
        stacked_row = None
        for n, img in enumerate(imgs_list):
            if stacked_row is None:

                stack = np.vstack((img, black_line_w))

                if messages is not None:
                    try:
                        cv2.putText(stack, "{}".format(messages[row_n][n]),
                                    (int(stack.shape[1]*0.3), int(stack.shape[0]*0.99)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 1)
                    except:
                        pass

                stacked_row = stack
                continue

            stack = np.vstack((img, black_line_w))

            if messages is not None:
                try:
                    cv2.putText(stack, "{}".format(messages[row_n][n]),
                                (int(stack.shape[1] * 0.3), int(stack.shape[0] * 0.99)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], 1)
                except:
                    pass

            stack = np.hstack((black_line_h, stack))
            stacked_row = np.hstack((stacked_row, stack))

        # stack rows
        if len(imgs_lists) > 1:

            if stacked_img is None:
                stacked_img = stacked_row
                continue
            stacked_img = np.vstack((stacked_img, stacked_row))

        else:
            stacked_img = stacked_row

    return stacked_img


# ----------------  Service Protection

class License:

    """
    check generated license key from license server. Required internet!
    license_data  -  base64 encoded string

    Usage:

        from sources.xlam import License
        print("Check license...")
        license_data = "zdarova chuvak"
        lic = License(license_data)
        if lic.check() is False:
            print("License not found.")
            exit(2)
        print("License found.")

    """

    __server: str
    __pass: str

    def __init__(self, license_data):

        data = base64.decodebytes(bytes(license_data, 'utf-8'))
        data = msgpack.unpackb(data, use_list=False, raw=False)

        self.__server = data["Server"]
        self.__pass = data["Pass"]

    def check(self):
        mid = self.__read_machineid()
        lic = self.__read_license()

        if mid is None or lic is None:
            return False

        data = {'uuid': mid, 'license': lic}
        resp = requests.post(self.__server, json=data)
        # print(resp.status_code, resp.text)
        if resp.status_code != 200:
            return False

        if resp.text != self.__pass:
            return False

        return True

    def __read_machineid(self):
        mid = None
        with open('/var/lib/dbus/machine-id', 'r') as file:
            mid = file.readline().strip()

        return mid

    def __read_license(self):
        lic = None
        with open("license", 'rb') as file:
            lic = base64.encodebytes(file.read().strip()).decode('utf-8')

        return lic


class HwBind:

    """
    check hardware for protection server.
    use base64 encoded NetCard/UsbDevice serial number

    Usage:
        # 1. open terminal --> ~$ ip a
        or find usb devices like keyboard/USB drive serial number
        # 2. find something like 'enp0s25' or wls1
        # 3. find 1c:61:3d:4b:54:12 serial number
        # 4. encode it via online base64 encoder
        # 5. put result to 'hw_key' value


        import base64
        from sources.xlam import HwBind
        print("Check hard ware...")
        hw_key = "zdarova chuvak esche raz!"
        hw = HwBind()
        if hw.check(base64.b64decode(hw_key).decode("utf-8")) is False:
            print('Hard ware check failed.')
            exit(0)
        print("Ok.")
    """

    def check(self, serial):
        for hwaddr in self.hwaddrs():
            if hwaddr == serial:
                return True

        for disk in self.disks():
            if disk.serial == serial:
                return True

        return False

    def disks(self):
        disks = []

        for filename in glob.glob("/sys/block/*/dev"):
            disk = Disk(filename)
            disks.append(disk)

        return disks

    def hwaddrs(self):
        hwaddrs = []

        # read interfaces names
        for addrpath in glob.glob("/sys/class/net/*/address"):
            with open(addrpath, 'r') as file:
                hwaddrs.append(file.read(17))

        return hwaddrs


class Disk:
    # dev_name: str
    # major_minor: str
    # serial: str

    def __init__(self, filename):
        # read major_minor number from /sys/block/*/dev
        with open(filename, "r") as file:
            self.major_minor = file.read().strip()

        # extract sd[a-z] from /sys/block/*/dev
        self.dev_name = filename.split("/")[2]

        # read serial number from /run/udev/data/b${major:minor}
        self.serial = self.__serial()

    def __serial(self):
        filename = "/run/udev/data/b{}".format(self.major_minor)
        serial = None
        with open(filename, "r") as file:
            for line in file:
                if line.startswith("E:ID_SERIAL_SHORT="):
                    serial = line.lstrip("E:ID_SERIAL_SHORT=").rstrip("\n")
                    break

        return serial

    def __str__(self):
        return "{self.dev_name}:{self.serial}".format(self=self)

    def equal_serialnumber(self, serial):
        return self.serial == serial


# ----------------  Other

class MotionDetector:

    """
    Detect motion on image via difference between masks of key frame and other next frame

    key_frame      -  np.array (frame without objects)
    binary_thresh  -  0/255 more than this thresh value will be on mask
    area_thresh    -  minimal white blob area on difference image

    """

    def __init__(self, key_frame, binary_thresh=80, area_thresh=500, debug=False):

        self.key_frame = cv2.GaussianBlur(cv2.cvtColor(key_frame, cv2.COLOR_BGR2GRAY), (3, 3), 2)
        self.binary_thresh = binary_thresh
        self.area_thresh = area_thresh
        self.debug = debug

    @staticmethod
    def grab_contours(cnts):
        # if the length the contours tuple returned by cv2.findContours
        # is '2' then we are using either OpenCV v2.4, v4-beta, or
        # v4-official
        if len(cnts) == 2:
            cnts = cnts[0]

        # if the length of the contours tuple is '3' then we are using
        # either OpenCV v3, v4-pre, or v4-alpha
        elif len(cnts) == 3:
            cnts = cnts[1]

        # otherwise OpenCV has changed their cv2.findContours return
        # signature yet again and I have no idea WTH is going on
        else:
            raise Exception(("Contours tuple must have length 2 or 3, "
                             "otherwise OpenCV changed their cv2.findContours return "
                             "signature yet again. Refer to OpenCV's documentation "
                             "in that case"))

        # return the actual contours array
        return cnts

    def detect_motion(self, img):

        results = []

        gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 2)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imh, imw = gray.shape[:2]

        frameDelta = cv2.absdiff(self.key_frame, gray)

        threshed = cv2.threshold(frameDelta, self.binary_thresh, 255, cv2.THRESH_BINARY)[1]
        threshed = cv2.dilate(threshed, None, iterations=3)

        cnts = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = self.grab_contours(cnts)

        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < self.area_thresh or cv2.contourArea(c) > 1200:
                continue

            # compute the bounding box for the contour, draw it on the frame, and update the text+
            x, y, w, h = cv2.boundingRect(c)
            # results.append([x, y, w, h])
            results.append([x / imw, y / imh, w / imw, h / imh])

        if self.debug:

            show_image(self.key_frame, "key_frame", delay=1)
            show_image(frameDelta, "delta", delay=1)
            show_image(threshed, "threshed", delay=1)

            if len(results):
                for box in results:
                    box = xywh_to_xyxy(img.shape[0], img.shape[1], box)
                    img = draw_bbox(img, box, bbox_color=(100, 10, 10))

            # stacked_img = stack_images_one_row(imgs_list=[self.key_frame, frameDelta, threshed, gray], stack_shape=224,
            #                      messages=["keyframe", "delta", "threshed", "motions"])

            show_image(img, "motions")

            # show_image(stacked_img)

        return results


class Reader:

    def __init__(self, name, src, type):

        self.name = name
        self.src = src
        self.type = type

        self.connected = False

        self.q = multiprocessing.Queue(maxsize=1)
        self.proc = None

        self.connect_and_start_read()

    def read_rtsp(self, name, src):

        try:

            proc_out = {"name": name}

            cap = cv2.VideoCapture(src)

            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

            proc_out["meta"] = {"h": h, "w": w, "fps": fps, "fourcc": fourcc}

            while True:

                status, frame = cap.read()
                proc_out["status"], proc_out["frame"] = status, frame
                proc_out["ts"] = time.time()

                self.q.put(proc_out)

        except:
            self.q.put({"name": name, "error": traceback.format_exc()})
            sys.exit()

    def read_from_files(self, name, src, type):

        files_list = []
        if type == "directory":
            files_list.extend(get_filelist(src, ["jpg", "png"]))
            files_list.extend(get_filelist(src, ["mp4", "avi", "mkv"]))

        elif type == "file":
            files_list.append(src)

        try:

            proc_out = {"name": name}

            for n, file in enumerate(files_list):

                if file.split(".")[-1] in ["mp4", "avi", "mkv"]:

                    cap = cv2.VideoCapture(file)

                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

                    proc_out["meta"] = {"h": h, "w": w, "fps": fps, "fourcc": fourcc}

                    while True:
                        status, frame = cap.read()
                        proc_out["status"], proc_out["frame"] = status, frame
                        proc_out["ts"] = time.time()

                        self.q.put(proc_out)

                elif file.split(".")[-1] in ["jpg", "png"]:

                    cap = cv2.VideoCapture(file)
                    status, frame = cap.read()
                    proc_out["status"], proc_out["frame"] = status, frame
                    self.q.put(proc_out)

            self.q.put({"status": "finished", "frame": 0})

        except:
            self.q.put({"name": name, "error": traceback.format_exc()})
            sys.exit()

    def get_frame(self):

        empty_q_time = 0.

        frame = None
        frame_meta = None

        while frame is None:

            if not self.proc.is_alive():
                self.connected = False
                self.kill()
                self.connect_and_start_read()
                return {"frame": frame, "frame_meta": frame_meta}

            if self.q.empty():
                empty_q_time += 0.1
                time.sleep(0.1)

                if empty_q_time > 15:
                    self.connected = False
                    self.kill()
                    self.connect_and_start_read()
                    return {"frame": frame, "frame_meta": frame_meta}

                continue

            out = self.q.get()

            if "error" in out:
                self.connected = False
                print("\nREADER PROCESS ERROR:\n", out["error"])
                self.kill()
                self.connect_and_start_read()
                return {"error": out["error"]}

            elif out["status"] is False:
                self.connected = False
                print("\nREADER FALSE STATUS\n")
                self.kill()
                self.connect_and_start_read()
                return {"frame": frame, "frame_meta": frame_meta}

            elif out["frame"] is None:
                self.connected = False
                print("\nREADER NONE FRAME\n")
                self.kill()
                self.connect_and_start_read()
                return {"frame": frame, "frame_meta": frame_meta}

            self.connected = True
            frame = out["frame"]
            frame_meta = out["meta"]

        return {"frame": frame, "frame_meta": frame_meta}

    def kill(self):
        try:
            os.kill(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def connect_and_start_read(self):

        if self.type in ["webcam", "rtsp_stream"]:
            self.proc = multiprocessing.Process(target=self.read_rtsp, args=(self.name, self.src,))

        elif self.type in ["file", "directory"]:
            self.proc = multiprocessing.Process(target=self.read_from_files, args=(self.name, self.src, self.type,))

        self.proc.start()
        time.sleep(2)


class Writer:

    def __init__(self, file_name, fps, height, width, fourcc='mp4v'):
        self.wr = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*fourcc), int(fps), (width, height))

    def write_to_file(self, frame):
        self.wr.write(frame)

    def finish_writing(self):
        self.wr.release()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



# def multi_proc_generator():



if __name__ == "__main__":

    # UPDATE ALL xlam files

    main_xmal = "./xlam.py"
    xlams = []

    # search
    for root, dirs, files in os.walk("/home/mikhail/PycharmProjects"):
        for file in files:
            if "xlam" in file and file.endswith(".py"):
                file_path = os.path.join(root, file)
                if file_path == main_xmal:
                    continue
                xlams.append(file_path)
                print("{}".format(len(xlams)), file_path)

    # update
    for xlam in xlams:
        try:
            shutil.copy(main_xmal, xlam)
        except shutil.SameFileError:
            continue

