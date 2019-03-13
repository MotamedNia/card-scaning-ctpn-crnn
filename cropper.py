import glob

import cv2
import numpy as np

class cropper(object):

    @staticmethod
    def resize_image(img):
        img_size = img.shape
        im_size_min = np.min(img_size[0:2])
        im_size_max = np.max(img_size[0:2])

        im_scale = float(600) / float(im_size_min)
        if np.round(im_scale * im_size_max) > 1200:
            im_scale = float(1200) / float(im_size_max)
        new_h = int(img_size[0] * im_scale)
        new_w = int(img_size[1] * im_scale)

        new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
        new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

        re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return re_im, (new_h / img_size[0], new_w / img_size[1])

    @staticmethod
    def crope(img,boxes):
        print(img.shape)
        # img, (rh, rw) = resize_image(img)
        print(img.shape)

        patches = []
        for i, roi in enumerate(boxes):

            print(roi)
            path = img[int(roi[1]):int(roi[5]), int(roi[0]):int(roi[4])]
            # roi = np.array(roi)
            # cv2.polylines(img, [roi[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 0, 0),
            #               thickness=2)

            patch = cv2.cvtColor(path,cv2.COLOR_RGB2GRAY)

            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # path = clahe.apply(path)
            # path = cv2.GaussianBlur(path, (5, 5), 0)
            # path = cv2.threshold(path, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            patches.append(patch)

        return patches