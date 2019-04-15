# coding=utf-8
import argparse
import csv
import os
import re
import shutil
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from pytesseract import pytesseract

from cropper import cropper

sys.path.append("./crnn/crnn.pytorch")
sys.path.append("./ctpn")
sys.path.append(os.getcwd())
from ctpn.nets import model_train as model
from ctpn.utils.rpn_msr.proposal_layer import proposal_layer
from ctpn.utils.text_connector.detectors import TextDetector


from crnn.crnnport import CRNNRecognizer

tf.app.flags.DEFINE_string('test_data_path', 'ctpn/data/demo/', '')
tf.app.flags.DEFINE_string('output_path', 'ctpn/data/res/', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'ctpn/checkpoints_mlt/', '')
FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


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


def main(argv=None):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    args = vars(ap.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            print('===============')
            print(args["image"])
            start = time.time()
            try:
                im = cv2.imread(args["image"])[:, :, ::-1]
            except:
                print("Error reading image {}!".format(args["image"]))

            img, (rh, rw) = resize_image(im)
            h, w, c = img.shape
            im_info = np.array([h, w, c]).reshape([1, 3])
            bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                   feed_dict={input_image: [img],
                                                              input_im_info: im_info})

            textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
            scores = textsegs[:, 0]
            textsegs = textsegs[:, 1:5]

            textdetector = TextDetector(DETECT_MODE='H')
            boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
            boxes = np.array(boxes, dtype=np.int)

            cost_time = (time.time() - start)
            print("cost time: {:.2f}s".format(cost_time))

            for i, box in enumerate(boxes):
                cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                              thickness=2)
            # img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
            img = img[:, :, ::-1]
            # cv2.imshow("frame",img)
            # cv2.waitKey(0)
            patches = cropper.crope(img,boxes)
            card_num = ""
            base = os.path.basename(args["image"])
            image_name = os.path.splitext(base)[0]

            year = ""
            mon = ""
            for patch in patches:
                filename = "1.png".format(os.getpid())
                cv2.imwrite(filename, patch)
                # load the image as a PIL/Pillow image, apply OCR, and then delete
                # the temporary file
                # text = pytesseract.image_to_string(Image.open(filename))
                text_tes = pytesseract.image_to_string(Image.open(filename), config='digits -psm 7')
                # crnn
                base_dir = './crnn/models/'
                model_path = base_dir + 'netCRNNcpu.pth'
                crnn_recog = CRNNRecognizer(model_path)

                text_crnn = crnn_recog.crnnRec(patch, use_gpu=0)
                print(text_crnn)
                text_crnn = str(text_crnn)

                digits_tes = re.sub("[^0-9/]", "", text_tes)
                digits_crnn = re.sub("[^0-9/]", "", text_crnn)

                if len(digits_crnn) == 16:
                    card_num = digits_crnn

                elif len(digits_tes) == 16:
                    card_num = digits_tes

                if len(digits_crnn) < 16:
                    if digits_crnn.find("/") !=  -1:
                        index = (digits_crnn.find("/"))
                        if index+3 <= len(digits_crnn):
                            year = digits_crnn[index-2:index]
                            mon = digits_crnn[index + 1:index + 3]
            with open('file.csv', mode='a') as employee_file:
                employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                employee_writer.writerow([image_name,card_num,year,mon])

                # if len([c for c in text if c.isdigit()]) > 10 and len([c for c in text if c.isdigit()]) < 17:
                #     print(text)
                #     # cv2.imshow("frame", patch)
                #     # cv2.waitKey(500)
                #     with open("Output.txt", "a") as text_file:
                #         text_file.write(args["image"]+" : "+text+" : crnn : "+text_crnn+"\n")


if __name__ == '__main__':
    tf.app.run()
