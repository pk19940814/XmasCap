#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : XmasCap.py
# @Author: Kom
# @Date  : 2017/12/26
# @Desc  :


from __future__ import print_function
import cv2
import sys
import urllib
import urllib2
import uuid
import os
import numpy as np

'''
运行命令： python face_detect.py head_url hat_path 
参考实例：nowcoder.com/discuss/65328
'''


def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    # 底图范围
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # 覆盖图范围
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # 无重合直接退出
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # 覆盖的图有透明通道
    if img_overlay.shape[2] == 4:
        alpha_mask = img_overlay[:, :, 3] / 255.0
        channels = img.shape[2]
        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha

        # 根据透明度融合
        for c in range(channels):
            img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] + alpha_inv * img[y1:y2, x1:x2, c])
    else:
        # 无透明度，直接覆盖掉
        img[y1:y2, x1:x2] = img_overlay[y1o:y2o, x1o:x2o]


imagePath = None
try:
    root_dir = '/home/nowcoder/bang/'

    # 头像底图
    imageUrl = sys.argv[1]
    cascPath = root_dir + 'haarcascade_frontalface_default.xml'
    # 帽子图片
    hatPath = sys.argv[2]

    faceCascade = cv2.CascadeClassifier(cascPath)

    # 把底图从网上下载到本地
    imagePath = str(uuid.uuid4()).replace('-', '') + '.jpeg'
    urllib.urlretrieve(imageUrl, imagePath)

    image = cv2.imread(imagePath)
    if image.shape[0] <> 300 or image.shape[1] <> 300:
        image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_CUBIC)

    hat = cv2.imread(hatPath, -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        # flags = cv2.CV_HAAR_SCALE_IMAGE
    )

    if len(faces) == 0:
        # 没找到人脸，中间靠上位置直接放一个帽子
        overlay_image_alpha(image, hat, (image.shape[1] / 2 - hat.shape[1] / 2, -10))
    else:
        tx = ty = th = tw = 0
        # 找到最大的人脸区域
        for (x, y, w, h) in faces:
            if w * h > tw * th:
                tx = x
                ty = y
                tw = w
                th = h
        # 根据人脸大小调整帽子大小
        hat = cv2.resize(hat, (int(2 * tw), int(2 * tw * hat.shape[0] / hat.shape[1])), interpolation=cv2.INTER_CUBIC)

        # 把帽子画到底图
        overlay_image_alpha(image, hat, (tx - (hat.shape[1]) / 4 + 20 * tw / 300, ty - hat.shape[0] * 3 / 5))

    # debug画出人脸区域
    # cv2.rectangle(image, (tx, ty), (tx+w, ty+h), (0, 255, 0), 2)

    target = root_dir + 'tmp/' + str(uuid.uuid4()).replace('-', '') + '.png'
    cv2.imwrite(target, image)
    # 清理底图
    os.remove(imagePath)
    print(target, end='')
except Exception as e:
    print('ERROR', end='')
finally:
    if os.path.exists(imagePath):
        os.remove(imagePath)
        pass
