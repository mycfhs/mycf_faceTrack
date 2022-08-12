import logging
import os
import time
import uuid
from operator import itemgetter
import cv2


def mkdir(path):
    path.strip()
    path.rstrip('\\')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def save_to_file(root_dic, tracker):
    filter_face_addtional_attribute_list = []
    for item in tracker.face_addtional_attribute:
        if item[2] < 1.4 and item[4] < 1:  # recommended thresold value
            filter_face_addtional_attribute_list.append(item)
    if len(filter_face_addtional_attribute_list) > 0:
        score_reverse_sorted_list = sorted(filter_face_addtional_attribute_list, key=itemgetter(4))
        mkdir(root_dic)
        cv2.imwrite("{0}/{1}.jpg".format(root_dic, str(uuid.uuid1())), score_reverse_sorted_list[0][0])


