# -*-coding:utf-8-*-



import os
import cv2
import time
import numpy as np


def min_filter(img, r=15):
    """
    the minimum filter to generate dark channel map
    :param img:
    :param r:   the radium of min-filter
    :return:
    """
    return cv2.erode(img, np.ones((2 * r + 1, 2 * r + 1)))


def guidance(I, p, guided_r, eps=0.001):
    '''
    the guided-filter, which is refer to the Matlab code on the Internet
    :param I:           guidance image
    :param p:           the image to be refined
    :param guided_r:    the radius of guided-filter
    :param eps:
    :return:
    '''
    m_I = cv2.boxFilter(I, -1, (guided_r, guided_r))
    m_p = cv2.boxFilter(p, -1, (guided_r, guided_r))
    m_Ip = cv2.boxFilter(I * p, -1, (guided_r, guided_r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (guided_r, guided_r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (guided_r, guided_r))
    m_b = cv2.boxFilter(b, -1, (guided_r, guided_r))
    return m_a * I + m_b


def get_A(img):
    """
    get the atmospheric light
    :param img:
    :return:
    """
    hazy_img = img
    dark_img = min_filter(np.min(img, 2))
    bins = 2000
    ht = np.histogram(dark_img, bins)
    d = np.cumsum(ht[0]) / float(dark_img.size)
    for i in range(bins - 1, 0, -1):
        if d[i] <= 0.999:
            break
    pixel_list = hazy_img[dark_img > ht[1][i]]
    intensity_list = np.mean(pixel_list, 1)
    intensity_list = intensity_list.tolist()
    pixel_index = intensity_list.index(max(intensity_list))
    return pixel_list[pixel_index]  ###,intensity_list[pixel_index]


def get_t(img, guided_r=81, eps=0.001):
    """
    get the transmission map
    :param img:
    :param guided_r:
    :param eps:
    :return:
    """
    guidance_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    A = get_A(img)
    middle_img = np.zeros(img.shape)
    for i in range(3):
        middle_img[:, :, i] = img[:, :, i] / A[i]
    t = np.min(middle_img, 2)
    # the coarse transmission map
    t = 1 - 0.95 * t
    # the refined transmission map
    t = guidance(guidance_img, t, guided_r, eps)
    t = np.clip(t, 0.1, 1)
    return t


def dehaze(img):
    """
    haze removal by Eq.5 in paper
    :param img:
    :return:
    """
    A = get_A(img)
    result = np.zeros(img.shape, dtype=np.float)
    t = get_t(img)

    for i in range(3):
        result[:, :, i] = (img[:, :, i] - A[i] * (1 - t)) / t
    return result


def dehaze_dir(file_dir, out_dir):
    """
    :param file_dir:    the folder of hazy images
    :param out_dir:     the folder of results
    :return:
    """
    files_list = os.listdir(file_dir)
    for i, img_name in enumerate(files_list):
        start_time = time.time()

        in_img_name = img_name
        in_img_dir = os.path.join(file_dir, in_img_name)
        out_img_dir = os.path.join(out_dir, in_img_name)

        hazy_img = cv2.imread(in_img_dir)
        dehazed_img = dehaze(hazy_img)
        cv2.imwrite(out_img_dir, dehazed_img)
        one_img_time = time.time() - start_time
        print("The {0} img named {1} is processed, time cost is: {2}".format(i, img_name, one_img_time))


if __name__ == '__main__':
    file_dir = ""
    out_dir = ""
    os.makedirs(out_dir, exist_ok=True)
    dehaze_dir(file_dir, out_dir)

