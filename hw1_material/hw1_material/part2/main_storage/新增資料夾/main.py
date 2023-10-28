import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter
from matplotlib import pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/2_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    rgb_list = np.genfromtxt(args.setting_path, delimiter=',', skip_header=1, skip_footer=1)
    sigma_values = np.genfromtxt(args.setting_path, delimiter=',', skip_header=6, usecols=(1, 3))
    sigma_s = int(sigma_values[0])
    sigma_r = sigma_values[1]
    print(f'{sigma_s}---{sigma_r}')
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
    cost = np.sum(np.abs(jbf_out.astype('int32') - bf_out.astype('int32')))
    print(f'Cost 1: {cost}')
    for i in range(5):
        print((rgb_list[i, 0],rgb_list[i, 1],rgb_list[i, 2]))
        img_gray = (img_rgb[:, :, 0] * rgb_list[i, 0] + img_rgb[:, :, 1] * rgb_list[i, 1] + img_rgb[:, :, 2] * rgb_list[i, 2]).astype(np.uint8)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
        cost = np.sum(np.abs(jbf_out.astype(np.int32) - bf_out.astype(np.int32)))
        print(f'Cost {i + 2}: {cost}')

    img_gray = (img_rgb[:, :, 0] * rgb_list[0, 0] + img_rgb[:, :, 1] * rgb_list[0, 1] + img_rgb[:, :, 2] * rgb_list[0, 2]).astype(np.uint8)
    jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
    plt.figure(1)
    plt.imshow(jbf_out)
    plt.figure(2)
    plt.imshow(img_gray, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()