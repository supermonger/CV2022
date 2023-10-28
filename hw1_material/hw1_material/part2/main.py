import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter

def get_cost(input, target):
    return np.sum(np.abs(input.astype(np.int32) - target.astype(np.int32)))

def RGB_weighted_average(image ,R, G, B):
    return  image[:,:,0] * R + image[:,:,1] * G + image[:,:,2] * B


def main():
    num = 2
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default=f'./testdata/{num}.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.int32)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int32)
    
    ### TODO ###
    # 1
    if num == 1:
        JBF = Joint_bilateral_filter(2, 0.1)
        setting = [[0.0,0.0,1.0],[0.0,1.0,0.0],[0.1,0.0,0.9],[0.1,0.4,0.5],[0.8,0.2,0.0]]
    elif num == 2:
        JBF = Joint_bilateral_filter(1, 0.05)
        setting = [[0.1,0.0,0.9],[0.2,0.0,0.8],[0.2,0.8,0.0],[0.4,0.0,0.6],[1.0,0.0,0.0]]
    
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb)

    def calculate(R, G, B):
        img_gray_modify = RGB_weighted_average(img_rgb ,R, G, B).astype(np.int32)
        jbf_out =  JBF.joint_bilateral_filter(img_rgb, img_gray_modify).astype(np.int32)
        return get_cost(jbf_out, bf_out)

    jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.int32)
    cost = get_cost(jbf_out, bf_out)
    print(cost)
    highest_cost = cost
    lowest_cost = cost

    for Red, Green, Blue in setting:
        print((Red, Green, Blue))
        cur_cost = calculate(Red, Green, Blue)
        print(cur_cost)
        if cur_cost > highest_cost:
            highest_cost = cur_cost
            hr, hg, hb = Red, Green, Blue
        elif cur_cost < lowest_cost:
            lowest_cost = cur_cost
            lr, lg, lb = Red, Green, Blue

    # 2
    img_gray_modify = RGB_weighted_average(img_rgb ,hr, hg, hb).astype(np.uint8)
    jbf_out =  JBF.joint_bilateral_filter(img_rgb, img_gray_modify).astype(np.uint8)
    print(jbf_out.shape)
    jbf_out = cv2.cvtColor(jbf_out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'highest_filtered_image_{num}.png', jbf_out)
    cv2.imwrite(f'highest_grayscaled_image_{num}.png', img_gray_modify)

    img_gray_modify = RGB_weighted_average(img_rgb ,lr, lg, lb).astype(np.uint8)
    jbf_out =  JBF.joint_bilateral_filter(img_rgb, img_gray_modify).astype(np.uint8)
    print(jbf_out.shape)
    jbf_out = cv2.cvtColor(jbf_out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'lowest_filtered_image_{num}.png', jbf_out)
    cv2.imwrite(f'lowest_grayscaled_image_{num}.png', img_gray_modify)

    print((hr, hg, hb))
    print((lr, lg, lb))


if __name__ == '__main__':
    main()