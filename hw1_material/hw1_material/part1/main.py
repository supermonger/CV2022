import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian

def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def normalize_dog(dog_image):
    return (255*(dog_image - np.min(dog_image))/(np.max(dog_image) - np.min(dog_image))).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=5.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./data/1.png', help='path to input image')
    parser.add_argument('--image_path2', default='./data/2.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img1 = cv2.imread(args.image_path, 0).astype(np.float32)
    img2 = cv2.imread(args.image_path2, 0).astype(np.float32)

    ### TODO ###
    # 1
    DoG = Difference_of_Gaussian(args.threshold)

    gaussian_images, _, _, _, _ = DoG.get_guassaian_images(img1)
    dog_images = DoG.get_dog_images(gaussian_images)

    for i, image in enumerate(dog_images):
        path = f'DoG{int(i/4)+1}_{i%4+1}.png'
        print(path)
        image = normalize_dog(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, image)
    
    # 2
    for threshold in [2,5,7]:
        DoG = Difference_of_Gaussian(threshold)
        key_points = DoG.get_keypoints(img2)
        rgb_img = cv2.imread(args.image_path2)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        for y, x in key_points:
            cv2.circle(rgb_img, (x, y), 5, (0, 255, 0), -1)
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'threshold{threshold}.png', bgr_img)

if __name__ == '__main__':
    main()