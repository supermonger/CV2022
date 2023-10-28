from turtle import width
import numpy as np
import cv2


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_guassaian_images(self, image):
        gaussian_images = []
        height, width = image.shape
        gaussian_images.append(image)
        for i in range(self.num_guassian_images_per_octave-1):
            img = cv2.GaussianBlur(image, (0, 0), self.sigma**(i+1))
            gaussian_images.append(img)
        scaled_image = cv2.resize(gaussian_images[-1], (int(width * 0.5), int(height * 0.5)), interpolation=cv2.INTER_NEAREST)
        downsample_height, downsample_width = scaled_image.shape
        gaussian_images.append(scaled_image)
        for i in range(self.num_guassian_images_per_octave-1):
            img = cv2.GaussianBlur(scaled_image, (0, 0), self.sigma**(i+1))
            gaussian_images.append(img)
        return gaussian_images, height, width, downsample_height, downsample_width

    def get_dog_images(self, gaussian_images):
        dog_images = []
        for img1, img2 in zip(gaussian_images[0:-1], gaussian_images[1:]):
            if img1.shape == img2.shape:
                img_subtract = cv2.subtract(img1, img2)
                dog_images.append(img_subtract)
        return dog_images

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images, height, width, downsample_height, downsample_width = self.get_guassaian_images(image)
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = self.get_dog_images(gaussian_images)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for i in range(2 * self.num_DoG_images_per_octave - 2):
            if dog_images[i].shape == dog_images[i+1].shape and dog_images[i+1].shape == dog_images[i+2].shape:
                h, w = dog_images[i].shape
                for j in range(1, h - 1):
                    for k in range(1, w - 1):
                        center = dog_images[i+1][j, k]
                        if abs(center) > self.threshold:
                            center_img = dog_images[i+1][j-1:j+2, k-1:k+2]
                            pre_img = dog_images[i][j-1:j+2, k-1:k+2]
                            next_img = dog_images[i+2][j-1:j+2, k-1:k+2]
                            cube = np.dstack((pre_img, center_img, next_img))
                            if center == np.max(cube) or center == np.min(cube):
                                if h == height and w == width:
                                    keypoints.append([j, k])
                                elif h == downsample_height and w == downsample_width:
                                    keypoints.append([2*j, 2*k])

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(np.array(keypoints), axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]
        return keypoints
