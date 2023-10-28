import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def RANSAC(kp_src, kp_dst, matches, iter_num, threshold, sample_num=4):
    print("RANSAC")
    best_H = np.eye(3)
    best_inlier_num, match_num = 0, len(matches)
    print(f'There is {match_num} matches')
    match_dst_sort = np.argsort([matches[i].distance for i in range(match_num)])
    src_idxs = [matches[i].queryIdx for i in range(match_num)]
    dst_idxs = [matches[i].trainIdx for i in range(match_num)]
    src_coordinate = np.concatenate((kp_src[src_idxs, 0:1], kp_src[src_idxs, 1:2], np.ones((match_num, 1))), axis=1)
    dst_coordinate = np.concatenate((kp_dst[dst_idxs, 0:1], kp_dst[dst_idxs, 1:2], np.ones((match_num, 1))), axis=1)
    for i in tqdm(range(iter_num)) :
        sample_idxs = random.sample(list(match_dst_sort[:30]), sample_num)
        
        this_H = solve_homography(src_coordinate[sample_idxs, 0:2], dst_coordinate[sample_idxs, 0:2])
        
        src_on_dst, dst_on_src = np.dot(this_H, src_coordinate.T).T, np.dot(np.linalg.inv(this_H), dst_coordinate.T).T
        src_on_dst, dst_on_src = src_on_dst / src_on_dst[:,2:3], dst_on_src / dst_on_src[:,2:3]
        dist = np.linalg.norm(src_coordinate-dst_on_src, axis=1)**2 +\
                np.linalg.norm(dst_coordinate-src_on_dst, axis=1)**2
        inlier_num = np.sum(dist < threshold)
        if inlier_num > best_inlier_num:
            best_inlier_num = inlier_num
            best_H = this_H
    print(f'max inlier number is {best_inlier_num}')
    return best_H

def stitch_image(img1, img2): # base on the image center
    output = np.zeros(img1.shape)
    img1_non_zero_mask, img2_non_zero_mask = np.all(img1 != 0, axis=2), np.all(img2 != 0, axis=2)
    img1_center_y, img1_center_x = np.nonzero(img1_non_zero_mask)
    img2_center_y, img2_center_x = np.nonzero(img2_non_zero_mask)
    img1_center_y, img1_center_x = np.average(img1_center_y), np.average(img1_center_x)
    img2_center_y, img2_center_x = np.average(img2_center_y), np.average(img2_center_x)
    overlap_mask = img1_non_zero_mask * img2_non_zero_mask
    overlap_y, overlap_x = np.nonzero(overlap_mask)
    nonoverlap_y, nonoverlap_x = np.nonzero(overlap_mask == 0)
    # weight1, weight2 = ((overlap_y-img2_center_y)**4 + (overlap_x-img2_center_x)**4), ((overlap_y-img1_center_y)**4 + (overlap_x-img1_center_x)**4)
    weight1, weight2 = (overlap_x-img2_center_x)**4, (overlap_x-img1_center_x)**4
    weight1, weight2 = (weight1/(weight1 + weight2)).reshape((-1,1)), (weight2/(weight1 + weight2)).reshape((-1,1))
    output[overlap_y, overlap_x, :] = img1[overlap_y, overlap_x, :]*weight1 + img2[overlap_y, overlap_x, :]*weight2
    output[nonoverlap_y, nonoverlap_x, :] = img1[nonoverlap_y, nonoverlap_x, :] + img2[nonoverlap_y, nonoverlap_x, :]
    return output

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    canvas = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    out = None

    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        kp1, kp2 = cv2.KeyPoint_convert(kp1), cv2.KeyPoint_convert(kp2)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des2, des1)
        # TODO: 2. apply RANSAC to choose best H
        H = RANSAC(kp2, kp1, matches=matches, iter_num=32000, threshold=3, sample_num=4)

        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, H)
        
        # TODO: 4. apply warping
        out = warping(im2, canvas, last_best_H, 0, canvas.shape[0], 0, canvas.shape[1], 'b')
        dst = stitch_image(dst, out)
    return dst

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    # imgs = [cv2.imread('../resource/siclab2{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)