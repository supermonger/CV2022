import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        
        ### TODO ###
        h, w, _ = img.shape
        output = np.zeros_like(img).astype(np.float64)
        den = np.zeros_like(img).astype(np.float64)
        
        # spatial kernel
        x, y = np.meshgrid(np.arange(self.wndw_size) - self.pad_w, np.arange(self.wndw_size) - self.pad_w)
        kernel_s = np.exp(-(x**2+y**2)/(2*self.sigma_s**2))
        # look up table for range kernel
        look_up_table = np.exp(- np.arange(256) * np.arange(256)/ 255**2 /(2*self.sigma_r**2))
        
        # main
        # guidance is grayscale
        if guidance.ndim == 2:
            for i in range(self.wndw_size):
                for j in range(self.wndw_size):
                    GrGs = kernel_s[i, j] * look_up_table[abs(padded_guidance[i:h+i, j:w+j] - guidance)]
                    for k in range(3):
                        output[:,:,k] = output[:,:,k] + GrGs * padded_img[i:h+i, j:w+j, k]
                        den[:,:,k] = den[:,:,k] + GrGs
        
        # guidance is RGB
        elif guidance.ndim == 3:
            for i in range(self.wndw_size):
                for j in range(self.wndw_size):
                    GrGs = kernel_s[i,j] * \
                        look_up_table[abs(padded_guidance[i:h+i, j:w+j, 0] - guidance[:, :, 0])] * \
                        look_up_table[abs(padded_guidance[i:h+i, j:w+j, 1] - guidance[:, :, 1])] * \
                        look_up_table[abs(padded_guidance[i:h+i, j:w+j, 2] - guidance[:, :, 2])]
                    for k in range(3):
                        output[:,:,k] = output[:,:,k] + GrGs * padded_img[i:h+i, j:w+j, k]
                        den[:,:,k] = den[:,:,k] + GrGs
                        
        output = output / den
        return np.clip(output, 0, 255).astype(np.uint8)