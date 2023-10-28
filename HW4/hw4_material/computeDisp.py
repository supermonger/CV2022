import numpy as np
import cv2
import cv2.ximgproc as xip

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il, Ir = Il.astype(np.float32), Ir.astype(np.float32)
    window_size = 3 # should be odd number
    pad_size = int((window_size-1)/2)
    print(f'max disparity is {max_disp}')
    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
    padded_Il = cv2.copyMakeBorder(Il, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT).astype(np.int32)
    padded_Ir = cv2.copyMakeBorder(Ir, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT).astype(np.int32)
    binary_pattern_l, binary_pattern_r = np.ones((h, w, ch*(window_size**2 - 1))).astype(bool), np.ones((h, w, ch*(window_size**2 - 1))).astype(bool)
    
    # calculate binary pattern
    counter = 0
    for i in range(window_size):
        for j in range(window_size):
            if i==int(window_size/2) and j==int(window_size/2):
                continue
            else:
                binary_pattern_l[:,:,counter:counter+ch] = (Il >= padded_Il[i:i+h, j:j+w, :])
                binary_pattern_r[:,:,counter:counter+ch] = (Ir >= padded_Ir[i:i+h, j:j+w, :])
                counter += ch
    Il_to_Ir_cost_of_disparity, Ir_to_Il_cost_of_disparity = np.zeros((h, w, max_disp)), np.zeros((h, w, max_disp))
    
    # calculate cost
    for disp in range(1, max_disp+1):
        temp_l, temp_r = binary_pattern_l[:, disp:w, :], binary_pattern_r[:, :w-disp, :]
        cost = np.count_nonzero(np.logical_xor(temp_l, temp_r), axis=2)
        # left to right (right as base)
        Il_to_Ir_cost_of_disparity[:, :cost.shape[1], disp-1] = cost
        Il_to_Ir_cost_of_disparity[:, cost.shape[1]:, disp-1] = cost[:,-1:]
        # right to left (left as base)
        Ir_to_Il_cost_of_disparity[:, w-cost.shape[1]:, disp-1] = cost
        Ir_to_Il_cost_of_disparity[:, :w-cost.shape[1], disp-1] = cost[:,0:1]
        
    Il_to_Ir_cost_of_disparity, Ir_to_Il_cost_of_disparity = Il_to_Ir_cost_of_disparity.astype(np.float32), Ir_to_Il_cost_of_disparity.astype(np.float32)
    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    filtered_Il_to_Ir_cost_of_disparity = np.zeros(Il_to_Ir_cost_of_disparity.shape).astype(np.float32)
    filtered_Ir_to_Il_cost_of_disparity = np.zeros(Ir_to_Il_cost_of_disparity.shape).astype(np.float32)
    for disp in range(max_disp):
        filtered_Il_to_Ir_cost_of_disparity[:,:,disp] = xip.jointBilateralFilter(Ir, Il_to_Ir_cost_of_disparity[:,:,disp], -1, 20, 20)
        filtered_Ir_to_Il_cost_of_disparity[:,:,disp] = xip.jointBilateralFilter(Il, Ir_to_Il_cost_of_disparity[:,:,disp], -1, 20, 20)
    
    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost
    # [Tips] Winner-take-all
    right_label = np.argmin(filtered_Il_to_Ir_cost_of_disparity, axis=2) + 1 
    left_label = np.argmin(filtered_Ir_to_Il_cost_of_disparity, axis=2) + 1
    right_label = right_label.astype(np.uint8)
    left_label = left_label.astype(np.uint8)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering

    # Left-right consistency check
    x_left, y_left = np.meshgrid(range(w), range(h))
    y_right, x_right = y_left, x_left - left_label
    same_mask = np.zeros((h,w)).astype(bool)
    bound_mask = x_right >= 0
    same_mask[bound_mask] = (left_label[y_left[bound_mask], x_left[bound_mask]] == right_label[y_right[bound_mask], x_right[bound_mask]])
    labels[same_mask] = left_label[same_mask]

    # Hole filling
    labels = fill_hole(labels, same_mask)

    # weighted median filter
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), labels.astype(np.uint8), 7)
    
    return labels.astype(np.uint8)

def fill_hole(labels, valid_mask):
    # find left nearest valid number
    h, w = labels.shape
    left_valid_disparity, right_valid_disparity = np.zeros((h, w)), np.zeros((h, w))
    for j in range(h):
        left_disparity, right_disparity = 10**3, 10**3
        for i in range(w):
            if not valid_mask[j, i]:
                left_valid_disparity[j, i] = left_disparity
            else:
                left_disparity = labels[j, i]
            if not valid_mask[j, w-1-i]:
                right_valid_disparity[j, w-1-i] = right_disparity
            else:
                right_disparity = labels[j, w-1-i]
    min_disparity = np.min(np.concatenate((np.expand_dims(left_valid_disparity, axis=2), np.expand_dims(right_valid_disparity, axis=2)), axis=2), axis=2)
    labels[np.logical_not(valid_mask)] = min_disparity[np.logical_not(valid_mask)].flatten()

    return labels