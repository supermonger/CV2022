import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)  (v=x'  u=x  x'=Hx)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    u = np.hstack((u, np.ones((N, 1))))
    v = np.hstack((v, np.ones((N, 1))))
    A = np.zeros(N*3*9)
    
    A01, A02, A12 = (v[:,2:3]*u).flatten(), (v[:,1:2]*u).flatten(), (v[:,0:1]*u).flatten()
    index = np.array([[27*i, 1+27*i, 2+27*i] for i in range(N)]).flatten()
    A[index+3], A[index+6], A[index+9], A[index+15], A[index+18], A[index+21] = -A01, A02, A01, -A12, -A02, A12
    
    A = A.reshape((N*3, 9))
    
    # TODO: 2.solve H with A
    
    _, _, V = np.linalg.svd(A)
    # print(V[:,-1])
    H = V[-1,:].reshape((3,3)) / V[-1,-1]
    
    return H

def backward_interpolate(obj_indexs, source):
    obj_x, obj_y = obj_indexs[:,0], obj_indexs[:,1]
    x_ceil, y_ceil = np.ceil(obj_indexs[:,0]).astype(int), np.ceil(obj_indexs[:,1]).astype(int)
    x_floor, y_floor = np.floor(obj_indexs[:,0]).astype(int), np.floor(obj_indexs[:,1]).astype(int)
    x_ceil_diff, x_floor_diff = (x_ceil-obj_x).reshape((-1,1)), (obj_x-x_floor).reshape((-1,1))
    y_ceil_diff, y_floor_diff = (y_ceil-obj_y).reshape((-1,1)), (obj_y-y_floor).reshape((-1,1))
    output = source[y_floor, x_floor,:]*x_ceil_diff*y_ceil_diff + \
                source[y_floor, x_ceil,:] * x_floor_diff * y_ceil_diff + \
                source[y_ceil, x_floor,:] * x_ceil_diff * y_floor_diff + \
                source[y_ceil, x_ceil,:] * x_floor_diff * y_floor_diff
    return output

def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """
    h_src, w_src, _ = src.shape
    h_dst, w_dst, _ = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    src_y, src_x = np.meshgrid(range(h_src), range(w_src))
    src_y, src_x = np.expand_dims(src_y, axis=2), np.expand_dims(src_x, axis=2)
    dst_y, dst_x = np.meshgrid(range(h_dst), range(w_dst))
    dst_y, dst_x = np.expand_dims(dst_y, axis=2), np.expand_dims(dst_x, axis=2)

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    dst_coordinate = np.concatenate((dst_x, dst_y, np.ones((w_dst, h_dst, 1))), axis=2)
    src_coordinate = np.concatenate((src_x, src_y, np.ones((w_src, h_src, 1))), axis=2)

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        dst_coordinate = dst_coordinate[xmin:xmax+1, ymin:ymax+1, :]
        dst_coordinate = dst_coordinate.reshape((-1,3))
        dst_on_source = np.dot(H_inv, dst_coordinate.T).T
        
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        dst_on_source /= dst_on_source[:,2:3]
        x_max_mask, x_min_mask = dst_on_source[:,0:1]<w_src-1, dst_on_source[:,0:1]>0
        y_max_mask, y_min_mask = dst_on_source[:,1:2]<h_src-1, dst_on_source[:,1:2]>0
        mask = np.ones((dst_on_source.shape[0], 1), dtype=bool)
        mask = np.concatenate((x_max_mask*x_min_mask, y_max_mask*y_min_mask, mask), axis=1)
        mask = np.all(mask, axis=1)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        dst_on_source, dst_coordinate = dst_on_source[mask,...], dst_coordinate[mask,...].astype(int)

        # TODO: 6. assign to destination image with proper masking
        pixel_values_after_interpolation = backward_interpolate(dst_on_source[:,0:2], src)
        dst[dst_coordinate[:,1], dst_coordinate[:,0], :] = pixel_values_after_interpolation

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        src_coordinate = src_coordinate[xmin:xmax+1, ymin:ymax+1, :]
        src_coordinate = src_coordinate.reshape((-1,3))
        src_on_dst = np.dot(H, src_coordinate.T).T
        
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        src_on_dst /= src_on_dst[:,2:3]
        x_max_mask, x_min_mask = src_on_dst[:,0:1]<w_dst-1, src_on_dst[:,0:1]>0
        y_max_mask, y_min_mask = src_on_dst[:,1:2]<h_dst-1, src_on_dst[:,1:2]>0
        mask = np.ones((src_on_dst.shape[0], 1), dtype=bool)
        mask = np.concatenate((x_max_mask*x_min_mask, y_max_mask*y_min_mask, mask), axis=1)
        mask = np.all(mask, axis=1)
        
        # TODO: 5.filter the valid coordinates using previous obtained mask
        src_on_dst, src_coordinate = src_on_dst[mask,...], src_coordinate[mask,...].astype(int)
        
        # TODO: 6. assign to destination image using advanced array indicing
        
        src_on_dst = np.around(src_on_dst).astype(int)
        pixel_values = src[src_coordinate[:,1], src_coordinate[:,0], :]
        sorted_idx = np.lexsort(src_on_dst.T)
        sorted_src_on_dst = src_on_dst[sorted_idx]
        
        unqID_mask = np.append(True, np.any(np.diff(sorted_src_on_dst,axis=0), axis=1))
        ID = unqID_mask.cumsum() - 1
        unq_src_on_dst = sorted_src_on_dst[unqID_mask]
        average_pixel_values0 = np.bincount(ID, pixel_values[sorted_idx, 0]) / np.bincount(ID)
        average_pixel_values1 = np.bincount(ID, pixel_values[sorted_idx, 1]) / np.bincount(ID)
        average_pixel_values2 = np.bincount(ID, pixel_values[sorted_idx, 2]) / np.bincount(ID)
        
        average_pixel_values = np.concatenate((average_pixel_values0.reshape((-1,1)),
                                               average_pixel_values1.reshape((-1,1)),
                                               average_pixel_values2.reshape((-1,1))),
                                               axis=1)
        dst[unq_src_on_dst[:,1], unq_src_on_dst[:,0], :] = average_pixel_values

    return dst
