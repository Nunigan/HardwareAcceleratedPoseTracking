# @Author: Simon Walser
# @Date:   2021-04-13 16:53:24
# @Last Modified by:   Simon Walser
# @Last Modified time: 2023-04-01 16:06:55

import numpy as np
import cv2
import jax.numpy as jnp
import jax
from jax import lax
from functools import partial
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import tensorrt as trt

################################################################################
#
# Class definitions
#
################################################################################

class LimbParserPAF:
    def __init__(self, min_dist, max_dist, max_people, ngbh_size, paf_thresh):
        # Vars
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_people = max_people
        self.paf_thresh = paf_thresh
        self.min_num_kp = 3

        # Subpixel detection
        top = (ngbh_size - 1) // 2
        bottom = ngbh_size // 2
        left = (ngbh_size - 1)  // 2
        right = ngbh_size // 2
        u = np.arange(-left, right+1)
        v = np.arange(-top, bottom+1)
        square = np.stack(np.meshgrid(u, v), axis=-1)
        mask_circle = np.linalg.norm(square, axis=-1) < (ngbh_size / 2)
        self.idx_offset = jnp.asarray(square.reshape([-1,2])[np.ravel(mask_circle)])

        self.start_pt_idx = jnp.array([0,1,2,3,1,5,6,1,8, 9,10, 8,12,13])
        self.end_pt_idx   = jnp.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
        self.limb_kp_idx = jnp.array([[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],
                                      [1,8],[8,9],[9,10],[10,11],[8,12],[12,13],
                                      [13,14]])
        valid_idx = np.array([[0,1,4,7],
                              [0,1,2,4,7],
                              [1,2,3],
                              [2,3],
                              [0,1,4,5,7],
                              [4,5,6],
                              [5,6],
                              [0,1,4,7,8,11],
                              [7,8,9,11],
                              [8,9,10],
                              [9,10],
                              [7,8,11,12],
                              [11,12,13],
                              [12,13]], dtype=object)
        valid_mask_npy = np.zeros([14,14,1,1], dtype='bool')
        for i, col_idx in enumerate(valid_idx):
            valid_mask_npy[i,col_idx] = True
        self.valid_mask = jnp.array(valid_mask_npy)

        clean_idx = np.array([[0],[2,3,4],[3,4],[4],[5,6,7],[6,7],[7],
                              [8,9,10,11,12,13,14],[9,10,11],[10,11],[11],
                              [12,13,14],[13,14],[14]], dtype=object)
        clean_mask = np.zeros([14,15], dtype=np.bool8)
        for i, col_idx in enumerate(clean_idx):
            clean_mask[i,col_idx] = True
        self.clean_mask = jnp.repeat(clean_mask[jnp.newaxis], self.max_people, axis=0)

        self.num_limbs = int(len(self.start_pt_idx))
        self.num_body_parts = int(np.max(self.limb_kp_idx) + 1)

    def _subpixel_detection(self, conf_peaks, conf_maps):
        max_detections = self.max_people * self.num_body_parts

        # Rearrange array to prioritize horizontal direction. This is relevant
        # if there are more detections in conf_peaks than max_detections.
        arg = jnp.transpose(conf_peaks, [1,2,0])
        ret = jnp.argwhere(arg, size=max_detections, fill_value=2_147_483_647)
        pixel_pos = ret.at[:,jnp.array([2,0,1])].get()

        valid_mask = jnp.any(pixel_pos != 2_147_483_647, axis=-1, keepdims=True)

        u_idx = pixel_pos[:,1,jnp.newaxis] + self.idx_offset[:,0]
        v_idx = pixel_pos[:,0,jnp.newaxis] + self.idx_offset[:,1]
        weights = conf_maps.at[v_idx,u_idx,pixel_pos[:,-1:]].get(mode='fill', fill_value=0)
        tmp = jnp.sum(jnp.stack([v_idx, u_idx]) * weights, axis=-1) / (jnp.sum(weights, axis=-1) + 1e-16)
        subpix_pos = jnp.where(valid_mask, x=jnp.transpose(tmp), y=jnp.array([[jnp.nan, jnp.nan]]))

        return pixel_pos, subpix_pos

    def _drop_repetitions(self, pixel_pos):
        valid_kp = jnp.all(pixel_pos != 2_147_483_647, axis=-1)
        is_valid = jnp.logical_and(valid_kp[:,jnp.newaxis], valid_kp)

        dist = jnp.linalg.norm(pixel_pos[:,jnp.newaxis,:-1] - pixel_pos[jnp.newaxis,:,:-1], axis=-1)
        is_close = jnp.logical_and(jnp.triu(dist < self.min_dist, k=1), is_valid)
        idx0, idx1 = jnp.where(is_close, size=pixel_pos.shape[0], fill_value=2_147_483_647)

        is_same = (pixel_pos[idx0,-1] == pixel_pos[idx1,-1])
        idx = jnp.where(is_same, idx1, 2_147_483_647)
        pixel_pos = pixel_pos.at[idx].set(2_147_483_647)

        return pixel_pos

    def _get_paf_values(self, paf, pt_1, pt_2):
        paf = jnp.reshape(paf, paf.shape[:2]+(paf.shape[2]//2, 2))

        length = jnp.linalg.norm(pt_2 - pt_1, axis=-1)
        v_1 = (pt_2 - pt_1) / (length[...,jnp.newaxis] + 1e-16)
        v_2 = jnp.flip(v_1, axis=-1) * jnp.asarray([[-1,1]], dtype=jnp.float32)

        tmp = jnp.meshgrid(jnp.arange(paf.shape[0]), jnp.arange(paf.shape[1]), indexing='ij')
        img_idx = jnp.repeat(jnp.stack(tmp, axis=-1)[jnp.newaxis,:,:,:], self.num_limbs, axis=0)
        img_idx = jnp.float32(img_idx)

        img_idx -= pt_1[:,jnp.newaxis,jnp.newaxis,:]

        length_mat = jnp.sum(v_1[:,jnp.newaxis,jnp.newaxis,:]*img_idx, axis=-1)
        width_mat  = jnp.sum(v_2[:,jnp.newaxis,jnp.newaxis,:]*img_idx, axis=-1)

        mask_length = jnp.logical_and(length_mat < length[:,jnp.newaxis,jnp.newaxis], length_mat >= 0.0)
        mask_width  = jnp.logical_and(width_mat  < 0.5, width_mat  >= -0.5)
        mask_radius = jnp.hypot(length_mat, width_mat) < 0.71
        mask = jnp.logical_or(jnp.logical_and(mask_length, mask_width), mask_radius)

        paf_ = jnp.reshape(jnp.transpose(paf, [2,0,1,3]), [self.num_limbs,-1,2])
        out  = jnp.where(jnp.reshape(mask, [jnp.shape(mask)[0],-1,1]), x=paf_, y=jnp.array([[[0,0]]]))

        return out

    def _calc_integral(self, pxl_points, pafs):
        diff_vector = pxl_points[:,:,1,:] - pxl_points[:,:,0,:]
        diff_vector /= (jnp.linalg.norm(diff_vector, axis=-1, keepdims=True) + 1e-6)

        # Iterate over all limb candidates / people
        init_val = jnp.zeros(pxl_points.shape[:-2])
        def body_fun(i, val):
            paf_val = self._get_paf_values(pafs, pxl_points[:,i,0,:], pxl_points[:,i,1,:])
            scalars = jnp.sum(diff_vector[:,jnp.newaxis,i,:] * paf_val, axis=-1)
            scalar_mean = jnp.sum(scalars, axis=-1) / (jnp.count_nonzero(scalars, axis=-1) + 1e-16) ** 1.2
            val = val.at[:,i].set(scalar_mean)
            return val
        paf_weights = lax.fori_loop(0, self.max_people**2, body_fun, init_val)
        return paf_weights

    def _calc_weights(self, pxl_points, paf, resolution_ratio):
        start_pt = jnp.transpose(pxl_points.at[:,self.start_pt_idx].get(), [1,0,2])
        end_pt = jnp.transpose(pxl_points.at[:,self.end_pt_idx].get(), [1,0,2])

        paf_endpoints = jnp.concatenate([jnp.repeat(start_pt, self.max_people, axis=1)[:,:,jnp.newaxis,:],
                                         jnp.tile(end_pt, [1,self.max_people,1])[:,:,jnp.newaxis,:]], axis=-2)
        paf_endpoints *= resolution_ratio

        # Calculate integral
        paf_weights_ = self._calc_integral(paf_endpoints, paf)
        mask = jnp.all(paf_endpoints != 0, axis=(-2,-1))
        weights = jnp.reshape(jnp.where(mask, x=paf_weights_, y=-1), (-1,self.max_people,self.max_people))

        return start_pt, end_pt, weights

    def _hungarian_algorithm(self, start_pt, end_pt, weights):
        idx_enum = jnp.arange(self.num_limbs)
        init_val = (jnp.zeros([self.num_limbs,self.max_people,2,2]),
                    -jnp.ones([self.max_people,self.num_limbs]),
                    weights)
        def body_fun(i, val):
            enp, w_sort, w = val
            weights_flat = jnp.reshape(w, [self.num_limbs,-1])
            idx_r, idx_c = jnp.unravel_index(jnp.argmax(weights_flat, axis=-1), shape=w.shape[1:])
            weights_paired = w.at[idx_enum,idx_r, idx_c].get()
            start_p_paired = start_pt.at[idx_enum,idx_r].get()
            end_p_paired   = end_pt.at[idx_enum,idx_c].get()
            enp = enp.at[idx_enum,i,0].set(start_p_paired)
            enp = enp.at[idx_enum,i,1].set(end_p_paired)
            w_sort = w_sort.at[i,idx_enum].set(weights_paired)
            w = w.at[idx_enum,idx_r,:].set(-jnp.inf)
            w = w.at[idx_enum,:,idx_c].set(-jnp.inf)
            return (enp, w_sort, w)
        endpoints, paf_weights_sorted, _ = lax.fori_loop(0, self.max_people, body_fun, init_val)

        return endpoints, paf_weights_sorted

    def _interconnect_limbs(self, endpoints, paf_weights):
        mask_1_1 = jnp.all(jnp.isclose(endpoints[:,jnp.newaxis,:,jnp.newaxis,1,:], endpoints[jnp.newaxis,:,jnp.newaxis,:,0,:]), -1)
        mask_1_2 = jnp.any(endpoints[:,jnp.newaxis,:,jnp.newaxis,1,:] != 0., axis=-1)
        mask_2_1 = jnp.all(jnp.isclose(endpoints[:,jnp.newaxis,:,jnp.newaxis,0,:], endpoints[jnp.newaxis,:,jnp.newaxis,:,1,:]), -1)
        mask_2_2 = jnp.all(jnp.isclose(endpoints[:,jnp.newaxis,:,jnp.newaxis,0,:], endpoints[jnp.newaxis,:,jnp.newaxis,:,0,:]), -1)
        mask_2_3 = jnp.any(endpoints[:,jnp.newaxis,:,jnp.newaxis,0,:] != 0., axis=-1)
        mask = ((mask_1_1 * mask_1_2) + (mask_2_1 * mask_2_3) + (mask_2_2 * mask_2_3)) * self.valid_mask

        ij = jnp.reshape(jnp.transpose(jnp.mgrid[:self.num_limbs, :self.max_people], [1,2,0]), [-1,2])
        init_val = (jnp.zeros([self.max_people, self.num_body_parts, 2], dtype=jnp.float32),
                    -jnp.ones_like(paf_weights, dtype=jnp.float32),
                    jnp.full_like(endpoints, jnp.nan))
        def body_fun(idx, val):
            kp_t, wgt_t, inst = val
            i = ij[idx,0]
            j = ij[idx,1]

            # Add limb and all of its neighbors
            neighbors = jnp.argwhere(mask[i,:,j], size=6, fill_value=2_147_483_647)
            neighbor_limb = neighbors.at[:,0].get()
            neighbor_pers = neighbors.at[:,1].get()
            neighbor_vals = endpoints.at[neighbor_limb,neighbor_pers].get(mode='fill', fill_value=0)
            neighbor_weights = paf_weights.at[neighbor_pers,neighbor_limb].get(mode='fill', fill_value=-1.)

            # Check if limb has already been added to instance
            entry_mask = jnp.all(inst[i] == endpoints[i,j], axis=(-2,-1))
            idx_pers = jnp.argwhere(entry_mask, size=1)

            # Check if a neighbor (of the current limb or its neighbors)
            # is already assigned to a person. Otherwise assign limb to a
            # person which has no corresponding limb assigned.
            not_occupied_pointwise = jnp.isnan(inst.at[neighbor_limb].get(mode='fill', fill_value=jnp.nan))
            not_occupied_personwise = jnp.all(not_occupied_pointwise, axis=(0,-2,-1))
            next_pers_idx = jnp.min(jnp.argwhere(not_occupied_personwise, size=self.max_people, fill_value=2_147_483_647))
            idx_pers = jnp.where(jnp.all(entry_mask==False), x=next_pers_idx, y=idx_pers)

            # Add limb and all of its neighbors
            inst = inst.at[neighbor_limb,idx_pers].set(neighbor_vals)
            wgt_t = wgt_t.at[idx_pers,neighbor_limb].set(neighbor_weights)
            kp_val = jnp.reshape(neighbor_vals, [-1,2])
            kp_idx = jnp.ravel(self.limb_kp_idx.at[neighbor_limb].get(mode='fill', fill_value=2_147_483_647))
            kp_t = kp_t.at[idx_pers,kp_idx].set(kp_val)
            return (kp_t, wgt_t, inst)

        kp_tensor, weight_tensor, instances = lax.fori_loop(0, ij.shape[0], body_fun, init_val)

        return kp_tensor, weight_tensor, instances

    def _prune_limbs(self, pixel_pos, paf_weights_sorted):
        # Skeleton pruning based on PAF threshold.
        # First, determine whether torso-part or limb-part of skeleton should be
        # kept if certain connection weight is below threshold
        kp_nonzero = jnp.any(pixel_pos, axis=-1)
        count_limb  = jnp.count_nonzero(jnp.logical_and(self.clean_mask, kp_nonzero[:,jnp.newaxis]), axis=-1)
        count_torso = jnp.count_nonzero(kp_nonzero, axis=-1, keepdims=True) - count_limb
        clean_mask_mask = (count_torso > count_limb)[...,jnp.newaxis]

        # Second, set keypoint (and all connected keypoints) to zero, whose
        # connection weight is below PAF threshold.
        clean_mask = jnp.where(clean_mask_mask, x=self.clean_mask, y=jnp.logical_not(self.clean_mask))
        kp_mask = jnp.any(jnp.logical_and((paf_weights_sorted < self.paf_thresh)[...,jnp.newaxis], clean_mask), axis=1)
        pixel_pos = jnp.where(kp_mask[...,jnp.newaxis], x=jnp.array([[[0.,0.]]]), y=pixel_pos)

        # Third, set all persons to zero who two or less keypoints
        num_kp_mask = jnp.sum(jnp.any(pixel_pos, axis=-1), axis=-1) < self.min_num_kp
        pixel_pos = jnp.where(num_kp_mask[...,jnp.newaxis,jnp.newaxis], x=0, y=pixel_pos)

        return pixel_pos

    def _sort_skeletons(self, pixel_pos, prev_centroid):
        median_arg = jnp.where(pixel_pos.at[:,:,1].get(), x=pixel_pos.at[:,:,1].get(), y=jnp.nan)
        centroid = jnp.nanmedian(median_arg, axis=-1, keepdims=True)
        dist = jnp.abs(centroid - prev_centroid)
        dist = jnp.where(jnp.isnan(dist), x=self.max_dist, y=dist)
        idx_sort = jnp.arange(self.max_people)

        init_val = (jnp.arange(self.max_people), dist)
        def body_fun(i, val):
            idx, d = val
            dist_flat = jnp.ravel(d)
            idx_r, idx_c = jnp.unravel_index(jnp.argmin(dist_flat, axis=0), shape=dist.shape)
            d = d.at[idx_r,:].set(jnp.inf)
            d = d.at[:,idx_c].set(jnp.inf)
            idx = idx.at[idx_c].set(idx_r)
            return (idx, d)
        idx_sort, _ = lax.fori_loop(0, self.max_people, body_fun, init_val)

        # Justify pixel_pos, i.e. left align non-zero values
        nonzero_mask = jnp.isfinite(centroid.at[idx_sort,0].get())
        nonzero_idx = jnp.squeeze(jnp.argwhere(nonzero_mask, size=self.max_people, fill_value=2_147_483_647), axis=1)
        idx_align = idx_sort.at[nonzero_idx].get(mode='fill', fill_value=2_147_483_647)
        ret_tensor_sort = pixel_pos.at[idx_align].get(mode='fill', fill_value=0)
        prev_centroid = centroid.at[idx_align,0].get(mode='fill', fill_value=jnp.nan)

        return ret_tensor_sort, prev_centroid

    def _extract_weights(self, pixel_pos, conf_maps):
        idx0 = jnp.int32(jnp.round(pixel_pos[...,0]))
        idx1 = jnp.int32(jnp.round(pixel_pos[...,1]))
        idx2 = jnp.repeat(jnp.arange(conf_maps.shape[-1])[jnp.newaxis], self.max_people, axis=0)
        tmp = conf_maps.at[idx0,idx1,idx2].get()
        weights = jnp.where(jnp.any(pixel_pos, axis=-1), x=tmp, y=0.)
        return weights

    @partial(jax.jit, static_argnums=(0,))
    def parse_keypoints(self, conf_peaks, conf_maps, paf_fields):
        # Subpixel resolution
        pixel_pos, subpix_pos = self._subpixel_detection(conf_peaks, conf_maps)

        # Remove points of same kind which are too close
        pixel_pos = self._drop_repetitions(pixel_pos)

        # Rearrange extracted keypoint indices
        init_val = (jnp.zeros([self.max_people, self.num_body_parts, 2], dtype=jnp.float32),
                    jnp.zeros(self.num_body_parts, dtype=jnp.int32))
        def body_fun(i, val):
            kp, count = val
            kp = kp.at[count[pixel_pos[i,-1]],pixel_pos[i,-1]].set(subpix_pos[i])
            count = count.at[pixel_pos[i,-1]].set(count[pixel_pos[i,-1]] + 1)
            return (kp, count)
        kp_proposals, _ = lax.fori_loop(0, self.max_people * self.num_body_parts, body_fun, init_val)

        ratio = (paf_fields.shape[0] / conf_maps.shape[0])
        start_pt, end_pt, paf_weights = self._calc_weights(kp_proposals, paf_fields, ratio)

        # Perform Hungarian algorithm
        endpoints, paf_weights = self._hungarian_algorithm(start_pt, end_pt, paf_weights)

        # Interconnect limbs
        ret_tensor, paf_weights, endpoints = self._interconnect_limbs(endpoints, paf_weights)

        # Prune limbs according to paf weight threshold
        ret_tensor = self._prune_limbs(ret_tensor, paf_weights)

        # Extract weights
        weights = self._extract_weights(ret_tensor, conf_maps)

        return ret_tensor, weights

class LimbParserNN:
    def __init__(self, min_dist, max_dist, max_people, ngbh_size):
        # Vars
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.max_people = max_people
        self.prev_centroid = jnp.full(self.max_people, jnp.nan)

        # Subpixel detection
        top = (ngbh_size - 1) // 2
        bottom = ngbh_size // 2
        left = (ngbh_size - 1)  // 2
        right = ngbh_size // 2
        u = np.arange(-left, right+1)
        v = np.arange(-top, bottom+1)
        square = np.stack(np.meshgrid(u, v), axis=-1)
        mask_circle = np.linalg.norm(square, axis=-1) < (ngbh_size / 2)
        self.idx_offset = jnp.asarray(square.reshape([-1,2])[np.ravel(mask_circle)])

    def _subpixel_detection(self, conf_peaks, conf_maps):
        max_detections = self.max_people * conf_peaks.shape[-1]

        # Rearrange array to prioritize horizontal direction. This is relevant
        # if there are more detections in conf_peaks than max_detections.
        arg = jnp.transpose(conf_peaks, [1,2,0])
        ret = jnp.argwhere(arg, size=max_detections, fill_value=2_147_483_647)
        pixel_pos = ret.at[:,jnp.array([2,0,1])].get()

        valid_mask = jnp.any(pixel_pos != 2_147_483_647, axis=-1, keepdims=True)

        u_idx = pixel_pos[:,1,jnp.newaxis] + self.idx_offset[:,0]
        v_idx = pixel_pos[:,0,jnp.newaxis] + self.idx_offset[:,1]
        weights = conf_maps.at[v_idx,u_idx,pixel_pos[:,-1:]].get(mode='fill', fill_value=0)
        tmp = jnp.sum(jnp.stack([v_idx, u_idx]) * weights, axis=-1) / (jnp.sum(weights, axis=-1) + 1e-16)
        subpix_pos = jnp.where(valid_mask, x=jnp.transpose(tmp), y=jnp.array([[jnp.nan, jnp.nan]]))

        return pixel_pos, subpix_pos

    def _drop_repetitions(self, pixel_pos):
        valid_kp = jnp.all(pixel_pos != 2_147_483_647, axis=-1)
        is_valid = jnp.logical_and(valid_kp[:,jnp.newaxis], valid_kp)

        dist = jnp.linalg.norm(pixel_pos[:,jnp.newaxis,:-1] - pixel_pos[jnp.newaxis,:,:-1], axis=-1)
        is_close = jnp.logical_and(jnp.triu(dist < self.min_dist, k=1), is_valid)
        idx0, idx1 = jnp.where(is_close, size=pixel_pos.shape[0], fill_value=2_147_483_647)

        is_same = (pixel_pos[idx0,-1] == pixel_pos[idx1,-1])
        idx = jnp.where(is_same, idx1, 2_147_483_647)
        pixel_pos = pixel_pos.at[idx].set(2_147_483_647)

        return pixel_pos

    def _calc_centroid(self, ref_position):
        # Calculate centroid
        rows_masked = jnp.where(jnp.any(ref_position, axis=-1), x=jnp.transpose(ref_position), y=jnp.nan)
        cols_masked = jnp.where(jnp.any(rows_masked==0, axis=-1), x=jnp.nan, y=jnp.transpose(rows_masked))
        arg_median = jnp.sort(cols_masked, axis=0)
        centroid = jnp.nanmedian(arg_median, axis=1, keepdims=True)
        return centroid

    def _sort_keypoints(self, ref_position, kp_proposals, centroid):
        # Determine distance matrix
        ref_position = jnp.where(ref_position == 0, x=jnp.nan, y=ref_position)
        dist = jnp.abs(ref_position[:,jnp.newaxis] - centroid[jnp.newaxis])
        dist = jnp.where(jnp.isnan(dist), x=self.max_dist, y=dist)

        # Perform assignment according to nearest neighbor principle
        idx_enum = jnp.arange(dist.shape[-1])
        init_val = (jnp.zeros_like(kp_proposals), dist)
        def body_fun(i, val):
            r, d = val
            dist_flat = jnp.reshape(d, [-1,d.shape[-1]])
            idx_r, idx_c = jnp.unravel_index(jnp.argmin(dist_flat, axis=0), shape=dist.shape[:-1])
            d = d.at[idx_r,:,idx_enum].set(jnp.inf)
            d = d.at[:,idx_c,idx_enum].set(jnp.inf)
            r = r.at[idx_c,idx_enum].set(kp_proposals[idx_r,idx_enum])
            return (r, d)
        ret_tensor, _ = lax.fori_loop(0, self.max_people, body_fun, init_val)

        return ret_tensor

    def _extract_weights(self, pixel_pos, conf_maps):
        idx0 = jnp.int32(jnp.round(pixel_pos[...,0]))
        idx1 = jnp.int32(jnp.round(pixel_pos[...,1]))
        idx2 = jnp.repeat(jnp.arange(conf_maps.shape[-1])[jnp.newaxis], self.max_people, axis=0)
        tmp = conf_maps.at[idx0,idx1,idx2].get()
        weights = jnp.where(jnp.any(pixel_pos, axis=-1), x=tmp, y=0.)
        return weights

    @partial(jax.jit, static_argnums=(0,))
    def parse_keypoints(self, conf_peaks, conf_maps, *args):
        # Extract keypoint indices
        num_body_parts = conf_peaks.shape[-1]
        max_detections = self.max_people * num_body_parts

        # Subpixel resolution
        pixel_pos, subpix_pos = self._subpixel_detection(conf_peaks, conf_maps)

        # Remove points of same kind which are too close
        pixel_pos = self._drop_repetitions(pixel_pos)

        # Rearrange extracted keypoint indices
        init_val = (jnp.zeros([self.max_people, num_body_parts, 2], dtype=jnp.float32),
                    jnp.zeros(num_body_parts, dtype=jnp.int32))
        def body_fun(i, val):
            kp, count = val
            kp = kp.at[count[pixel_pos[i,-1]],pixel_pos[i,-1]].set(subpix_pos[i])
            count = count.at[pixel_pos[i,-1]].set(count[pixel_pos[i,-1]] + 1)
            return (kp, count)
        kp_proposals, _ = lax.fori_loop(0, max_detections, body_fun, init_val)

        # Assign keypoints to nearest centroid in horizontal direction
        ref_position = kp_proposals.at[...,1].get()
        centroid = self._calc_centroid(ref_position)
        ret_tensor = self._sort_keypoints(ref_position, kp_proposals, centroid)

        # Extract weights
        weights = self._extract_weights(ret_tensor, conf_maps)

        return ret_tensor, weights

class UpperBoundCalculator(object):
    def __init__(self, max_people, min_dist):
        self.max_people = max_people
        self.min_dist = min_dist

    @partial(jax.jit, static_argnums=(0,))
    def get_supremum(self, conf_peaks):
        # Get pixel index of local maxima
        num_body_parts = conf_peaks.shape[-1]
        max_detections = self.max_people * num_body_parts
        pixel_pos = jnp.argwhere(conf_peaks, size=max_detections, fill_value=2_147_483_647)

        # Drop detections which are too close to neighboor and of same kind
        dist = jnp.linalg.norm(pixel_pos[:,jnp.newaxis,:-1] - pixel_pos[jnp.newaxis,:,:-1], axis=-1)
        is_close = jnp.triu(dist < self.min_dist, k=1)
        idx0, idx1 = jnp.where(is_close, size=pixel_pos.shape[0], fill_value=2_147_483_647)

        is_same = (pixel_pos[idx0,-1] == pixel_pos[idx1,-1])
        idx = jnp.where(is_same, idx1, 2_147_483_647)
        pixel_pos = pixel_pos.at[idx].set(2_147_483_647)

        # Find out upper bound on number of people in image
        unique, unique_counts = jnp.unique(pixel_pos.at[:,2].get(), size=num_body_parts+1, fill_value=2_147_483_647, return_counts=True)
        supremum = jnp.max(unique_counts, initial=0, where=unique != 2_147_483_647)

        return supremum

class TRTModel(object):
    """ Model used for inference with optimized TensorRT engine.
    """
    def __init__(self, file_name):
        cuda.init()
        self.cfx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()

        logger = trt.Logger()
        trt.init_libnvinfer_plugins(logger, '')
        runtime = trt.Runtime(logger)

        # Deserialize engine
        with open(file_name, 'rb') as f:
            buf = f.read()
            self.engine = runtime.deserialize_cuda_engine(buf)
        self.context = self.engine.create_execution_context()

        self.bindings = []
        self.host_inputs = []
        self.cuda_inputs = []
        self.input_shapes = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.output_shapes = []
        for binding in self.engine:
            tensor_shape = self.context.get_tensor_shape(binding)
            size = trt.volume(tensor_shape)
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(cuda_mem))
            if self.engine.get_tensor_mode(binding).name == 'INPUT':
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
                self.input_shapes.append(tensor_shape)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
                self.output_shapes.append(tensor_shape)

    def predict(self, x):
        self.cfx.push()

        x = x.astype(np.float32)
        np.copyto(self.host_inputs[0], x.ravel())

        # Inference
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        cuda.memcpy_dtoh_async(self.host_outputs[1], self.cuda_outputs[1], self.stream)
        cuda.memcpy_dtoh_async(self.host_outputs[2], self.cuda_outputs[2], self.stream)
        self.stream.synchronize()

        # Reshape output
        outputs = [self.host_outputs[0].reshape(self.output_shapes[0]),
                   self.host_outputs[1].reshape(self.output_shapes[1]),
                   self.host_outputs[2].reshape(self.output_shapes[2])]

        self.cfx.pop()

        return outputs

    def get_shape(self, idx=-1):
        if idx < 0:
            ret_val = self.input_shapes
        else:
            ret_val = self.input_shapes[idx]
        return ret_val

    def open(self):
        pass

    def close(self):
        self.cfx.pop()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, tb):
        self.close()

class MultiViewPoseEstimation(TRTModel):
    def __init__(self, path_model, paf_thresh=0.05, min_dist=12, ngbh_size=31, max_people=4):
        super().__init__(path_model)

        self.path_model = path_model
        self.paf_thresh = paf_thresh
        self.min_dist = min_dist
        self.ngbh_size = ngbh_size
        self.max_dist = np.max(self.input_shapes[0][1:-1])

        dummy_paf = jnp.zeros(self.output_shapes[0])
        dummy_conf = jnp.zeros(self.output_shapes[1])

        # Prepare objects
        limb_parser = LimbParserPAF(self.min_dist, self.max_dist, max_people, self.ngbh_size, self.paf_thresh)
        # limb_parser = LimbParserNN(self.min_dist, self.max_dist, max_people, self.ngbh_size)
        self.fun = jax.jit(jax.vmap(limb_parser.parse_keypoints), backend='gpu')
        self.fun(dummy_conf, dummy_conf, dummy_paf)

    def predict(self, frames):
        # Run neural network
        paf_fields, conf_maps, conf_peaks = super().predict(frames)
        kp_proposals_jax, kp_confidence_jax = self.fun(conf_peaks, conf_maps, paf_fields)

        # Parse keypoints
        kp_proposals_npy = np.asarray(kp_proposals_jax)
        kp_confidence_npy = np.asarray(kp_confidence_jax)

        mask = np.any(kp_proposals_npy, axis=(0,-2,-1))
        return kp_proposals_npy[:,mask], kp_confidence_npy[:,mask]

################################################################################
#
# Main functions
#
################################################################################

if __name__ == "__main__":

    img_res = (256,256)
    exposure = 40
    gain_raw = 370
    framerate = 20.

    # cam_process = MultiBaslerEmulation(img_res, exposure, gain_raw, framerate)

    batch_size = 4
    trt_engine_path = '/models/trt/ICAIPose_5_test/model.engine'
    model = MultiViewPoseEstimation(trt_engine_path)
    shape = model.get_shape(0)

    img = cv2.imread('data/multi.jpg')
    cv2.imshow('image', img)
    cv2.waitKey(1)
    result = model.predict(img)

    # for _ in range(500):
    #     data = cam_process.get_img(1)
    #     cv2.imshow('image', data[1])
    #     cv2.waitKey(1)
    #     result = model.predict(data)

    model.close()

