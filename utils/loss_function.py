import torch
import numpy as np
from scipy.integrate import simps

def compute_fr_and_auc(nmes, thres=0.10, step=0.0001):
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    nme = np.mean(nmes)

    return nme, fr, auc

def calc_nme(landmarks_pred, landmarks_gt):
    ION_list = []  # inner occular norm
    norm_indices = [60, 72]  # WFLW

    # target_w_size and preds : n, 98 , 2
    target_np = landmarks_gt.cpu().numpy()
    pred_np = landmarks_pred.cpu().numpy()

    for target, pred in zip(target_np, pred_np):
        diff = target - pred
        norm = np.linalg.norm(target[norm_indices[0]] - target[norm_indices[1]])
        c_ION = np.sum(np.linalg.norm(diff, axis=1)) / (diff.shape[0] * norm)
        ION_list.append(c_ION)

    Sum_ION = np.sum(ION_list)  # the ion of this batch
    # need div the dataset size to get nme

    return Sum_ION, ION_list

def video_NME_NMJ(model_kpts, gt_kpts):
    """
    Compute NME and NMJ over all frames in video.
    model_kpts.shape = gt_kpts.shape = (n_frames, n_kpts, 2)
    """
    NME = video_NME(model_kpts, gt_kpts)
    NMJ = video_NMJ(model_kpts, gt_kpts)

    return NME, NMJ

def video_NME(model_kpts, gt_kpts):
    """
    Compute NME over all frames in video.
    model_kpts.shape = gt_kpts.shape = (n_frames, n_kpts, 2)
    """

    inter_occ_norms = np.linalg.norm(gt_kpts[:, 60, :] - gt_kpts[:, 72, :], axis=1, keepdims=True) # (n_frames, 1)
    NMEs = 100 * np.linalg.norm(gt_kpts - model_kpts, axis=2) / inter_occ_norms  # (n_frames, n_kpts)
    NME = np.mean(np.mean(NMEs, axis=1), axis=0)

    return NME

def video_NMJ(model_kpts, gt_kpts):
    """
    Compute NMJ over all frames in video.
    model_kpts.shape = gt_kpts.shape = (n_frames, n_kpts, 2)
    """

    x_scales = np.max(gt_kpts[:, :, 0], axis=1) - np.min(gt_kpts[:, :, 0], axis=1)
    y_scales = np.max(gt_kpts[:, :, 1], axis=1) - np.min(gt_kpts[:, :, 1], axis=1)
    faces_scales = np.sqrt(x_scales * y_scales / (256 ** 2)).reshape(-1, 1)  # normalize in units of standard areas

    delta = gt_kpts - model_kpts  # (n_frames, n_kpts, 2)
    NMJs = 100 * np.linalg.norm(delta[:-1] - delta[1:], axis=2) / faces_scales[1:]  # this is always the same, as one value per kpt. What changes is aggregation across all kpts and across all frames
    NMJ = rms(rms(NMJs, axis=1), axis=0)

    return NMJ

def rms(x, axis=0):
    return np.sqrt(np.mean(x**2, axis=axis))
