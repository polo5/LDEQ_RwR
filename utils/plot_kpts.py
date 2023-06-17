
import cv2
import numpy as np
from copy import deepcopy
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt


def plot_kpts_img(img, ikpts, confidence=None, gt_xpred=None, gt_ypred=None):

    circle_size = 1

    # ikpts = ikpts * img.shape[1]

    if 1:
        if img.size == img.shape[0] * img.shape[1]:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        up_size = 128.0 # img.shape[0]
        # pdb.set_trace()
        img = cv2.resize(img, (128, 128))

        for ind, kpt in enumerate(ikpts):
            if not any(np.isnan(kpt)):
                # color = colors[ind]
                # img = plot_cross(img, kpt=(int(up_size * kpt[0]), int(up_size * kpt[1])),
                #                  color=color, lnt=2)
                color_pred = (0,0,125)
                color_gt = (0,125,0)

                if gt_xpred is not None:
                    # pdb.set_trace()
                    cv2.circle(img, (int(up_size * gt_xpred[ind]), int(up_size * gt_ypred[ind])), 1, color=color_pred)



                if not (confidence is None):
                    max_confidence = 10

                    # pdb.set_trace()
                    current_confidence = max(1,int((1 - confidence[ind]) * max_confidence)+1)

                    if current_confidence > 0:
                        cv2.circle(img, (int(up_size * kpt[0]), int(up_size * kpt[1])), current_confidence, thickness=1, color=color_gt)

                else:
                    cv2.circle(img, (int(up_size * kpt[0]), int(up_size * kpt[1])), circle_size, thickness=-1, color=color_gt)

                # print(int(up_size* kpt[0]), int(up_size * kpt[1]))


    return img


def plot_kpts_grid(xpred, ypred, img, grid=[5, 5], save_path="./image.png", confidence=None, gt_xpred=None, gt_ypred=None):

    grid_size = int(min(grid[0], np.sqrt(img.shape[0])))
    grid = [grid_size, grid_size]

    im_save_ind = 0
    all_images = list()
    nrows = grid[0]
    ncols = grid[0]

    for img_i, kpts_x_predicted, kpts_y_predicted in zip(img, xpred, ypred):
        if im_save_ind > (nrows * ncols - 1):
            continue
        kpts_i_predicted = np.vstack((kpts_x_predicted, kpts_y_predicted)).T
        img_r = deepcopy(img_i.transpose((1, 2, 0)))

        confidence_loc = confidence[im_save_ind] if (confidence is not None) else None

        img_plot = plot_kpts_img(img_r, kpts_i_predicted, confidence=confidence_loc, gt_xpred=gt_xpred[im_save_ind], gt_ypred=gt_ypred[im_save_ind])
        all_images.append(img_plot)
        im_save_ind += 1

    all_images = np.asarray(all_images, dtype=np.uint8)
    height = img_plot.shape[0]
    width = img_plot.shape[1]

    nrows = int(nrows)
    ncols = int(ncols)
    height = int(height)
    width = int(width)

    all_images = all_images.reshape(nrows, ncols, height, width, 3).swapaxes(1, 2).reshape(height * nrows, width * ncols, 3)

    if save_path is not None: cv2.imwrite(save_path, all_images)
