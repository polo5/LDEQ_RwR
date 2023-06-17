"""code adapted from https://github.com/starhiking/HeatmapInHeatmap"""

import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import numpy as np
import sys
sys.path.append('.')
import os
from PIL import Image
import math

def flip_points(data_type="WFLW"):
    data_type = data_type.upper()
    points_flip = None
    if data_type == 'WFLW':
        points_flip = [32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,46,45,44,43,42,50,49,48,47,37,36,35,34,33,41,40,39,38,51,52,53,54,59,58,57,56,55,72,71,70,69,68,75,74,73,64,63,62,61,60,67,66,65,82,81,80,79,78,77,76,87,86,85,84,83,92,91,90,89,88,95,94,93,97,96]
        assert len(points_flip) == 98
    else:
        print('No such data!')
        exit(0)
    return points_flip

def pad_crop(image, target):
    """
        add pad for the overflow points
        image need change data type
        border_pad : 8px
    """
    image_height, image_width = image.size

    l, t = np.min(target, axis=0)
    r, b = np.max(target, axis=0)

    # if the over border is left than grid_size, pass
    grid_size = 0.5 / image_height

    if l > -grid_size and t > -grid_size and r < (1 + grid_size) and b < (1 + grid_size):
        target = np.maximum(target, 0)
        target = np.minimum(target, 1)
        return image, target
    border_pad_value = 8
    image_np = np.array(image).astype(np.uint8)
    border_size = np.zeros(4).astype('int')  # upper bottom left right
    if l < 0:
        border_size[2] = math.ceil(-l * image_height) + border_pad_value  # left
    if t < 0:
        border_size[0] = math.ceil(-t * image_width) + border_pad_value  # upper
    if r > 1:
        border_size[3] = math.ceil((r - 1) * image_height) + border_pad_value  # right
    if b > 1:
        border_size[1] = math.ceil((b - 1) * image_width) + border_pad_value  # bottom
    border_img = np.zeros((image_width + border_size[0] + border_size[1],
                           image_height + border_size[2] + border_size[3], 3)).astype(np.uint8)

    border_img[border_size[0]: border_size[0] + image_height,
    border_size[2]: border_size[2] + image_width, :] = image_np

    image_pil = Image.fromarray(border_img.astype('uint8'), 'RGB')
    image_pil = image_pil.resize((image_height, image_width))
    target = (target * np.array([image_height, image_width]) +
              np.array([border_size[2], border_size[0]])) / np.array([border_img.shape[1], border_img.shape[0]])

    return image_pil, target

def ignore_crop(target):
    target = np.maximum(target, 0)
    target = np.minimum(target, 1)
    return target

def check_size(imgs,im_size):
    """
        check dataset read success and size is same as input size.
        Args:
            imgs: [Image,...]
            config : config py
    """
    for i in range(len(imgs)):
        img = imgs[i]
        if img.height != im_size or img.width != im_size:
            print("{}th Image is not applicable ({},{}),need delete or resize.".format(i+1,img.height,img.width))
            exit(-1)

class FaceDataset(data.Dataset):
    def __init__(self, root_dir, split):
        assert split in ["test", "test_occlusion", "test_makeup", "test_largepose", "test_illumination", "test_expression", "test_blur"], "nvidia won't release training pipeline sorry..."
        self.data_path = os.path.join(root_dir, split)
        self.normalize = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        label_path = os.path.join(root_dir, split+".txt")
        with open(label_path,'r') as f:
            data_txt = f.readlines() #These landmarks are given in [0,1] range
        data_info = np.array([x.strip().split() for x in data_txt])

        self.img_paths = data_info[:,0].copy()
        self.pts_array = data_info[:,1:].astype(np.float32).reshape(data_info.shape[0],-1,2).copy()
        self.imgs = [Image.open(os.path.join(self.data_path,img_path)).convert('RGB') for img_path in self.img_paths]
        check_size(self.imgs, im_size=256)
        print("Finished loading WFLW dataset")

    def __getitem__(self,index):

        img = self.imgs[index].copy()
        kpts = self.pts_array[index].copy() #numpy array of landmarks in [0,1] scale

        # ignore or pad crop,both are ok.
        img, kpts = pad_crop(img,kpts)
        # target = ignore_crop(target)
        img = self.normalize(img)
        kpts = torch.from_numpy(kpts).float()

        data = {
            "image": img,
            "kpts": torch.clip(kpts, 0.0, 1.0), # return kpts in [0,1]
        }

        return data


    def __len__(self):
        return self.pts_array.shape[0]

if __name__=='__main__':
    from utils.plot_kpts import plot_kpts_grid
    from utils.helpers import set_torch_seeds
    set_torch_seeds(0)

    root_dir = '/home/paul/Datasets/Keypoints/WFLW/HIH'

    val_dataset = FaceDataset(root_dir=root_dir, split='test')
    val_loader = data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0, drop_last=False, pin_memory=False)
    print(len(val_dataset))

    for i, data in enumerate(val_loader):
        x, kpts = data['image'], data['kpts']
        print(x.shape, kpts.shape, torch.min(kpts), torch.max(kpts))
        break

    x = 255 * (x * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))  # unormalize for plotting

    plot_kpts_grid(kpts[..., 0].data.cpu().numpy(),
                   kpts[..., 1].data.cpu().numpy(),
                   torch.flip(x, dims=(1,)).data.cpu().numpy(),  # RGB to BGR
                   grid=[10, 10],
                   save_path='./WFLWv2_images.png',
                   confidence=None,
                   gt_xpred=kpts[..., 0].data.cpu().numpy(),
                   gt_ypred=kpts[..., 1].data.cpu().numpy())

