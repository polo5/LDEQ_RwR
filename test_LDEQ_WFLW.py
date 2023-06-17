import time
import argparse
import numpy as np
import torchvision.transforms as transforms

from utils.helpers import *
from utils.loss_function import *
from utils.normalize import Normalize, HeatmapsToKeypoints
from datasets.WFLW_V.helpers import *
from models.ldeq import LDEQ, weights_init

heatmaps_to_keypoints = HeatmapsToKeypoints()


class DEQInference(object):

    def __init__(self, args):
        self.args = args

        ## Model
        ckpt = torch.load(args.landmark_model_weights, map_location='cpu')
        self.train_args = ckpt['args']
        self.train_args.stochastic_max_iters = False #use maximum iters at inference time so perf repeatable
        self.model = LDEQ(self.train_args).cuda()
        self.model.apply(weights_init)
        self.model.load_state_dict(ckpt['state_dict'], strict=False)
        self.model.eval()
        print(f'Restored weights for {self.train_args.landmark_model_name} from {self.args.landmark_model_weights}')

        ## Video stuff
        self.normalize = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def get_z0(self, batch_size):
        if self.train_args.z0_mode == 'zeros':
            return torch.zeros(batch_size, self.train_args.z_width, self.train_args.heatmap_size, self.train_args.heatmap_size, device='cuda')
        else:
            raise NotImplementedError

    def test_WFLW(self):
        """test code adapted from https://github.com/starhiking/HeatmapInHeatmap"""
        from datasets.WFLW.dataset import FaceDataset
        from torch.utils.data import DataLoader
        WFLW_splits = ["test", "test_largepose", "test_expression", "test_illumination",
                       "test_makeup", "test_occlusion", "test_blur"]
        self.model.eval()
        print(f'Running inference for splits {WFLW_splits}')

        for split in WFLW_splits:
            test_dataset = FaceDataset(root_dir=args.dataset_path, split=split)
            dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

            SME = 0.0
            IONs = None

            with torch.no_grad():
                for data in dataloader_test:
                    x, keypoints = data["image"].cuda(), data["kpts"].cuda()
                    output = self.model(x, mode=self.train_args.model_mode, args=self.train_args, z0=self.get_z0(x.shape[0]))
                    pred_keypoints = output['keypoints']

                    sum_ion, ion_list = calc_nme(pred_keypoints, keypoints)
                    SME += sum_ion
                    IONs = np.concatenate((IONs, ion_list), 0) if IONs is not None else ion_list

            nme, fr, auc = compute_fr_and_auc(IONs, thres=0.10, step=0.0001)

            print(f'\n------------ {split} ------------')
            print("NME %: {}".format(nme * 100))
            print("FR_{}% : {}".format(0.10, fr * 100))
            print("AUC_{}: {}".format(0.10, auc))


##########################


def main(args):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    t0 = time.time()
    solver = DEQInference(args)
    solver.test_WFLW()

    print(f'Total time: {format_time(time.time()-t0)}')
    print(f'Max mem: {torch.cuda.max_memory_allocated(device="cuda") / (1024 ** 3):.1f} GB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEQ Inference')
    parser.add_argument('--landmark_model_weights', default='/home/paul/Documents/RWR_publishing/WFLW/final.pth.tar')
    parser.add_argument('--dataset_path', type=str, default="/home/paul/Datasets/Keypoints/WFLW/HIH/")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)

    args = parser.parse_args()

    print('\nStarting...')

    main(args)
