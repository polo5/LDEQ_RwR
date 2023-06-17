import torch
import torch.nn as nn
import pdb
import torch


class Normalize(nn.Module):
    """normalize to [0,1]"""

    def __init__(self, n_channels, mode, beta=1.0, learn_beta=False):
        super().__init__()
        self.mode=mode
        assert mode in ['softargmax', 'linear'], f"norm {mode} not recognized"

        if mode=='softargmax':
            if learn_beta:
                self.nonlinearity = nn.Softplus()
                self.beta = torch.nn.Parameter(torch.ones(n_channels) * beta).view(1,-1,1,1).cuda() #one beta per heatmap. TODO: why isn't this put on gpu when we put whole model?
            else:
                self.nonlinearity = lambda x:x
                self.beta=beta

    def forward(self, heatmaps):

        if self.mode=='softargmax':
            heatmaps = heatmaps - torch.amax(heatmaps, dim=(2,3), keepdim=True)
            heatmaps = torch.exp(self.nonlinearity(self.beta)*heatmaps) #nonlinearity makes sure beta is positive so that exp input still in [-inf,0] so that output is in [0,1]

        elif self.mode=='linear':
            heatmaps_max, heatmaps_min = torch.amax(heatmaps, dim=(2, 3), keepdim=True), torch.amin(heatmaps, dim=(2, 3), keepdim=True)  # shape (B,n_kpts,1,1)
            heatmaps = (heatmaps - heatmaps_min) / (heatmaps_max - heatmaps_min +1e-5)

        return heatmaps


class HeatmapsToKeypoints(nn.Module):
    """converts 2D heatmaps into (x,y) coordinates in range [0,1] that our loss can use"""

    def __init__(self):
        super().__init__()
        self.first_run = True

    def forward(self, heatmaps):
        """ heatmap values must all be between 0 and 1. This is achieved with the Normalize class above """
        B, n_keypoints, H, W = heatmaps.shape
        heatmaps = heatmaps/(1e-4+torch.sum(heatmaps, dim=[2,3], keepdim=True)) #now heatmap values all sum to 1

        if self.first_run:
            col_vals = torch.arange(0, W)
            self.col_grid = col_vals.repeat(H, 1).view(1, 1, H, W).to(heatmaps.device) #each column is a single repeated number
            row_vals = torch.arange(0, H).view(H, -1)
            self.row_grid = row_vals.repeat(1, W).view(1, 1, H, W).float().to(heatmaps.device) #each row is a single repeated number
            self.first_run = False

        weighted_x = heatmaps * self.col_grid
        x_vals = weighted_x.sum(dim=[2,3])/H #in range [0,1], shape (B,98)
        weighted_y = heatmaps * self.row_grid
        y_vals = weighted_y.sum(dim=[2, 3])/H #in range [0,1], shape (B,98)
        out = torch.stack((x_vals, y_vals), dim=2)

        # TODO: not sure this is still correct if using linear normalization for heatmaps:
        var_x =  ((self.col_grid - x_vals.unsqueeze(2).unsqueeze(3)).pow(2)*heatmaps).sum(dim=[2, 3]) #this is like a variance term and can take on large values (~600 for heatmap size 64)
        var_y =  ((self.row_grid - y_vals.unsqueeze(2).unsqueeze(3)).pow(2)*heatmaps).sum(dim=[2, 3])
        #NB: if x_vals=mean=5, then (col_grid - x_vals) will be a grid, with 0 at location of the mean and polynomially increasing values around the mean
        # then heatmaps weighs this grid with the spread of predictions. If heatmaps is non zero only in location of mean, then sigma_x = 0

        stds = torch.sqrt(0.5*var_x+0.5*var_y)/H #shape (B,98), i.e. one std value per heatmap.

        return out, stds #out is (B, 98, 2)



class HeatmapsToKeypointsNoSum(nn.Module):
    """converts 2D heatmaps into (x,y) coordinates in range [0,1] that our loss can use,
    This can be used if input heatmaps are already divided by their sum
    """

    def __init__(self):
        super().__init__()
        self.first_run = True

    def forward(self, heatmaps):
        """ heatmap values must all be between 0 and 1. This is achieved with the Normalize class above """
        B, n_keypoints, H, W = heatmaps.shape
        # heatmaps = heatmaps/(1e-4+torch.sum(heatmaps, dim=[2,3], keepdim=True))

        if self.first_run:
            col_vals = torch.arange(0, W)
            self.col_grid = col_vals.repeat(H, 1).view(1, 1, H, W).to(heatmaps.device) #each column is a single repeated number
            row_vals = torch.arange(0, H).view(H, -1)
            self.row_grid = row_vals.repeat(1, W).view(1, 1, H, W).float().to(heatmaps.device) #each row is a single repeated number
            self.first_run = False

        weighted_x = heatmaps * self.col_grid
        x_vals = weighted_x.sum(dim=[2,3])/H #in range [0,1], shape (B,98)
        weighted_y = heatmaps * self.row_grid
        y_vals = weighted_y.sum(dim=[2, 3])/H #in range [0,1], shape (B,98)
        out = torch.stack((x_vals, y_vals), dim=2)

        # TODO: not sure this is still correct if using linear normalization for heatmaps:
        var_x =  ((self.col_grid - x_vals.unsqueeze(2).unsqueeze(3)).pow(2)*heatmaps).sum(dim=[2, 3]) #this is like a variance term and can take on large values (~600 for heatmap size 64)
        var_y =  ((self.row_grid - y_vals.unsqueeze(2).unsqueeze(3)).pow(2)*heatmaps).sum(dim=[2, 3])
        #NB: if x_vals=mean=5, then (col_grid - x_vals) will be a grid, with 0 at location of the mean and polynomially increasing values around the mean
        # then heatmaps weighs this grid with the spread of predictions. If heatmaps is non zero only in location of mean, then sigma_x = 0

        stds = torch.sqrt(0.5*var_x+0.5*var_y)/H #shape (B,98), i.e. one std value per heatmap.

        return out, stds




