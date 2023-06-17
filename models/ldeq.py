import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

from utils.solvers import root_solver
from utils.normalize import *

def make_cell(args):
    return eval(args.cell_name)(args)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal(m.weight.data)
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.GroupNorm):
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

####################################################################################

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, norm='BN', GN_groups=1, no_relu=False):
        super(Conv, self).__init__()
        assert norm in ['BN', 'GN', 'None'], f"norm given {norm} unrecognized"
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = (lambda x:x) if no_relu else nn.LeakyReLU()
        self.norm = (lambda x:x) if norm=='None' else (nn.BatchNorm2d(out_dim) if norm=='BN' else nn.GroupNorm(GN_groups, out_dim))

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.norm(out)
        return out


class Hourglass(nn.Module):
    """
    Write out explicitly since nested formulation is too messy for multi-resolution.
    Same as default Hourglass for hg_dpeth=4 and n<10 and double_output=True
    Downside is that this now only supports hg_depth=4

    It was verified to have the same number of parameters (228572, if base_width=16 and increment=8)
    """

    def __init__(self, base_width, width_increment, norm='GN', GN_groups=8):
        super().__init__()
        self.downres = nn.AvgPool2d(2, 2)
        self.upres = nn.Upsample(scale_factor=2)
        w, i = base_width, width_increment
        w1, w2, w3, w4 = w + i, w + 2 * i, w + 3 * i, w + 4 * i

        self.same1 = Conv(w, w, 3, norm=norm, GN_groups=GN_groups)
        self.upchan1 = Conv(w, w1, 3, norm=norm, GN_groups=GN_groups)
        self.same2 = Conv(w1, w1, 3, norm=norm, GN_groups=GN_groups)
        self.upchan2 = Conv(w1, w2, 3, norm=norm, GN_groups=GN_groups)
        self.same3 = Conv(w2, w2, 3, norm=norm, GN_groups=GN_groups)
        self.upchan3 = Conv(w2, w3, 3, norm=norm, GN_groups=GN_groups)
        self.same4 = Conv(w3, w3, 3, norm=norm, GN_groups=GN_groups)
        self.upchan4 = Conv(w3, w4, 3, norm=norm, GN_groups=GN_groups)

        self.bottlneck = Conv(w4, w4, 3, norm=norm, GN_groups=GN_groups)

        self.downchan1 = Conv(w4, w3, 3, norm=norm, GN_groups=GN_groups)
        self.downchan2 = Conv(2 * w3, w3, 1, norm=norm, GN_groups=GN_groups)
        self.same5 = Conv(w3, w3, 5, norm=norm, GN_groups=GN_groups)
        self.downchan3 = Conv(w3, w2, 3, norm=norm, GN_groups=GN_groups)
        self.downchan4 = Conv(2 * w2, w2, 1, norm=norm, GN_groups=GN_groups)
        self.same6 = Conv(w2, w2, 5, norm=norm, GN_groups=GN_groups)
        self.downchan5 = Conv(w2, w1, 3, norm=norm, GN_groups=GN_groups)
        self.downchan6 = Conv(2 * w1, w1, 1, norm=norm, GN_groups=GN_groups)
        self.same7 = Conv(w1, w1, 5, norm=norm, GN_groups=GN_groups)
        self.downchan7 = Conv(w1, w, 3, norm=norm, GN_groups=GN_groups)
        self.downchan8 = Conv(2 * w, w, 1, norm=norm, GN_groups=GN_groups)
        self.same8 = Conv(w, w, 5, norm=norm, GN_groups=GN_groups)

    def forward(self, x):
        same1 = self.same1(x)
        downres1 = self.downres(same1)
        upchan1 = self.upchan1(downres1)

        same2 = self.same2(upchan1)
        downres2 = self.downres(same2)
        upchan2 = self.upchan2(downres2)

        same3 = self.same3(upchan2)
        downres3 = self.downres(same3)
        upchan3 = self.upchan3(downres3)

        same4 = self.same4(upchan3)
        downres4 = self.downres(same4)
        upchan4 = self.upchan4(downres4)

        # -----------------------------
        bottleneck = self.bottlneck(upchan4)
        # -----------------------------

        downchan1 = self.downchan1(bottleneck)
        upres1 = self.upres(downchan1)
        stack = torch.cat((same4, upres1), 1)
        downchan2 = self.downchan2(stack)
        same5 = self.same5(downchan2) + downchan2

        downchan3 = self.downchan3(same5)
        upres2 = self.upres(downchan3)
        stack = torch.cat((same3, upres2), 1)
        downchan4 = self.downchan4(stack)
        same6 = self.same6(downchan4) + downchan4

        downchan5 = self.downchan5(same6)
        upres3 = self.upres(downchan5)
        stack = torch.cat((same2, upres3), 1)
        downchan6 = self.downchan6(stack)
        same7 = self.same7(downchan6) + downchan6

        downchan7 = self.downchan7(same7)
        upres4 = self.upres(downchan7)
        stack = torch.cat((same1, upres4), 1)
        downchan8 = self.downchan8(stack)
        same8 = self.same8(downchan8) + downchan8

        return same8

####################################################################################

class Cell0(nn.Module):
    """same as Cell0 but always outputs data in [0,1]. We try various normalization techniques"""
    def __init__(self, args):
        super().__init__()
        norm_layer = 'BN' if args.cell_use_bn_for_explicit and args.model_mode=='explicit' else 'GN'
        self.tail = Conv(args.z_width+args.injection_width, args.cell_base_width, 1, norm=norm_layer, GN_groups=args.cell_gn_groups)
        self.hourglass = Hourglass(args.cell_base_width, args.cell_width_increment, norm=norm_layer, GN_groups=args.cell_gn_groups)
        self.head = Conv(args.cell_base_width, args.z_width, 1, stride=1, norm='None', no_relu=True)
        self.features_to_heatmaps = (lambda x:x) if args.cell_norm=='None' else Normalize(args.z_width, mode=args.cell_norm, beta=args.cell_softargmax_beta, learn_beta=args.cell_learn_softargmax_beta)

    def forward(self, z, injection):
        # print(z.shape, injection.shape)
        out = self.tail(torch.cat([z, injection],dim=1))
        out = self.hourglass(out)
        out = self.head(out) #heatmap size
        out = self.features_to_heatmaps(out) #heatmap = normalized features

        return out

####################################################################################

class DEQLayer(nn.Module):
    """
    A DEQ layer applies the same cell with weight-sharing for several iterations.
    It can do so explicitly (track operations in autograd) or implicitly (only track very last iteration)
    """

    def __init__(self, cell, args):
        super().__init__()
        self.cell, self.heatmap_size, self.z_width = cell, args.heatmap_size, args.z_width

    def _forward_explicit(self, x, args, z0, save_trajectory=False):
        fwd_logs = None
        trajectory = [z0] if save_trajectory else []
        out = z0
        depth = 2 if args is None else args.explicit_depth # torchinfo debug
        for _ in range(depth):
            out = self.cell(out, injection=x)
            if save_trajectory: trajectory.append(out.detach())
        # no need to do one more tracked forward pass here because they're all tracked already
            
        return out, fwd_logs, trajectory

    def _forward_implicit(self, x, args, z0, save_trajectory=False):
        trajectory = []
        z_shape = (x.shape[0], self.z_width, self.heatmap_size, self.heatmap_size) #agnostic to x dimensions
        z_shape_solver = (x.shape[0], self.z_width*self.heatmap_size*self.heatmap_size, 1) #agnostic to x dimensions
        func = lambda z: self.cell(z.view(z_shape), injection=x).view(z_shape_solver) #inputs/outputs vector of shape z_shape_solver

        stochastic_max_iters = args.stochastic_max_iters if self.training else False
        max_iters = max(1, round(args.max_iters/2)) if (not self.training and args.stochastic_max_iters) else args.max_iters

        with torch.no_grad():
            z_star, fwd_logs = root_solver(f=func, x0=z0, max_iters=max_iters, solver_args=args, stochastic_max_iters=stochastic_max_iters, save_trajectory=save_trajectory, name="forward")

        if self.training:
            z_star_new = func(z_star.requires_grad_()) #extra tracked step so we create a computational graph

            if args.solver=='fpi':
                with torch.no_grad():
                    fwd_logs['final_solver_error'] = float(torch.norm(z_star_new - z_star)/(torch.norm(z_star_new)+1e-9)) #same as the one in solver_logs if using tracing. But fpi doesn't do tracing.

            if not args.JFB:
                def backward_hook(grad):
                    if self.hook is not None:
                        self.hook.remove()
                        torch.cuda.synchronize()
                    func = lambda y: torch.autograd.grad(z_star_new, z_star, y, retain_graph=True)[0] + grad
                    solution, solver_logs_bwd = root_solver(f=func, x0=torch.zeros_like(grad), max_iters=max(1, round(args.max_iters/2)) if args.stochastic_max_iters else args.max_iters, solver_args=args, stochastic_max_iters=False, save_trajectory=False, name="backward")
                    # solution, solver_logs_bwd = root_solver(f=func, x0=torch.rand_like(grad), solver_args=args, stochastic_max_iters=False, save_trajectory=False, name="backward") #not good
                    if args.verbose_solver:
                        print(f'original grad: scale = {torch.mean(torch.abs(grad)):01.1e}, pos sign frac = {100*torch.mean((torch.sign(grad)+1)/2):02.0f}%')
                        print(f'  new    grad: scale = {torch.mean(torch.abs(solution)):01.1e}, pos sign frac = {100*torch.mean((torch.sign(solution)+1)/2):02.0f}%')
                        print(f' ---   change: value = {100*torch.mean(torch.abs((solution-grad)))/torch.mean(torch.abs(grad)):02.0f}%, sign: {100*torch.mean((torch.sign(grad)-torch.sign(solution))/2):02.0f}%')
                    return solution
                self.hook = z_star_new.register_hook(backward_hook) #WARNING: leads to memory leak if not cleared with .backward() at each batch

        else:
            if args.take_one_less_inference_step:
                z_star_new = z_star
            else:
                with torch.no_grad():
                    z_star_new = func(z_star)  # usually don't need to take this step at inference if close enough to solution already
                    if args.solver=='fpi': fwd_logs['final_solver_error'] = float(torch.norm(z_star_new - z_star)/(torch.norm(z_star_new)+1e-9))
                    # Note that when this extra step is performed we are actually taking max_iters+1 iterations

        if save_trajectory: #change shape and add z0
            trajectory = [z.view(z_shape) for z in fwd_logs['trajectory']]
            trajectory.insert(0, z0.view(z_shape))
            del fwd_logs['trajectory']

        return z_star_new.view(z_shape), fwd_logs, trajectory

    def forward(self, x, mode, args, z0=None, save_trajectory=False):
        z_shape = (x.shape[0], self.z_width, self.heatmap_size, self.heatmap_size)
        z_shape_solver = (x.shape[0], self.z_width*self.heatmap_size*self.heatmap_size, 1)

        if mode=='explicit':
            z0 = z0.view(*z_shape)
            out, fwd_logs, trajectory = self._forward_explicit(x, args, z0, save_trajectory)
        elif mode=='implicit':
            z0 = z0.view(*z_shape_solver)
            out, fwd_logs, trajectory = self._forward_implicit(x, args, z0, save_trajectory)
        else:
            raise NotImplementedError


        z_star_copy = out.detach().view(*z_shape)
        return out, z_star_copy, fwd_logs, trajectory

####################################################################################

class LDEQ(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        out_width = args.injection_width

        self.tail = nn.Sequential(
            Conv(3, out_width // 4, 7, 2),
            Conv(out_width // 4, out_width // 2, 3, 1),
            nn.MaxPool2d(2, 2),
            Conv(out_width // 2, out_width, 3, 1))

        cell = make_cell(args)
        self.deq_layer = DEQLayer(cell, args) #outputs potentially already [0,1] normalized heatmaps
        self.final_features_to_heatmaps = Normalize(args.z_width, 'softargmax', args.cell_softargmax_beta, False) if args.cell_norm=='None' else lambda x:x #output of cell possibly already normalized
        self.heatmaps_to_keypoints = HeatmapsToKeypoints()

    def forward(self, x, mode='implicit', args=None, z0=None, save_trajectory=False):
        """
        mode = 'implicit' or 'explicit'. Explicit is done with weight sharing
        zc only added for mode==implicit_broyden_strategy1_forward_only
        """

        x = self.tail(x)
        z_star, z_star_copy, fwd_logs, trajectory = self.deq_layer(x, mode, args, z0, save_trajectory) #z0 and z_star can be tensors or lists of tensors.
        out = self.final_features_to_heatmaps(z_star)
        preds, uncertainty = self.heatmaps_to_keypoints(out[:,:self.args.n_keypoints,:,:])
        results = {'keypoints': preds, 'uncertainty': uncertainty, 'fwd_logs': fwd_logs, 'z_star':z_star_copy, 'trajectory':trajectory}

        return results



if __name__=='__main__':
    from utils.helpers import *
    import torchinfo
    args = type('config', (object,), {})()

    # args.model_name = "sequnet"
    args.landmark_model_name = "udeq"
    args.cell_name = "Cell0"
    args.model_mode = 'explicit'
    args.explicit_depth = 2
    args.injection_width = 224
    args.cell_base_width = 224
    args.cell_width_increment = 24
    args.cell_gn_groups = 1
    args.cell_use_bn_for_explicit = True

    args.cell_norm = 'softargmax'
    args.cell_softargmax_beta = 1.0
    args.cell_learn_softargmax_beta = False

    args.solver='anderson'
    args.take_one_less_inference_step=False
    args.stop_mode='rel'
    args.abs_diff_target=5e-3
    args.rel_diff_target=5e-3
    args.fpi_max_iters=10
    args.anderson_max_iters=10
    args.anderson_m=6
    args.anderson_lam=1e-4
    args.anderson_beta = 1.0
    args.verbose_solver = True
    args.JFB = False

    args.im_size=256
    args.n_keypoints = 98
    args.cell_extra_latent_width = 0
    args.input_channels, args.heatmap_size = 3, 64
    args.downres_factor = args.im_size // args.heatmap_size
    args.z_width = args.n_keypoints + args.cell_extra_latent_width

    model = LDEQ(args).cuda()
    images = torch.randn((2, 3, args.im_size, args.im_size)).cuda()
    z0 = torch.zeros(2, args.z_width, args.heatmap_size, args.heatmap_size, device='cuda')

    out = model(images, args.model_mode, args, z0)
    print(out['keypoints'].shape)
    print(model)


