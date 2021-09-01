
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 12:12:08 2018

@author: https://discuss.pytorch.org/t/fastest-way-to-find-nearest-neighbor-for-a-set-of-points/5938/12

slightly adapted
"""

import torch
from utils.se3 import *
from transforms3d.quaternions import mat2quat, quat2mat, qmult, qinverse
from torch_rot_utils import allo_to_ego_mat_torch, quat2mat_torch

def expanded_pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    if y is not None:
         differences = x.unsqueeze(1) - y.unsqueeze(0)
    else:
        differences = x.unsqueeze(1) - x.unsqueeze(0)
    distances = torch.sum(differences * differences, -1)
    return distances


def batch_expanded_pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    if y is not None:
         differences = x.unsqueeze(2) - y.unsqueeze(1)
    else:
        differences = x.unsqueeze(2) - x.unsqueeze(1)
    distances = torch.sum(differences * differences, -1)
    return distances



def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
    #print('zz', zz)
    #print('xx', xx)
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    #print('rx', rx)
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    P = (rx.t() + ry - 2*zz)
    return P

def NN_loss(x, y, dim=0):
    dist = pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    return values.mean()

def batch_pairwise_dist(a,b):
    x,y = a,b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2,1))
    yy = torch.bmm(y, y.transpose(2,1))
    zz = torch.bmm(x, y.transpose(2,1))
    #print('zz', zz.min().item())
    diag_ind = torch.arange(0, num_points, device=a.device, dtype=torch.long )
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2,1) + ry - 2*zz)
    return P

#def batch_NN_loss(x, y, dim=1):
#    assert dim != 0
#    pdb.set_trace()
#    dist = batch_pairwise_dist(x,y)
#    values, indices = dist.min(dim=dim)
#    return values.mean(dim=-1)


    

class PLoss(torch.nn.Module):
    ''' Input: estimated points and ground truth points of shape (bs, 3, num_points)
    Output: average distance loss between points. Points that are closer than 
    the margin are not added to the loss'''
    def __init__(self, loss_function=torch.nn.MSELoss(), margin=0):
        super(PLoss, self).__init__()
        self.loss = loss_function
        self.margin = margin
        
    def forward(self, pts_est, pts_gt, symmetries=[], reduce=True):
        batchsize = pts_est.shape[0]
        #print('points', points.shape)
        num_points = pts_est.shape[-1]

        diffs = pts_est - pts_gt
        #print('diffs', diffs.shape, diffs.min().item(), diffs.max().item())
        diffs_sq = torch.pow(diffs,2)
        
        #assert diffs_sq.equal(torch.mul(diffs, diffs)), 'maybe that'
        #print('diffs_squared', diffs_sq.shape)
        distances = torch.sum(diffs_sq, dim=1)

        if reduce:
            #print('bs np ', (batchsize , num_points))
            return torch.sum(distances) / (2 * batchsize * num_points)
        else:
            return distances 
 
class SLoss(torch.nn.Module):
    ''' Input: estimated points and ground truth points of shape (bs, 3, num_points)
    Output: the l2 loss between the estimated transformed points and the corresponding
        nearest point from the ground truth rotated points.. ''' 
    def __init__(self, margin=0):
        super(SLoss, self).__init__()
        self.margin = margin
    
    def forward(self,  pts_est, pts_gt, symmetries=[], reduce=True):
        num_rois = pts_est.shape[0]
        #print('points', points.shape)
        num_points = pts_est.shape[-1]

        dists = batch_pairwise_dist(pts_est.transpose(-2,-1), pts_gt.transpose(-2,-1))

        #print('dists', dists.shape)
        distances = torch.min(dists, dim=2)[0]
        #print('distances', distances)
    
        if reduce:
            assert distances.shape[0] == num_rois and distances.shape[1] == num_points, 'distances has wrong shape'
            #print('bs np ', (batchsize , num_points))
            return torch.sum(distances) / (2* num_points * num_rois)
        else:
            #print('distances.shape', distances.shape)
            return distances # no division by 2

class PTS_Loss(torch.nn.Module):
    def __init__(self,margin=0):
        super(PTS_Loss,self).__init__()
        self.margin = margin
        self.sloss = SLoss()
        self.ploss = PLoss()

    def forward(self, q_est, q_gt, T, pts, symmetries=[], ego=False):
        #getting some values
        size = symmetries.nelement()
        num_sym = symmetries.sum()
        num_non_sym = size-num_sym
        symmetries = symmetries.bool().squeeze()

        #permuting the point clouds
        pts = pts.permute(0,2,1)

        #convert quaternions to matricies
        R_gt = quat2mat_torch(q_gt)
        R_est = quat2mat_torch(q_est)

        if ego:
            R_gt = allo_to_ego_mat_torch(T, R_gt)
       
        #applies rotation
        pts_est = R_est@pts
        pts_gt = R_gt@pts
        
        p_loss = self.ploss(pts_est[~symmetries], pts_gt[~symmetries])

        if symmetries.any():
            s_loss = self.sloss(pts_est[symmetries], pts_gt[symmetries])
        else:
            s_loss = 0

        return num_sym*s_loss/size + num_non_sym*p_loss/size

