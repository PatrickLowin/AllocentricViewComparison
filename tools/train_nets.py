import _init_paths
import torch
import torchvision
from resnet_pose import ResNet_Pose
from datasets.factory import  get_dataset
from utils.se3 import allocentric2egocentric, egocentric2allocentric
from loss import PTS_Loss
import visdom
from utils.pose_error import reproj, re
from transforms3d.quaternions import quat2mat, mat2quat
import numpy as np
from visdom import Visdom
from tqdm import tqdm

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

class Data_cutter(torch.utils.data.Dataset):
    def __init__(self,ycb_data, bs=8):
        self.ycb = ycb_data
        self.len = self.ycb.__len__()
        self.bs = bs

    def __getitem__(self,idx):
        bs_idx = 0
        img_batch = torch.zeros(self.bs, 3, 64, 64)
        pts_batch = torch.zeros(self.bs, 2620, 3)
        class_id_batch = torch.zeros(self.bs)
        symmetry_batch = torch.zeros(self.bs)
        R_batch = torch.zeros(self.bs, 4)
        T_batch = torch.zeros(self.bs, 3)
        whole_image_batch = []

        while bs_idx != self.bs:
            idx = np.random.randint(0, self.len)
            data = self.ycb.__getitem__(idx)
            bbox_ = [torch.Tensor(data['gt_boxes'])[:-1,:-1]]
            bbox = [(bbox_[0][bbox_[0].any(1)>0]).view(-1,4)]
            whole_image_batch.append(torch.nn.functional.interpolate(data['image_color'].unsqueeze(0), size=(128,128)))
            img = torchvision.ops.roi_align(data['image_color'].unsqueeze(0), bbox, output_size=(64,64), aligned=True)
            pts = data['points']
            symmetry = data['symmetry']
            poses = data['poses']

            class_id = poses[:, 1]
            class_id = class_id[class_id>0]
            class_id = class_id.astype(np.int8)

            T = poses[:,6:]
            R = poses[:,2:6]
            #assert ((pts.sum(1).sum(1)==0)==(R.sum(1)==0)).all()
            
            #remove background
            pts = pts[class_id]

            mask = R.sum(1)!=0
            R = R[mask]
            T = T[mask]
            symmetry = symmetry[mask]
            
            if R.ndim==1:
                R = torch.Tensor(R).unsqueeze(0)
                T = torch.Tensor(T).unsqueeze(0)
                pts = torch.Tensor(pts).unsqueeze(0)
                symmetry = symmetry.unsqueeze(0)
            if (T.sum(1)==0).any():
                import ipdb;ipdb.set_trace()
            
            variable_batch_size = R.shape[0]
            
            if variable_batch_size+bs_idx - self.bs > 0:
                
                remaining_space = self.bs - bs_idx
                
                rand_ind = torch.randint(0, variable_batch_size, (remaining_space,))
                
                img = img[rand_ind]
                pts = pts[rand_ind]
                class_id = class_id[rand_ind]
                symmetry = symmetry[rand_ind]
                R = R[rand_ind]
                T = T[rand_ind]

                variable_batch_size = remaining_space
            
                
            
            img_batch[bs_idx: bs_idx + variable_batch_size] = img
            R_batch[bs_idx: bs_idx + variable_batch_size] = torch.Tensor(R)
            T_batch[bs_idx: bs_idx + variable_batch_size] = torch.Tensor(T)

            if class_id.ndim > 0:
                class_id_batch[bs_idx: bs_idx + variable_batch_size] = torch.Tensor(class_id)
                symmetry_batch[bs_idx: bs_idx + variable_batch_size] = torch.Tensor(symmetry)
                pts_batch[bs_idx: bs_idx + variable_batch_size] = torch.Tensor(pts[:,0:2620])
            else:
                class_id_batch[bs_idx: bs_idx + variable_batch_size] = torch.tensor(class_id)
                symmetry_batch[bs_idx: bs_idx + variable_batch_size] = torch.tensor(symmetry)
                pts_batch[bs_idx: bs_idx + variable_batch_size] = torch.Tensor(pts[0:2620])
                
            bs_idx += variable_batch_size
            
        return {'img':img_batch, 'pts':pts_batch, 'classes':class_id_batch, 'symmetry':symmetry_batch, 'R': R_batch, 'T':T_batch, 'whole_img': torch.stack(whole_image_batch)} #, 'bbox':bbox , 'whole_img':data['image_color']

    def __len__(self):
        return self.len 


device = 'cuda:0'
#DataLoader
ycb_cropped = Data_cutter(get_dataset('ycb_video_train'), bs=32)
dataloader = torch.utils.data.DataLoader(ycb_cropped, batch_size=1, drop_last=True)
num_classes = ycb_cropped.ycb._num_classes-1

plotter = VisdomLinePlotter(env_name='Comparison Allocentric vs Egocentric')
vis = visdom.Visdom(env='Comparison Allocentric vs Egocentric')

#create 2 models that are trained in parallel for ego- and allocentric view
model_ego = ResNet_Pose().to(device)
model_allo = ResNet_Pose().to(device)

#optimizer
optim_ego = torch.optim.Adam(model_ego.parameters(), lr=1e-04)
optim_allo = torch.optim.Adam(model_allo.parameters(), lr=1e-04)

scheduler_ego = torch.optim.lr_scheduler.CosineAnnealingLR(optim_ego, T_max= 5000)
scheduler_allo = torch.optim.lr_scheduler.CosineAnnealingLR(optim_allo, T_max= 5000)
pts_loss = PTS_Loss().to(device)

data_len = dataloader.__len__()

#train loop
for epoch in range(0,10000):
    all_errs = []
    ego_errs = []

    re_errors_allo = []
    re_errors_ego = []
    counts_counter = 0
    for i, data in tqdm(enumerate(dataloader)):
        #load data
        img = data['img'].squeeze().to(device)

        if img.dim()<4:
            img = img.unsqueeze(0).to(device)

        quat = data['R'].squeeze(0).to(device)
        T = data['T'].squeeze(0).to(device)
        classes = data['classes'].squeeze(0).to(device)
        pts = data['pts'].squeeze(0).to(device)
        bs = img.shape[0]
        syms = data['symmetry'].squeeze(0).to(device)
        #print(pts.shape)       

        output_ego = model_ego(img) #bs, 22,4
        output_allo = model_allo(img)#bs, 22,4
        
        idx0 = torch.arange(0,bs)
        #get output for category
        output_ego = output_ego[idx0,classes.long()]
        output_allo = output_allo[idx0,classes.long()]

        #calculate step for egocentric model  
        ego_loss = pts_loss(quat, output_ego, T, pts, syms, ego=True)
        ego_loss.backward()
        optim_ego.step()

        
        #calculate step for allocentric model
        allo_loss = pts_loss(quat, output_allo, T, pts, syms)
        allo_loss.backward()
        optim_allo.step()

        ego_errs.append(ego_loss)
        all_errs.append(allo_loss)
        counts_counter += (~syms.int()*(-1)).sum()
        
        assert not (ego_loss.isnan() or allo_loss.isnan())
        assert not (ego_loss.isinf() or allo_loss.isinf())

        for j in range(quat.shape[0]):
            if syms[j]==0:
                re_errors_allo.append(re(quat2mat(quat[j].detach().cpu().numpy()), quat2mat(output_allo[j].detach().cpu().numpy())))
                re_errors_ego.append(re(quat2mat(quat[j].detach().cpu().numpy()), quat2mat(output_ego[j].detach().cpu().numpy())))
        
        if i%100==0:
            #check data
            vis.images(torch.clamp(data['whole_img'].squeeze()+0.5, 0.,1.), win='whole image')
            vis.images(torch.clamp(img+0.5, 0.,1.), win='ROIS')
            
            # for k,pc in enumerate(pts):
            #     vis.scatter(pc, win=str(k))

            ego_loss = torch.Tensor(ego_errs).mean()
            allo_loss = torch.Tensor(all_errs).mean()

            counts_allo = torch.zeros(120)
            counts_ego = torch.zeros(120)

            for j in range(0,len(re_errors_allo)):
                counts_allo[int(re_errors_allo[j]):]+=1
                counts_ego[int(re_errors_ego[j]):]+=1

            counts_allo = counts_allo/counts_counter.cpu()
            counts_ego = counts_ego/counts_counter.cpu()
            
            vis.bar(counts_allo, win='allocentric', opts = dict(title='allocentric'))
            vis.bar(counts_ego, win='egocentric', opts = dict(title='egocentric'))

            #print(epoch, data_len, i)
            plotter.plot('AP Integral', 'allocentric','AP Integral', epoch*data_len + i, np.trapz(counts_allo)/119)
            plotter.plot('AP Integral', 'egocentric','AP Integral', epoch*data_len + i, np.trapz(counts_ego)/119)

            plotter.plot('View Error', 'egocentric','View Error', epoch*data_len + i, ego_loss )
            plotter.plot('View Error', 'allocentric','View Error', epoch*data_len + i, allo_loss)

            all_errs = []
            ego_errs = []

            re_errors_allo = []
            re_errors_ego = []
            counts_counter = 0

    scheduler_allo.step()
    scheduler_ego.step()
    torch.save(model_allo.state_dict(),'../checkpoints_allo_ego/model_allo.pth')
    torch.save(model_ego.state_dict(),'../checkpoints_allo_ego/model_ego.pth')
    print('['+str(i)+'/16] Loss Egocentric',ego_loss.item())
    print('['+str(i)+'/16] Loss Allocentric',allo_loss.item())
    print('---------------------------------------------')