import torch
import numpy as np
import copy
from metro.utils.geometric_layers import orthographic_projection


class GAZEFROMBODY(torch.nn.Module):

    def __init__(self, args, bert):
        super(GAZEFROMBODY, self).__init__()
        self.bert = bert
        self.encoder1 = torch.nn.Linear(3*14,32)
        self.tanh = torch.nn.Tanh()
        self.encoder2 = torch.nn.Linear(32,3)
        self.flatten  = torch.nn.Flatten()
        self.flatten2  = torch.nn.Flatten()

        self.metromodule = copy.deepcopy(bert)
        self.body_mlp1 = torch.nn.Linear(14*3,32)
        #self.body_mlp1 = torch.nn.Linear(args.n_frames*14*3,32)
        self.body_tanh1 = torch.nn.PReLU()
        self.body_mlp2 = torch.nn.Linear(32,32)
        self.body_tanh2 = torch.nn.PReLU()
        self.body_mlp3 = torch.nn.Linear(32,3)

        self.lstm = torch.nn.LSTM(3, 256, 1, batch_first=False)
        self.label = torch.nn.Linear(256,3)

        self.n_frames = args.n_frames
        self.device = args.device


    def transform_head(self, pred_3d_joints):
        Nose = 13

        pred_head = pred_3d_joints[:, Nose,:]
        return pred_3d_joints - pred_head[:, None, :]

    def transform_body(self, pred_3d_joints):
        Torso = 12

        pred_torso = pred_3d_joints[:, Torso,:]
        return pred_3d_joints - pred_torso[:, None, :]


    def forward(self, image, images, smpl, mesh_sampler, is_train=False):
        batch_size = image.size(0)
        self.bert.eval()
        self.metromodule.eval()

        RSholder = 7
        LSholder = 10
        Nose = 13
        Head = 9
        Torso = 12

        with torch.no_grad():
            _, tmp_joints, _, _, _, _, _, _ = self.metromodule(image, smpl, mesh_sampler)

        ##pred_joints = torch.stack(pred_joints, dim=3)
        pred_joints = self.transform_head(tmp_joints)
        mx = self.flatten(pred_joints)
        mx = self.body_mlp1(mx)
        mx = self.body_tanh1(mx)
        mx = self.body_mlp2(mx)
        mx = self.body_tanh2(mx)
        mx = self.body_mlp3(mx)
        mdir = mx


        # metro inference
        # pred_xをテンソルとして初期化 (形状: [n_frames, batch_size, feature_size])
        pred_x = torch.zeros(self.n_frames, batch_size, 3).to(self.device)
        for i in range(self.n_frames):
            _, pred_3d_joints, _, _, _, _, _, _ = self.bert(images[i], smpl, mesh_sampler)
            pred_3d_joints_gaze = self.transform_head(pred_3d_joints)
            #pred_joints.append(tmp_joints)

            x = self.flatten(pred_3d_joints_gaze)
            x = self.encoder1(x)
            x = self.tanh(x)
            x = self.encoder2(x)# [batch, 3]
            pred_x[i] = x
        
        #print("len pred_x:",len(pred_x)) # n_frames 7
        #print("len pred_x:",len(pred_x[0])) # batch size 1
        #print("len pred_x:",len(pred_x[0][0])) # dir 3
        dir, _ = self.lstm(pred_x)
        dir = dir[self.n_frames//2, :, :]
        dir = self.label(dir)

        dir = dir + mx#/l2[:,None]

        if is_train == True:
            return dir, mdir
        if is_train == False:
            return dir#, pred_vertices
