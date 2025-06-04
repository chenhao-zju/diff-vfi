import os
import abc
import datetime
import random
import json

# import h5py
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# _all_data_ = ['boy_talk1_1', 'boy_talk2', 'boy_talk3_1', 'girl_talk2', 'girl_talk4_2', 'girl_talk5_1']
_all_data_ = ['boy_talk1_1', 'boy_talk4_1', 'girl_talk4_2', 'girl_talk5_2']


def rotation_vectors_to_quaternions(rotation_vectors):
    """
    将多个旋转向量批量转换为四元数
    :param rotation_vectors: 旋转向量数组，形状为 [N, 3]
    :return: 四元数数组，形状为 [N, 4]
    """
    thetas = np.linalg.norm(rotation_vectors, axis=1)
    thetas[thetas < 1e-6] = 1e-6
    
    u = rotation_vectors / thetas[:, np.newaxis]
    
    w = np.cos(thetas / 2)
    xyz = u * np.sin(thetas / 2)[:, np.newaxis]
    
    quaternions = np.hstack((w[:, np.newaxis], xyz))
    return quaternions


def quaternions_to_rotation_vectors(quaternions):
    """
    将多个四元数批量转换为旋转向量
    :param quaternions: 四元数数组，形状为 [N, 4]
    :return: 旋转向量数组，形状为 [N, 3]
    """
    w = quaternions[:, 0]
    xyz = quaternions[:, 1:]

    thetas = 2 * np.arccos(w)
    thetas[thetas < 1e-6] = 1e-6
    
    sin_half_theta = np.sin(thetas / 2)
    u = xyz / sin_half_theta[:, np.newaxis]
    rotation_vectors = u * thetas[:, np.newaxis]
    return rotation_vectors



def read_pkl(pkl_pth, whole_body=False):
    with open(pkl_pth, 'rb') as f:
        all_dataset = pickle.load(f)[0]

    cur_jaw, cur_leyes, cur_reyes, cur_exp, betas, cur_transl, cur_pose, cur_global_orient, cur_left_hand_pose, cur_right_hand_pose = \
        all_dataset['jaw_pose'], all_dataset['leye_pose'], all_dataset['reye_pose'], all_dataset['expression'], all_dataset['betas'], \
        all_dataset['transl'], all_dataset['body_pose_axis'], all_dataset['global_orient'], all_dataset['left_hand_pose'], all_dataset['right_hand_pose']

    cur_global_orient = rotation_vectors_to_quaternions(cur_global_orient.squeeze())

    poses = np.concatenate((cur_global_orient, cur_pose, cur_jaw, cur_leyes, cur_reyes, cur_left_hand_pose, cur_right_hand_pose), axis=-1)
    expressions = cur_exp

    B = poses.shape[0]

    if whole_body:
        # print(pkl_pth, betas.shape, cur_transl.shape, poses.shape, expressions.shape)
        body = np.concatenate(( cur_transl, poses, expressions ), axis=-1)
        global_info = np.concatenate(( cur_transl, cur_global_orient ), axis=-1)
        return global_info, body, all_dataset
    else:
        return poses, expressions, all_dataset


def write_pkl(poses, expressions, all_dataset, start_video, start_index, fixed_number, mask_ratio, pkl_pth):
    index = start_index-start_video

    cur_global_orient, cur_pose, cur_jaw, cur_leyes, cur_reyes, cur_left_hand_pose, cur_right_hand_pose \
    = poses[index: index+mask_ratio, :3], poses[index: index+mask_ratio, 3:66], poses[index: index+mask_ratio, 66:69], poses[index: index+mask_ratio, 69:72], poses[index: index+mask_ratio, 72:75], poses[index: index+mask_ratio, 75:87], poses[index: index+mask_ratio, 87:99]

    cur_exp = expressions[index: index+mask_ratio]

    all_dataset['jaw_pose'] = cur_jaw
    all_dataset['leye_pose'] = cur_leyes 
    all_dataset['reye_pose'] = cur_reyes
    all_dataset['expression'] = cur_exp
    all_dataset['body_pose_axis'] = cur_pose
    all_dataset['global_orient'] = cur_global_orient[:, None, :]
    all_dataset['left_hand_pose'] = cur_left_hand_pose
    all_dataset['right_hand_pose'] = cur_right_hand_pose

    all_dataset['betas'] = np.repeat(a=all_dataset['betas'][0][start_index-1], repeats=mask_ratio, axis=0)[None, :]
    all_dataset['transl'] = np.repeat(a=all_dataset['transl'][start_index-1][None, :], repeats=mask_ratio, axis=0)

    all_dataset['batch_size'] = mask_ratio
    all_dataset['pose_embedding'] = all_dataset['pose_embedding'][start_index: start_index+mask_ratio]

    with open(pkl_pth, 'wb') as handle:
        pickle.dump(all_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_dataset


def write_body_pkl(body, all_dataset, start_video, start_index, fixed_number, mask_ratio, pkl_pth):
    index = start_index-start_video

    # betas, transl, poses, expressions = body[:, 0], body[:, 1:4], body[:, 4:103], body[:, 103: 203]
    # transl, poses, expressions = body[:, :3], body[:, 3:102], body[:, 102:]
    transl, poses, expressions = body[:, :3], body[:, 3:103], body[:, 103:]

    # cur_global_orient, cur_pose, cur_jaw, cur_leyes, cur_reyes, cur_left_hand_pose, cur_right_hand_pose \
    # = poses[index: index+mask_ratio, :3], poses[index: index+mask_ratio, 3:66], poses[index: index+mask_ratio, 66:69], poses[index: index+mask_ratio, 69:72], poses[index: index+mask_ratio, 72:75], poses[index: index+mask_ratio, 75:87], poses[index: index+mask_ratio, 87:99]
    cur_global_orient, cur_pose, cur_jaw, cur_leyes, cur_reyes, cur_left_hand_pose, cur_right_hand_pose \
    = poses[index: index+mask_ratio, :4], poses[index: index+mask_ratio, 4:67], poses[index: index+mask_ratio, 67:70], poses[index: index+mask_ratio, 70:73], poses[index: index+mask_ratio, 73:76], poses[index: index+mask_ratio, 76:88], poses[index: index+mask_ratio, 88:100]

    cur_exp = expressions[index: index+mask_ratio]

    # cur_betas = betas[index: index+mask_ratio]
    cur_transl = transl[index: index+mask_ratio]

    cur_global_orient = quaternions_to_rotation_vectors(cur_global_orient)
    print(cur_global_orient.shape)

    all_dataset['jaw_pose'] = cur_jaw
    all_dataset['leye_pose'] = cur_leyes 
    all_dataset['reye_pose'] = cur_reyes
    all_dataset['expression'] = cur_exp
    all_dataset['body_pose_axis'] = cur_pose
    all_dataset['global_orient'] = cur_global_orient[:, None, :]
    # all_dataset['global_orient'] = np.repeat(a=all_dataset['global_orient'][start_index-1][None, :], repeats=mask_ratio, axis=0)
    all_dataset['left_hand_pose'] = cur_left_hand_pose
    all_dataset['right_hand_pose'] = cur_right_hand_pose

    # all_dataset['betas'] = cur_betas[None, :]
    all_dataset['transl'] = cur_transl
    # all_dataset['transl'] = np.repeat(a=all_dataset['transl'][start_index-1][None, :], repeats=mask_ratio, axis=0)

    all_dataset['batch_size'] = mask_ratio
    # all_dataset['pose_embedding'] = all_dataset['pose_embedding'][start_index: start_index+mask_ratio]
    all_dataset.pop('pose_embedding')

    with open(pkl_pth, 'wb') as handle:
        pickle.dump(all_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_dataset


def write_body_pkl_100(body, all_dataset, start_video, start_index, fixed_number, mask_ratio, pkl_pth):
    # index = start_index-start_video
    index = start_index

    # betas, transl, poses, expressions = body[:, 0], body[:, 1:4], body[:, 4:103], body[:, 103: 203]
    # transl, poses, expressions = body[:, :3], body[:, 3:102], body[:, 102:]
    transl, poses, expressions = body[:, :3], body[:, 3:103], body[:, 103:]

    # cur_global_orient, cur_pose, cur_jaw, cur_leyes, cur_reyes, cur_left_hand_pose, cur_right_hand_pose \
    # = poses[index: index+mask_ratio, :3], poses[index: index+mask_ratio, 3:66], poses[index: index+mask_ratio, 66:69], poses[index: index+mask_ratio, 69:72], poses[index: index+mask_ratio, 72:75], poses[index: index+mask_ratio, 75:87], poses[index: index+mask_ratio, 87:99]
    cur_global_orient, cur_pose, cur_jaw, cur_leyes, cur_reyes, cur_left_hand_pose, cur_right_hand_pose \
    = poses[index: index+mask_ratio, :4], poses[index: index+mask_ratio, 4:67], poses[index: index+mask_ratio, 67:70], poses[index: index+mask_ratio, 70:73], poses[index: index+mask_ratio, 73:76], poses[index: index+mask_ratio, 76:88], poses[index: index+mask_ratio, 88:100]

    cur_exp = expressions[index: index+mask_ratio]

    # cur_betas = betas[index: index+mask_ratio]
    cur_transl = transl[index: index+mask_ratio]

    print(cur_global_orient.shape)
    cur_global_orient = quaternions_to_rotation_vectors(cur_global_orient)
    print(cur_global_orient.shape)

    # all_dataset['jaw_pose'] = all_dataset['jaw_pose'][start_video:start_video+fixed_number]
    # all_dataset['jaw_pose'][index:index+mask_ratio] = cur_jaw
    # all_dataset['leye_pose'] = all_dataset['leye_pose'][start_video:start_video+fixed_number]
    # all_dataset['leye_pose'][index:index+mask_ratio] = cur_leyes 
    # all_dataset['reye_pose'] = all_dataset['reye_pose'][start_video:start_video+fixed_number]
    # all_dataset['reye_pose'][index:index+mask_ratio] = cur_reyes
    # all_dataset['expression'] = all_dataset['expression'][start_video:start_video+fixed_number]
    # all_dataset['expression'][index:index+mask_ratio] = cur_exp
    # all_dataset['body_pose_axis'] = all_dataset['body_pose_axis'][start_video:start_video+fixed_number]
    # all_dataset['body_pose_axis'][index:index+mask_ratio] = cur_pose
    all_dataset['global_orient'] = all_dataset['global_orient'][start_video:start_video+fixed_number]
    all_dataset['global_orient'][index:index+mask_ratio] = cur_global_orient[:, None, :]
    # # all_dataset['global_orient'] = np.repeat(a=all_dataset['global_orient'][start_index-1][None, :], repeats=mask_ratio, axis=0)
    # all_dataset['left_hand_pose'] = all_dataset['left_hand_pose'][start_video:start_video+fixed_number]
    # all_dataset['left_hand_pose'][index:index+mask_ratio] = cur_left_hand_pose
    # all_dataset['right_hand_pose'] = all_dataset['right_hand_pose'][start_video:start_video+fixed_number]
    # all_dataset['right_hand_pose'][index:index+mask_ratio] = cur_right_hand_pose

    # all_dataset['betas'] = cur_betas[None, :]
    all_dataset['transl'] = all_dataset['transl'][start_video:start_video+fixed_number]
    all_dataset['transl'][index:index+mask_ratio] = cur_transl
    # all_dataset['transl'] = np.repeat(a=all_dataset['transl'][start_index-1][None, :], repeats=mask_ratio, axis=0)

    all_dataset['batch_size'] = fixed_number
    # all_dataset['pose_embedding'] = all_dataset['pose_embedding'][start_index: start_index+mask_ratio]
    all_dataset.pop('pose_embedding')

    with open(pkl_pth, 'wb') as handle:
        pickle.dump(all_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return all_dataset


class PoseData(Dataset):

    def __init__(self,
                 data_path='/home/bingxing2/ailab/group/ai4earth/haochen/dataset/pose_dataset/anchor_allposes',
                 fixed_number=100,
                 run_mode='train',
                 train_wholebody=False):

        super(PoseData, self).__init__()
        
        self.data_path = data_path
        self.fixed_number = fixed_number
        self.train_wholebody = train_wholebody

        self._get_all_files()
        self._get_mean_std()
        # print(self.all_pths)


    def _get_all_files(self):
        self.all_pths = []
        for root, dirs, files in os.walk(self.data_path):
            for name in files:
                if name == 'final_all.pkl':
                    src_pth = os.path.join(root, name)

                    self.all_pths.append(src_pth)

    def _get_mean_std(self):
        if self.train_wholebody:
            all_body = {name: [] for name in _all_data_ }

            for pth in self.all_pths:
                name = pth.split('/')[-5]
                global_info, body, _ = read_pkl(pth, whole_body=self.train_wholebody)
                all_body[name].append(global_info)

            all_body = {key: np.concatenate(value, axis=0) for key, value in all_body.items()}
  
            self.body_mean = {key: np.mean(value, axis=0) for key, value in all_body.items()} 
            self.body_std =  {key: np.std(value, axis=0) for key, value in all_body.items()} 

            mean_std = {'body_mean': {key: value.tolist() for key, value in self.body_mean.items() }, 'body_std': {key: value.tolist() for key, value in self.body_std.items() },  
                        }

            mean_std = json.dumps(mean_std)
            f2 = open(self.data_path + '/body_mean_std.json', 'w')
            f2.write(mean_std)
            f2.close()

        else:
            all_poses = {name: [] for name in _all_data_ }
            all_expressions = {name: [] for name in _all_data_ }
            
            for pth in self.all_pths:
                name = pth.split('/')[-5]
                poses, expressions, _ = read_pkl(pth, whole_body=self.train_wholebody)

                all_poses[name].append(poses)
                all_expressions[name].append(expressions)

            all_poses = {key: np.concatenate(value, axis=0) for key, value in all_poses.items()}
            all_expressions = {key: np.concatenate(value, axis=0) for key, value in all_expressions.items()}

            self.poses_mean = {key: np.mean(value, axis=0) for key, value in all_poses.items()} 
            self.poses_std = {key: np.std(value, axis=0) for key, value in all_poses.items()} 

            self.expressions_mean = {key: np.mean(value, axis=0) for key, value in all_expressions.items()} 
            self.expressions_std = {key: np.std(value, axis=0) for key, value in all_expressions.items()}


            mean_std = {'poses_mean': {key: value.tolist() for key, value in self.poses_mean.items() }, 'poses_std': {key: value.tolist() for key, value in self.poses_std.items() }, 'expressions_mean': {key: value.tolist() for key, value in self.expressions_mean.items() }, 'expressions_std': {key: value.tolist() for key, value in self.expressions_std.items() } }

            mean_std = json.dumps(mean_std)
            f2 = open(self.data_path + '/mean_std.json', 'w')
            f2.write(mean_std)
            f2.close()


    def __len__(self):
        length = len(self.all_pths)
        return length


    def __getitem__(self, idx):
        data_pth = self.all_pths[idx]
        self.name = data_pth.split('/')[-5]

        if self.train_wholebody:
            global_info, body, _ = read_pkl(data_pth, self.train_wholebody)
            # body = self._process_fn(body)
            # return body
            global_info = self._process_fn(global_info)
            return global_info
        else:
            poses, expressions, _ = read_pkl(data_pth, self.train_wholebody)
            poses, expressions = self._process_fn( (poses, expressions) )
            return poses, expressions


    def _normalize(self, samples):
        if self.train_wholebody:
            samples = (samples - self.body_mean[self.name]) / self.body_std[self.name]
            return samples
        else:
            poses, expressions = samples
            poses = (poses - self.poses_mean[self.name]) / self.poses_std[self.name]
            expressions = (expressions - self.expressions_mean[self.name]) / self.expressions_std[self.name]
            return (poses, expressions)


    def _fix_dataset(self, samples):
        if self.train_wholebody:
            pose_length = samples.shape[0]
            self.num_poses = (pose_length // self.fixed_number) + 1
            samples_list = [samples[i*self.fixed_number:(i+1)*self.fixed_number] for i in range(self.num_poses)]

            if samples_list[-1].shape[0] != self.fixed_number:
                samples_list[-1] = samples[-self.fixed_number:]

            all_samples = np.stack(samples_list, axis=0)
            return all_samples
        
        else:
            poses, expressions = samples
            pose_length = poses.shape[0]

            self.num_poses = (pose_length // self.fixed_number) + 1

            poses_list = [poses[i*self.fixed_number:(i+1)*self.fixed_number] for i in range(self.num_poses)]
            expressions_list = [expressions[i*self.fixed_number:(i+1)*self.fixed_number] for i in range(self.num_poses)]

            if poses_list[-1].shape[0] != self.fixed_number:
                poses_list[-1] = poses[-self.fixed_number:]
            if expressions_list[-1].shape[0] != self.fixed_number:
                expressions_list[-1] = expressions[-self.fixed_number:]

            all_poses = np.stack(poses_list, axis=0)
            all_expressions = np.stack(expressions_list, axis=0)

            return (all_poses, all_expressions)


    def _process_fn(self, samples):
        '''process_fn'''
        samples = self._normalize(samples)
        samples = self._fix_dataset(samples)

        if self.train_wholebody:
            body = np.squeeze(samples)
            return torch.tensor(body)

        else:
            poses, expressions = samples
            poses = np.squeeze(poses)
            expressions = np.squeeze(expressions)
            return torch.tensor(poses), torch.tensor(expressions)


def get_data_loader_npy(data_path='/home/bingxing2/ailab/group/ai4earth/haochen/dataset/anchor_allposes', fixed_number=100, run_mode='train', train_wholebody=False, distributed=False):

    dataset = PoseData(data_path=data_path,
                        fixed_number=fixed_number,
                        run_mode='train',
                        train_wholebody=train_wholebody)

    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=5,
                            shuffle=False, #(sampler is None),
                            sampler=sampler if run_mode=='train' else None,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available())

    return dataloader, dataset


if __name__ == '__main__':
    from tqdm import tqdm
    from einops import rearrange
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    dataloader, dataset = get_data_loader_npy(data_path='/home/bingxing2/ailab/group/ai4earth/haochen/dataset/pose_dataset/anchor_trainposes', fixed_number=70, run_mode='train', train_wholebody=True, distributed=False)

    all_pths = dataset.all_pths

    for idx, data in tqdm(enumerate(dataloader)):
        try:
            data = rearrange(data, 'b n l c -> (b n) l c')
            print(data.shape)
        except:
            print(all_pths[idx])

        # if idx == 2: exit()

        # data = rearrange(data, 'b n l c -> (b n) l c')
        # diff_result = data[:, 1:].diff(axis=-1)- data[:, :-1].diff(axis=-1)
        # diff_result = diff_result.pow(2).mean(dim=-1)
        # print( diff_result>1.1 )
        # print(data.shape, diff_result.shape)
        # if idx == 2: exit()
