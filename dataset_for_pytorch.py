import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pypcd4 import PointCloud


class MyPointDataset(Dataset):
    def __init__(self, root_dir, pointset='train'):
        self.root_dir = root_dir
        self.pointset = pointset
        #数据路径
        self.data_path = os.path.join(root_dir, pointset)
        self.data_list = os.listdir(self.data_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        point_name = self.data_list[index]
        path = os.path.join(self.data_path, point_name)
        point = PointCloud.from_path(path)
        point = point.numpy()

        label = point[:, -1]
        point = np.delete(point, (3, 4), axis=1)

        return point, label

def collate_fn(data):
    data2stack = np.concatenate([item[0] for item in data], axis=0).astype(np.float32)
    label2stack = np.concatenate([item[1] for item in data], axis=0).astype(int)

    return data2stack, label2stack

if __name__ == '__main__':
    dataset = MyPointDataset(root_dir='./point_file', pointset='train')
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=collate_fn)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i,j in enumerate(dataloader):
        point, label = j
        print(point.shape)
        # print(label.shape)


