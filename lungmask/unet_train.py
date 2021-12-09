from lungmask.resunet import UNet
import torch
from lungmask.utils1 import make_dataset
import torch.utils.data as data
from abc import ABC
import os
import SimpleITK as sitk
import time
import numpy as np
from lungmask import utils
# from torchvision.transforms import transforms
# import matplotlib.pyplot as plt
# import scipy.ndimage as ndimage
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseDataset(data.Dataset, ABC):
    def __init__(self, dir):
        """
        dir: File directory.
        """
        self.dir = dir
        self.img_list = sorted(make_dataset(dir, mode='data'))
        self.roi_list = sorted(make_dataset(dir, mode='roi'))
        self.A_size = len(self.img_list)  # get the size of dataset
        self.B_size = len(self.roi_list)  # get the size of roi-set

        assert(self.A_size == self.B_size)
        if self.A_size == 0:
            raise(RuntimeError("Found 0 datafiles in: " + dir))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.img_list[index % self.A_size]  # make sure index is within then range
        B_path = self.roi_list[index % self.B_size]  # make sure index is within then range

        # apply image transformation
        A = sitk.ReadImage(A_path)  # data
        B = sitk.ReadImage(B_path)  # roi

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


class UNet_train(ABC):
    """
    Set a train set U-net model.
    """
    def __init__(self, in_channels, out_channels):
        # super(UNet_train, self).__init__()
        self.u_model = UNet(in_channels=in_channels, n_classes=out_channels).to(device)
        self.criterion = torch.nn.BCEWithLogitsLoss().to(device)
        self.optimizer = torch.optim.Adam(self.u_model.parameters())

    def set_input(self, input):
        """
        input: single image & roi file.
        """
        self.img_data = input['A']  # .to(device)
        self.roi_data = input['B']  # .to(device)
        self.preprocess()

    def preprocess(self):
        img_raw = sitk.GetArrayFromImage(self.img_data)
        roi_raw = sitk.GetArrayFromImage(self.roi_data)
        directions = np.asarray(self.img_data.GetDirection())
        if len(directions) == 9:
            img_raw = np.flip(img_raw, np.where(directions[[0, 4, 8]][::-1] < 0)[0])
            roi_raw = np.flip(roi_raw, np.where(directions[[0, 4, 8]][::-1] < 0)[0])

        tvolslices, labelslices, self.xnew_box = utils.preprocess(img_raw, label=roi_raw, resolution=[256, 256])
        # 取box是为了加速大矩阵的赋值，只需考虑身体的有效部分 灰度大于-500的躯干
        # tvolslices[tvolslices > 600] = 600
        self.tvolslices = np.divide((tvolslices + 1024), 1624)
        torch_ds_val = utils.LungLabelsDS_inf(self.tvolslices, labelset=labelslices)
        self.dataloader_val = torch.utils.data.DataLoader(torch_ds_val, batch_size=1, shuffle=False, num_workers=1,
                                                     pin_memory=False)

    def optimize_parameters(self):
        # data_pred = np.empty((np.append(0, self.tvolslices[0].shape)), dtype=np.uint8)
        # roi_pred = np.empty((np.append(0, self.tvolslices[0].shape)), dtype=np.uint8)
        Loss = 0
        epoch_loss = 0

        for i, (X, Y) in enumerate(self.dataloader_val):
            # print('i %s' % i)
            if (i % 2 == 0) and (i // 80 == 0):
                # 小于80层 且取 其偶数层
                # print('i %s' % i)
                # X dataset; Y labelset
                X = X.float().to(device)
                # Y = torch.unsqueeze(torch.tensor(ndimage.zoom(Y[0,0], 68/256, order=0)),0)
                trans_func = transforms.Compose([transforms.Resize((68, 68))])
                Y = trans_func(Y.float().to(device))

                prediction = self.u_model(X)
                # self.pls = torch.max(prediction, 1)[0].cpu()  # .numpy().astype(np.uint8)
                # plt.imshow(pls[0])

                # data_pred = np.vstack((data_pred, pls))
                # roi_pred = np.vstack((roi_pred, Y))
                # Loss += self.criterion(self.pls, Y.float().to(device))
                Loss += self.criterion(prediction, Y)

        # Loss = self.criterion(torch.from_numpy(data_pred.astype(np.float)),
        #                       torch.from_numpy(roi_pred.astype(np.float)))
        self.optimizer.zero_grad()
        Loss.backward()
        epoch_loss += Loss.item()
        print(epoch_loss)
        self.optimizer.step()


if __name__ == '__main__':
    # build up dataset/model
    dir = r'D:\Work\Data\TestData4Unet\newdata'
    dataset = BaseDataset(dir)
    model = UNet_train(in_channels=1, out_channels=1)
    total_iters = 0
    torch.cuda.empty_cache()

    # state_dict = torch.load(os.path.join(dir, 'epoch%s_iter%s.pth' % (1, 9)))
    # model.u_model.load_state_dict(state_dict)

    for epoch in range(1, 5):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_start_time = time.time()  # timer for data loading per iteration

        for i, data in enumerate(dataset):
            print('Total_iters %s' % total_iters)
            iter_now_time = time.time()
            total_iters += 1

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % 10 == 0:
                time_pass = - iter_start_time + iter_now_time
                print('Total Iters is %s and passing time is %s' % (total_iters, time_pass))
                torch.save(model.u_model.cpu().state_dict(), os.path.join(dir,'epoch%s_iter%s.pth'%(epoch, i)))
                # state_dict = torch.load(os.path.join(dir, 'epoch%s_iter%s.pth' % (1, 1)))
                # model.u_model.load_state_dict(state_dict)
                model.u_model.to(device)  # 保存完了一定要这一步

        epoch_now_time = time.time()
        print('Total Epoch is %s and passing time is %s' % (epoch+1, epoch_now_time-epoch_start_time))

