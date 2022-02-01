import nori2 as nori
from torch.utils.data import Dataset
from PIL import Image
import io
import os
 
class ImageNet(Dataset):
    def __init__(self, root, train, transform=None):
        super(ImageNet, self).__init__()
        self.nori_fetcher = None
        self.f_list = []
        if train:
            data_dir = os.path.join(root, 'imagenet.lt_train.nori.list')
        else:
            data_dir = os.path.join(root, 'imagenet.val.nori.list')
 
        with open(data_dir) as g:
            l = g.readline()
            while l:
                ls = l.split()
                self.f_list.append(ls)
                l = g.readline()

        self.num_classes = 1000
        self.cls_num_list = [0] * self.num_classes
        for i in range(len(self.f_list)):
            self.cls_num_list[int(self.f_list[i][1])] += 1
        self.transform = transform
 
    def __getitem__(self, idx):
        if self.nori_fetcher is None:
            self.nori_fetcher = (
                nori.Fetcher() 
            )
        ls = self.f_list[idx]
        img = Image.open(io.BytesIO(self.nori_fetcher.get(ls[0])))
        img = img.convert('RGB')
        label = int(ls[1])
        
        # training with two views
        if self.transform is not None and type(self.transform) == list:
            img_org = self.transform[0](img)
            img_cont = self.transform[1](img)
            return img_org, img_cont, label

        # evaluating with one view
        elif self.transform is not None:
            img_org = self.transform(img)
            return img_org, label
 
    def __len__(self):
        return len(self.f_list)