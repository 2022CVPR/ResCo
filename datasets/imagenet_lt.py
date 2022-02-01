from torch.utils.data import Dataset
from PIL import Image
import os

class ImageNet(Dataset):
    def __init__(self, root, train, transform=None):
        super(ImageNet, self).__init__()
        self.img_path = []
        self.targets = []
        self.transform = transform
        if train:
            data_txt = os.path.join(root, 'ImageNet_LT_train.txt')
        else:
            data_txt = os.path.join(root, 'ImageNet_LT_test.txt')            
        
        with open(data_txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))
        
        self.num_classes = 1000
        self.cls_num_list = [0] * self.num_classes
        for i in range(len(self.targets)):
            self.cls_num_list[int(self.targets[i])] += 1
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.targets[index]

        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            if self.transform is not None and type(self.transform) == list:
                img_org = self.transform[0](img)
                img_cont = self.transform[1](img)
                return img_org, img_cont, label

            # evaluating with one view
            elif self.transform is not None:
                img_org = self.transform(img)
                return img_org, label