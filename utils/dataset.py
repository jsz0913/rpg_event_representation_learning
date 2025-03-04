import numpy as np
from os import listdir
from os.path import join


def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    # 把事件漂移，筛选
    x_shift, y_shift = np.random.randint(-max_shift, max_shift+1, size=(2,))
    events[:,0] += x_shift
    events[:,1] += y_shift
    valid_events = (events[:,0] >= 0) & (events[:,0] < W) & (events[:,1] >= 0) & (events[:,1] < H)
    events = events[valid_events]

    return events

def random_flip_events_along_x(events, resolution=(180, 240), p = 0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:,0] = W - 1 - events[:,0]
    return events

## dataset必须包含 __len__   __getitem__
class NCaltech101:
    def __init__(self, root, augmentation=False):
        # 路径下文件名列表
        self.classes = listdir(root)

        self.files = []
        self.labels = []

        self.augmentation = augmentation

        for i, c in enumerate(self.classes):
            # i 是 从 0 开始循环
            # root下每个文件夹下所有文件名
            # root/c/f
            new_files = [join(root, c, f) for f in listdir(join(root, c))]
            # 列表用加号也行
            self.files += new_files
            # len(new_files) 个 i
            self.labels += [i] * len(new_files)  

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        # 标签 是 数字
        label = self.labels[idx]
        f = self.files[idx]
        # npy中返回numpy数组
        events = np.load(f).astype(np.float32)

        if self.augmentation:
            events = random_shift_events(events)
            events = random_flip_events_along_x(events)

        return events, label
