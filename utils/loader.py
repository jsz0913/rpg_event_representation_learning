import torch
import numpy as np

from torch.utils.data.dataloader import default_collate


class Loader:
    # dataset为class flags为class 
    def __init__(self, dataset, flags, device):
        self.device = device
        # 0 到 所有npy名字的个数
        split_indices = list(range(len(dataset))) # 对象必须定义 __len__
        # 注意dataset定义每次取出idx对应
        sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
        # sampler给出索引
        # collate_events 如何合并batch个样本
        self.loader = torch.utils.data.DataLoader(dataset, batch_size =f lags.batch_size, sampler=sampler,
                                             num_workers= flags.num_workers, pin_memory=flags.pin_memory,
                                             collate_fn= collate_events)
    # 凡是可以for循环的，都是Iterable
    # 凡是可以next()的，都是Iterator
    def __iter__(self): # 代表Iterable
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)

def collate_events(data):
    labels = []
    events = []
    # 需要弄清data到底是什么
    # data 是 batchsize 大小的列表
    # 每一个值都是 dataset[index]
    for i, d in enumerate(data):
        labels.append(d[1])
        # d[0] 对应 events N * 4
        # x y t p i  (i代表第几个)
        ev = np.concatenate( [d[0]   , i * np.ones((len(d[0]),1), dtype=np.float32)   ] ,  1)
        events.append(ev)
    # events
    # 去掉列表最外层维度
    # x y t p i
    events = torch.from_numpy(np.concatenate(events,0))
    labels = default_collate(labels)
    return events, labels
