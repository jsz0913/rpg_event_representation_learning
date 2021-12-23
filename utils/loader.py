import torch
import numpy as np

from torch.utils.data.dataloader import default_collate


class Loader:
    # dataset为class flags为class 
    def __init__(self, dataset, flags, device):
        self.device = device
        # 0 到 所有npy名字
        split_indices = list(range(len(dataset))) # 对象必须定义 __len__
        # 
        sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
        
        
        self.loader = torch.utils.data.DataLoader(dataset, batch_size=flags.batch_size, sampler=sampler,
                                             num_workers=flags.num_workers, pin_memory=flags.pin_memory,
                                             collate_fn=collate_events)

    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)


def collate_events(data):
    labels = []
    events = []
    for i, d in enumerate(data):
        labels.append(d[1])
        ev = np.concatenate([d[0], i * np.ones((len(d[0]),1), dtype=np.float32)],1)
        events.append(ev)
    events = torch.from_numpy(np.concatenate(events,0))
    labels = default_collate(labels)
    return events, labels
