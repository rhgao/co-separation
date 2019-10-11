import torch.utils.data
from data.base_data_loader import BaseDataLoader
from utils.utils import object_collate

def CreateDataset(opt):
    dataset = None
    if opt.model == 'audioVisualMUSIC':
        from data.audioVisual_dataset import AudioVisualMUSICDataset
        dataset = AudioVisualMUSICDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.model)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        if opt.mode == "train":
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=False,
                num_workers=int(opt.nThreads),
                collate_fn=object_collate)
        elif opt.mode == 'val':
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=False,
                num_workers=2,
                collate_fn=object_collate)          

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
