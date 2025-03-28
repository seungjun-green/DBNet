from data.GD import GD
from data.GS import GS
from data.dataset import SynthTextDataset
from torch.utils.data import DataLoader, random_split


# return train_dataloader and val_dataloader
def create_dataloaders(batch_size, val_split, mat_file, images_dir):
    generate_gs = GS(min_text_size=8, shrink_ratio=0.4)
    generate_gd = GD(dil_ratio==0.4, thresh_min=0.3, thresh_max=0.7)
    
    dataset = SynthTextDataset(
        mat_file=mat_file,
        images_dir=images_dir,
        gt_prob_binary=generate_gs,
        gt_threshold=generate_gd,
        target_size=(640, 640) # during training use 640X640 size as shown in the paper
    )
    
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # use batch size of 16 for A100 40GB
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_dataloader, val_dataloader