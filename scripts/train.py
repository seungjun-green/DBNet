import torch
import torch.nn.functional as F
from models.model_builder import DBNet
from data.dataloader import create_dataloaders
from tqdm.notebook import tqdm
import os

def calculate_loss(outputs, batch, alpha=1.0, beta=10.0):
    # unpack the model outputs
    P = outputs['probability_map'] # (N, 1, H, W)
    B_hat = outputs['binary_map'] # (N, 1, H, W)
    T = outputs['threshold_map'] # (N, 1, H, W)

    # unpack the gt(batch)
    g_s = batch['g_s'] # (N, 1, H, W)
    g_s_mask = batch['g_s_mask'] # (N, H, W)
    g_d = batch['g_d'] # (N, H, W)
    g_d_mask = batch['g_d_mask'] # (N, H, W)
    g_s_mask = g_s_mask.unsqueeze(1)  # (N, 1, H, W)
    g_d_mask = g_d_mask.unsqueeze(1)  # (N, 1, H, W)

    # BCE loss for the probability map
    loss_prob = F.binary_cross_entropy(P, g_s, reduction='none')
    loss_prob = (loss_prob * g_s_mask).sum() / (g_s_mask.sum() + 1e-6)

    # BCE loss for the approximate binary map
    loss_binary = F.binary_cross_entropy(B_hat, g_s, reduction='none')
    loss_binary = (loss_binary * g_s_mask).sum() / (g_s_mask.sum() + 1e-6)

    # L1 loss for the threshold map
    loss_thresh = F.l1_loss(T, g_d.unsqueeze(1), reduction='none')
    loss_thresh = (loss_thresh * g_d_mask).sum() / (g_d_mask.sum() + 1e-6)

    # combine the losses using weighting factor
    total_loss = loss_prob + alpha * loss_binary + beta * loss_thresh
    
    return total_loss
  

class Trainer:
  def __init__(self, backbone, batch_size, val_split, epoch_num, max_confidence, log_per_epoch, mat_file, images_dir, device, checkpoint_path):
    self.batch_size = batch_size
    self.val_split = val_split
    
    self.train_dataloader, self.val_loader = create_dataloaders(batch_size, val_split, mat_file, images_dir)
    self.model = DBNet(backbone=backbone, input_channels=[256, 512, 1024, 2048], k=50, adaptive=True, serial=True, bias=False)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
    self.epoch_num = epoch_num
    self.max_confidence = max_confidence
    self.loss_fn = calculate_loss
    self.log_per_epoch = log_per_epoch
    self.log_step = len(self.train_dataloader) // self.log_per_epoch
    self.device = device
    self.checkpoint_path = checkpoint_path
    
    self.model = self.model.to(device)
    

  def save_checkpoint(self, checkpoint_folder, epoch, step_count, val_loss):
    
    sub_path = f"epoch_{epoch}_step_{step_count}_loss{val_loss}.pth"
    file_path = os.path.join(checkpoint_folder, sub_path)
    
    checkpoint = {
      'epoch': epoch,
      'step_count': step_count,
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict() 
    }
    
    torch.save(checkpoint, file_path)
    tqdm.write(f"Saved checkpoint to {file_path}")
  
  def eval(self):
    self.model.eval()
    total_val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
      for batch in self.val_loader:
        images = batch['image'].to(self.device)
        batch['g_s'] = batch['g_s'].to(self.device)
        batch['g_s_mask'] = batch['g_s_mask'].to(self.device)
        batch['g_d'] = batch['g_d'].to(self.device)
        batch['g_d_mask'] = batch['g_d_mask'].to(self.device)

        outputs = self.model(images, True)
        loss = self.loss_fn(outputs, batch, alpha=1.0, beta=10.0)
        
        total_val_loss += loss.item()
        num_batches += 1

    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
    self.model.train()
    
    return round(avg_val_loss, 4)
  
  def train(self):
    self.model.train()
    for epoch in range(self.epoch_num):
      progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epoch_num}")
      for step_count, batch in enumerate(progress_bar):
        images = batch['image'].to(self.device)
        batch['g_s'] = batch['g_s'].to(self.device)
        batch['g_s_mask'] = batch['g_s_mask'].to(self.device)
        batch['g_d'] = batch['g_d'].to(self.device)
        batch['g_d_mask'] = batch['g_d_mask'].to(self.device)
        
        outputs = self.model(images, True)
        # use 1 for alpha and 10 for beta as shown in the paper
        loss = self.loss_fn(outputs, batch, alpha=1.0, beta=10.0)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        progress_bar.set_postfix(loss=loss.item())
        
        # log step
        if step_count % self.log_step == 0 or step_count == len(self.train_dataloader) - 1:
          val_loss = self.eval()
          self.save_checkpoint(self.checkpoint_path, epoch, step_count, val_loss)