import argparse
import math
import os
import sys
import json
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from PIL import Image
import warnings

from models import models, Unet

warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = True

def configure_optimizers(model, args):
    """Configures optimizers based on the new ResDiff structure."""
    module = model.module if isinstance(model, nn.DataParallel) else model
    vae_params, cfm_params, aux_params = module.get_trainable_parameters()

    print(f"Found {len(vae_params)} parameters for VAE optimizer.")
    optimizer_vae = optim.Adam(vae_params, lr=args.learning_rate)
    
    optimizer_cfm = None
    if cfm_params:
        print(f"Found {len(cfm_params)} parameters for CFM optimizer.")
        optimizer_cfm = optim.Adam(cfm_params, lr=args.learning_rate_cfm)

    aux_optimizer = None
    if aux_params:
        print("Found auxiliary parameters for aux_optimizer.")
        aux_optimizer = optim.Adam(aux_params, lr=args.aux_learning_rate)

    return optimizer_vae, optimizer_cfm, aux_optimizer


class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2, beta=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.beta = beta

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target) * 255**2
        
        if "CFMLoss" in output and output["CFMLoss"] is not None:
            out["cfm_loss"] = output["CFMLoss"]
        else:
            out["cfm_loss"] = torch.tensor(0.0, device=target.device)

        out["rd_loss"] = self.lmbda  * out["mse_loss"] + out["bpp_loss"]
        out["loss"] = out["rd_loss"] + self.beta * out["cfm_loss"]
        
        return out


class ImageDataset(data.Dataset):
    def __init__(self, path_dir, transform=None):
        self.path_dir = path_dir
        self.transform = transform
        self.images = [f for f in os.listdir(path_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.path_dir, self.images[index])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        torch.save({"epoch":state['epoch'], "state_dict":state["state_dict"]}, os.path.join(os.path.dirname(filename), "best_model.tar"))



def train_one_epoch(model, criterion, train_dataloader, optimizers, epoch, train_step, cfm_accumulation_steps=1,tb_writer=None, clip_max_norm=None):
    model.train()
    device = next(model.parameters()).device
    
    optimizer_vae, optimizer_cfm, aux_optimizer = optimizers
    accumulation_steps = cfm_accumulation_steps

    module = model.module if isinstance(model, nn.DataParallel) else model
    vae_params, cfm_params, aux_params = module.get_trainable_parameters()

    for i, x in enumerate(train_dataloader):
        x = x.to(device)

        optimizer_vae.zero_grad()
        if optimizer_cfm and i % accumulation_steps == 0:
            optimizer_cfm.zero_grad()
        if aux_optimizer:
            aux_optimizer.zero_grad()
        
        out = model(x, epoch=epoch)
        out_criterion = criterion(out, x)
        loss = out_criterion["loss"]

        loss = loss / accumulation_steps
        loss.backward()
        
        if clip_max_norm and clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(vae_params, clip_max_norm)
            if aux_optimizer:
                torch.nn.utils.clip_grad_norm_(aux_params, clip_max_norm)
        
        optimizer_vae.step()
        if optimizer_cfm and i % accumulation_steps == 0:
            if clip_max_norm and clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(cfm_params, clip_max_norm)
            optimizer_cfm.step()
        
        if aux_optimizer:
            aux_optimizer.step()

        train_step += 1
        if tb_writer and i % 10 == 0:
            tb_writer.add_scalar('train/loss', out_criterion["loss"].item(), train_step)
            tb_writer.add_scalar('train/rd_loss', out_criterion["rd_loss"].item(), train_step)
            tb_writer.add_scalar('train/mse_loss', out_criterion["mse_loss"].item(), train_step)
            tb_writer.add_scalar('train/bpp_loss', out_criterion["bpp_loss"].item(), train_step)
            tb_writer.add_scalar('train/cfm_loss', out_criterion["cfm_loss"].item(), train_step)
    
    return train_step

def eval_epoch(model, criterion, eval_dataloader, epoch, tb_writer=None):

    model.eval()
    device = next(model.parameters()).device
    
    total_rd_loss = 0.0
    total_mse_loss = 0.0
    total_bpp_loss = 0.0
    total_samples = 0

    print(f"Running evaluation for epoch {epoch}...")
    for x in eval_dataloader:
        x = x.to(device).contiguous()
        batch_size = x.shape[0]
        out = model(x)
        out_criterion = criterion(out, x)

        total_rd_loss += out_criterion["rd_loss"].mean().item() * batch_size
        total_mse_loss += out_criterion["mse_loss"].mean().item() * batch_size
        total_bpp_loss += out_criterion["bpp_loss"].mean().item() * batch_size
        total_samples += batch_size

    avg_rd_loss = total_rd_loss / total_samples
    avg_mse = total_mse_loss / total_samples
    avg_bpp = total_bpp_loss / total_samples
    psnr = 10 * math.log10(255**2 / avg_mse)

    print(f"Epoch(Eval) {epoch}: PSNR: {psnr:.4f} dB | BPP: {avg_bpp:.6f} | RD Loss: {avg_rd_loss:.6f}")

    if tb_writer:
        tb_writer.add_scalar('eval/rd_loss', avg_rd_loss, epoch)
        tb_writer.add_scalar('eval/bpp_loss', avg_bpp, epoch)
        tb_writer.add_scalar('eval/mse_loss', avg_mse, epoch)
        tb_writer.add_scalar('eval/psnr', psnr, epoch)
        
    return avg_rd_loss

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Advanced Training Script for ResDiff with CFM.")
    parser.add_argument("--train_data", type=str, default='../dataset/open-images-v6/train/data/', required=True)
    parser.add_argument("--eval_data", type=str, default='../dataset/open-images-v6/test/data/', required=True)
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate for VAE.")
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float, help="Auxiliary loss learning rate.")
    parser.add_argument("--learning-rate-cfm", default=1e-5, type=float, help="Learning rate for CFM Model.")
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.0067)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--clip_max_norm", default=1.0, type=float)
    parser.add_argument("--save_path", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("-n", "--num-workers", type=int, default=16, help="Dataloaders threads.")
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--M", type=int, default=192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.05, help="Weight for the CFM loss.")
    parser.add_argument("--unet-dim", type=int, default=64)
    parser.add_argument("--use_norm", action="store_true", default=True, help="Norm Residual and Conditional Information")
    # CFM specific arguments
    parser.add_argument("--cfm-sigma", type=float, default=0.1, help="Sigma for Conditional Flow Matching noise.")
    parser.add_argument("--cfm-sampling-steps", type=int, default=50, help="Number of steps for ODE solver during sampling.")
    parser.add_argument(
        "--cfm-matcher", 
        type=str, 
        default="base", 
        choices=['base', 'ot', 'sb', 'vp', 'target', 'mean'],
        help="Type of Conditional Flow Matcher to use: base (Independent), ot (Optimal Transport), sb (Schrodinger Bridge), vp (Variance Preserving), mean (MeanFlow)."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="relic-elic", 
        choices=list(models.keys()),
        help="Type of ReLIC models."
    )
    parser.add_argument(
        "--cfm-accumulation-steps",
        type=int, 
        default=1,
        help="Number of steps to accumulate gradients for the CFM model before an optimizer update."
    )
    parser.add_argument("--lazy-init", action="store_true", help="Enable lazy initialization of CFM components.")
    parser.add_argument("--init-threshold", type=int, default=10, help="Epoch to start training CFM components in lazy mode.")

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    print(args)
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    eval_transforms = transforms.Compose([transforms.CenterCrop(args.patch_size), transforms.ToTensor()])
    train_dataset = ImageDataset(args.train_data, transform=train_transforms)
    eval_dataset = ImageDataset(args.eval_data, transform=eval_transforms)

    args_dict = vars(args)
    args_dict['Train_length'] = len(train_dataset)
    args_dict['Eval_length'] = len(eval_dataset)
    config_filepath = os.path.join(args.save_path, "run_parameters.json")
    with open(config_filepath, 'w') as f:
        json.dump(args_dict, f, indent=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define model kwargs from args
    unet_kwargs = {'dim': args.unet_dim, 'channels': args.M * 3, 'out_dim': args.M}
    cfm_wrapper_kwargs = {
        'matcher_type': args.cfm_matcher,
        'sigma': args.cfm_sigma, 
        'num_sampling_steps': args.cfm_sampling_steps
    }
    
    net = models[args.model](
        N=args.N, M=args.M,
        lazy_init=args.lazy_init,
        init_threshold=args.init_threshold,
        cfm_model_class=Unet,
        cfm_model_kwargs=unet_kwargs,
        cfm_wrapper_kwargs=cfm_wrapper_kwargs,
        use_norm=args.use_norm
    )
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                       shuffle=True, pin_memory=(device == "cuda"))
    eval_dataloader = data.DataLoader(eval_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers,
                                      shuffle=False, pin_memory=(device == "cuda"))
    optimizer_vae, optimizer_cfm, aux_optimizer = configure_optimizers(net, args)
    optimizers = (optimizer_vae, optimizer_cfm, aux_optimizer)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_vae, milestones=[40, 80], gamma=0.2)
    criterion = RateDistortionLoss(lmbda=args.lmbda, beta=args.beta)
    tb_writer = SummaryWriter(args.log_dir)

    train_step = 0
    last_epoch = 0
    if args.checkpoint:
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer_vae.load_state_dict(checkpoint["optimizer_vae"])
        optimizer_cfm.load_state_dict(checkpoint["optimizer_cfm"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if 'step' in checkpoint:
            train_step = checkpoint["step"]

    best_loss = float("inf")

    for epoch in range(last_epoch, args.epochs):
        
        diff_lr = optimizer_cfm.param_groups[0]['lr'] if optimizer_cfm else 0.0
        print(f"--- Epoch {epoch} | VAE LR: {optimizer_vae.param_groups[0]['lr']:.2e} | Diff LR: {diff_lr:.2e} | Batch: {train_dataloader.batch_size} ---")

        train_step = train_one_epoch(
            net, criterion, train_dataloader, optimizers, epoch, train_step, cfm_accumulation_steps=args.cfm_accumulation_steps,
            tb_writer=tb_writer, clip_max_norm=args.clip_max_norm
        )
        
        current_loss = eval_epoch(net, criterion, eval_dataloader, epoch, tb_writer)
        
        lr_scheduler.step()
        
        is_best = current_loss < best_loss
        best_loss = min(current_loss, best_loss)

        latest_checkpoint_dict = {
            "epoch": epoch, "state_dict": net.state_dict(), "loss": best_loss,
            "optimizer_vae": optimizer_vae.state_dict(), "optimizer_cfm": optimizer_cfm.state_dict(),
            "aux_optimizer": aux_optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(), "step": train_step,
        }
        save_checkpoint(
            latest_checkpoint_dict, is_best,
            os.path.join(args.save_path, f"ckp{int(args.lmbda * 10000)}.tar"),
        )

    tb_writer.close()

if __name__ == "__main__":
    main(sys.argv[1:])