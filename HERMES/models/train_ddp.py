import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import *
from model import *
import os
from torchvision import transforms

import matplotlib.pyplot as plt
plt.style.use('/glade/u/home/aidenz/Documents/repos/EAS6995/az-paper-twocol.mplstyle')
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)
def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        tensorboard_path: str = "tensorboard_logs",
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"]) # torchrun provided gpu_id/rank
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if self.gpu_id == 0:
            self.logger = SummaryWriter(tensorboard_path)
            self.wl = np.loadtxt('/glade/derecho/scratch/aidenz/data/HERMES_labels/wl_grid.txt')
        else:
            self.logger = None
            self.wl = None
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.spec_out_path = os.path.join(os.environ['scratch_dir'], 'data/HERMES_logs/spec_out/')
        # Check if spec_out_path exists, if not create it
        if not os.path.exists(self.spec_out_path):
            os.makedirs(self.spec_out_path)

        self.normalization_factor = 1e-3


        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _log(self, msg):
        if self.gpu_id == 0:
            print(msg)

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
        # Since epochs_run was the epoch that was completed, we want to start from the next epoch
        self.epochs_run = self.epochs_run + 1 

    def _run_batch(self, source, spectra, rtop, validate=False):
        # torch.cuda.empty_cache()
        self.optimizer.zero_grad()
        output = self.model(source) # Last element is the R_top, which is only for loss
        
        loss_MSE = F.mse_loss(output, spectra)
        loss_delta = L_phys_delta(self.model, source, rtop, gamma=1e-3, norm_factor=self.normalization_factor)
        loss_chem = L_phys_chem(self.model, source, gamma=1e4)
        # Add loss terms and add machine precision to avoid NaN
        loss = loss_MSE + loss_delta + loss_chem + torch.finfo(torch.float32).eps
        if loss.isnan().any():
            print("NaN loss detected")
            print("Loss MSE:", loss_MSE.item())
            print("Loss delta:", loss_delta.item())
            print("Loss chem:", loss_chem.item())
        

        loss.backward()
        self.optimizer.step()
        # if self.gpu_id == 0:
        #     # Print loss items
        #     print('Batch size:', len(source),"| Loss MSE:", loss_MSE.item(), "| Loss delta:", loss_delta.item(), "| Loss chem:", loss_chem.item())
        #     print('Total Loss:', loss.item())
            # self._plot_spectra(self.model, epoch=1)
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))['parameters'])
        self.train_data.sampler.set_epoch(epoch)
        loss = 0
        for samples in self.train_data:
            spectra = samples['spectra'].to(self.gpu_id)
            parameters = samples['parameters'].to(self.gpu_id)
            rtop = samples['rtop'].to(self.gpu_id)
            loss = loss + self._run_batch(parameters, spectra, rtop)
        train_loss = loss / len(self.train_data.dataset)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | Loss: {train_loss:.4f}")
        
        dist.barrier()
        self.model.eval()
        val_loss = self._validate_epoch()
        dist.barrier()
        val_loss = torch.Tensor([val_loss]).to(self.gpu_id)
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)

        train_loss = torch.Tensor([train_loss]).to(self.gpu_id)
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)

        if self.gpu_id == 0:
            self.logger.add_scalar("Loss/val", val_loss/int(os.environ["WORLD_SIZE"]), epoch)
            self.logger.add_scalar("Loss/train", train_loss/int(os.environ["WORLD_SIZE"]), epoch)
            self.logger.flush()
            self._save_spectra(self.model, epoch)
            self._plot_spectra(self.model, epoch)
        self.model.train()

    def _validate_epoch(self):
        val_loss = 0
        for samples in self.val_data:
            spectra = samples['spectra'].to(self.gpu_id)
            parameters = samples['parameters'].to(self.gpu_id)
            rtop = samples['rtop'].to(self.gpu_id)
            val_loss = val_loss + self._validate_batch(parameters, spectra, rtop)
            
        return val_loss / len(self.val_data.dataset)

    def _validate_batch(self, val_source, val_spectra, rtop):
        with torch.no_grad():
            output = self.model(val_source)
            loss_MSE = F.mse_loss(output, val_spectra)
            loss_delta = L_phys_delta(self.model, val_source, rtop, gamma=1e-3, norm_factor=self.normalization_factor)
            # loss_chem = L_phys_chem(self.model, val_source, gamma=1.e4)
            loss = loss_MSE + loss_delta # + loss_chem
            return loss.item()

    def _plot_spectra(self, net, epoch):
        if self.gpu_id != 0:
            return
        with torch.no_grad():
            earth_params = torch.tensor([
                        5772, 1, 4.4, 0, 1, 1, 255, -5.8, -1.9, -0.6, -7.6, -3.5 ,-6.5 ]).to(self.gpu_id)
            earth_params = earth_params.unsqueeze(0)
            output = net(earth_params)
            output = output.cpu().numpy()
            plt.plot(self.wl, output[0], label='Model Output')
            plt.xlabel('Wavelength (um)')
            plt.ylabel('Transit Depth')
            plt.title(f"Epoch {epoch} | Model Output")
            plt.savefig(os.path.join(self.spec_out_path, f"epoch_{epoch}.png"))
            plt.close()

    def _save_spectra(self, net, epoch):
        if self.gpu_id != 0:
            return
        with torch.no_grad():
            earth_params = torch.tensor([
                        5772, 1, 4.4, 0, 1, 1, 255, -5.8, -1.9, -0.6, -7.6, -3.5 ,-6.5 ]).to(self.gpu_id)
            earth_params = earth_params.unsqueeze(0)
            output = net(earth_params)
            output = output.cpu().numpy()
            np.savetxt(os.path.join(self.spec_out_path, f"epoch_{epoch}.txt"), output[0], fmt='%.6e')


    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._log(f'Starting epoch {epoch}')
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            



def load_train_objs():
    data_dir = '/glade/derecho/scratch/aidenz/data/HERMES_data/'
    label_csv_path = '/glade/derecho/scratch/aidenz/data/HERMES_labels/sampled_parameters.csv'
    dataset = spectraDataset(label_csv_path, data_dir, transform=transforms.Compose([
        ToTensor(),
        Normalize()
    ]))
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(13))
    model = SpecGen(in_channels=13, out_channels=44228)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    return train_dataset, val_dataset, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt", tensorboard_path: str = "tensorboard_logs"):
    ddp_setup()
    train_dataset, val_dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_dataset, batch_size)
    val_data = prepare_dataloader(val_dataset, batch_size)
    trainer = Trainer(model, train_data, val_data, optimizer, save_every, snapshot_path, tensorboard_path)
    trainer.train(total_epochs)
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('run_name', type=str, help='Name of the run')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=1, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=256, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    # Get scratch dir
    scratch_dir = os.path.join(os.environ['scratch_dir'], 'data/HERMES_logs', args.run_name)

    main(args.save_every, args.total_epochs, args.batch_size, \
        snapshot_path=os.path.join(scratch_dir, 'snapshot.pt'), tensorboard_path=os.path.join(scratch_dir, 'tensorboard_logs'))