import argparse
import os
import math
import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import logging
from dataclasses import dataclass
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from model import LeNet5


logger = logging.getLogger(Path(__file__).name)


@dataclass
class WorldInfo:
    """
    Store information about the current world environment for the distributed training run.
    """
    rank: int
    local_rank: int
    world_size: int

    @classmethod
    def from_env(cls) -> "WorldInfo":
        return cls(
            rank=int(os.environ["RANK"]),
            local_rank=int(os.environ["LOCAL_RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )


class TrainingProcedure:
    """
    The training procedure for training LeNet-5 using Distributed Data Parallel (DDP).
    """
    def __init__(self, info: WorldInfo, n_epochs: int, device: str = "cpu"):
        self.info = info
        self.n_epochs = n_epochs
        self.device = device

        # Set up the process group for all ranks
        dist.init_process_group(backend="gloo", rank=self.info.rank, world_size=self.info.world_size)

        model = LeNet5(n_classes=10).to(device)
        model.reset_parameters()
        self.model = DDP(model)

    def train(self):
        """
        Training routine including dataset and dataloader setup.
        """
        # Define train validation and test datasets for the mnist image files
        training_data = datasets.MNIST(
            root="mnist_data",
            train=True,
            transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
            download=False
        )

        # Use 10% of the training data as a held-out validation set
        indices = torch.arange(len(training_data))
        split = math.floor(len(training_data) * 0.1)

        # Ensure that every rank gets the same batch of data in each iteration by using a constant seed
        train_indices, valid_indices = indices[split:], indices[:split]
        train_dataset = Subset(training_data, train_indices)
        valid_dataset = Subset(training_data, valid_indices)

        train_sampler = DistributedSampler(
            dataset=train_dataset,
            num_replicas=self.info.world_size,
            rank=self.info.rank,
            shuffle=True,
            drop_last=True
        )

        valid_sampler = DistributedSampler(
            dataset=valid_dataset,
            num_replicas=self.info.world_size,
            rank=self.info.rank,
            shuffle=False,
            drop_last=True
        )

        # Setup training and validation data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, sampler=train_sampler, num_workers=2)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=32, sampler=valid_sampler, num_workers=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, foreach=False)

        # Train the model
        self._fit(
            train_loader=train_loader,
            valid_loader=valid_loader,
            train_sampler=train_sampler,
            valid_sampler=valid_sampler,
            loss_function=criterion,
            optimizer=optimizer,
            nb_epochs=self.n_epochs,
            device=self.device,
            primary_node=self.info.rank == 0
        )

    def _fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        train_sampler: DistributedSampler,
        valid_sampler: DistributedSampler,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        nb_epochs: int = 100,
        device: str = "cpu",
        primary_node: bool = False
    ) -> None:
        """
        Training procedure for the LeNet architecture.
        """
        for i in range(1, nb_epochs + 1):
            train_running_loss = 0
            valid_running_loss = 0
            sample_counter = 0
            train_sampler.set_epoch(i)
            valid_sampler.set_epoch(i)

            # Only print progress bar on the primary rank
            with tqdm.tqdm(total=len(train_loader) + len(valid_loader), disable=not primary_node) as pbar:
                # Perform training iteration (1 Epoch)
                self.model.train()
                for x_train, y_train in train_loader:
                    # Move training data to target device
                    x_train = x_train.to(device)
                    y_train = y_train.to(device)

                    # Clear gradient fields
                    optimizer.zero_grad()

                    # Forward pass
                    logits = self.model(x_train)
                    loss = loss_function(logits, y_train)

                    train_running_loss += loss.item() * x_train.size(0)
                    sample_counter += x_train.size(0)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    # Update description at each update step
                    pbar.set_description(f"Epoch {i}: Train loss: {train_running_loss / sample_counter:.04f} "
                                         f"-- Validation loss: ......")
                    pbar.update(1)

                # Evaluate on validation data (After 1 epoch)
                train_running_loss = train_running_loss / sample_counter
                self.model.eval()
                sample_counter = 0
                with torch.no_grad():
                    for x_val, y_val in valid_loader:
                        # Move validation data to target device
                        x_val = x_val.to(device)
                        y_val = y_val.to(device)

                        # Forward pass and record loss
                        logits = self.model(x_val)
                        loss = loss_function(logits, y_val)
                        valid_running_loss += loss.item() * x_val.size(0)
                        sample_counter += x_val.size(0)

                        # Update description at each validation step
                        pbar.set_description(f"Epoch {i}: Train loss: {train_running_loss:.04f} "
                                             f"-- Validation loss: {valid_running_loss / sample_counter:.04f}")
                        pbar.update(1)

    def get_model(self) -> nn.Module:
        """
        Return the LeNet-5 model without the DDP wrapper.
        """
        return self.model.module

    def __del__(self):
        # Cleanup the process group
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for the LeNet model using DDP.")
    parser.add_argument("--n_epochs", help="Number of epochs to train.", default=3)
    args = parser.parse_args()

    # Define number of epochs for training:
    n_epochs = int(args.n_epochs)

    # Obtain values for world size, rank and local rank
    info = WorldInfo.from_env()

    # Initialize the logger for the command line output
    logging.basicConfig(
        level=logging.INFO,
        format=f'[%(asctime)s] [%(name)-15s] [RANK: {info.rank}] [%(levelname)8s]: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S'
    )

    # Train the LeNet-5 model using DDP
    procedure = TrainingProcedure(info, n_epochs=n_epochs)
    procedure.train()
    if info.rank == 0:
        trained_model = procedure.get_model()
        checkpoint_dir = Path("./checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(trained_model.state_dict(), checkpoint_dir / f"epoch={n_epochs:03d}.pth")
