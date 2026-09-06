import argparse
import torch
import torch.nn.functional as F
import logging
import tqdm
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import LeNet5


logger = logging.getLogger(Path(__file__).name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for the LeNet model trained with DDP.")
    parser.add_argument("epochs", help="Which epoch to load model weights.")
    args = parser.parse_args()

    # Which epoch to load
    epoch = int(args.epochs)

    # Initialize the logger for the command line output
    logging.basicConfig(
        level=logging.INFO,
        format=f'[%(asctime)s] [%(name)-15s] [%(levelname)8s]: %(message)s',
        datefmt='%d.%m.%y %H:%M:%S'
    )

    # Inference decoding will be done on the cpu
    device = "cpu"

    # Load the model weight from the checkpoints folder
    checkpoints_dir = Path("./checkpoints")
    model = LeNet5(n_classes=10)
    model.load_state_dict(torch.load(checkpoints_dir / f"epoch={epoch:03d}.pth", weights_only=True))

    # Define the test dataset for the mnist image files
    test_dataset = datasets.MNIST(
        root="mnist_data",
        train=False,
        transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
        download=False
    )

    # Set up the test dataloader
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Run the model in inference mode on the test data
    model.eval()
    correct = 0

    with torch.no_grad():
        for x_test, y_test in tqdm.tqdm(test_loader, desc="Evaluating"):
            # Move unseen data to target device
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            # Forward pass and obtain predicted class
            logits = model(x_test)
            values = F.softmax(logits, dim=1).argmax(dim=1)
            correct += (values == y_test).sum().item()

    score = correct / len(test_dataset)
    logger.info(f"Final accuracy score on unseen data after {epoch} epochs: {score * 100:.2f}%")
