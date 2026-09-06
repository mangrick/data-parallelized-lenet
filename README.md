# Training LeNet-5 with distributed data parallel

This project demonstrates deep neural network training with distributed data parallelism (DDP) using PyTorch's distributed library.
Similar to the tensor parallelism project, a standard LeNet-5 is used as the network architecture. 
The project shows how to initialize the process group and how to wrap the model in a distributed data container for synchronizing the gradients across ranks.
Moreover, the project focuses on using distributed data sampling, so that each model replica receives a different part of the minibatch to compute and synchronize gradients. 

## Obtaining the data
The MNIST dataset can be downloaded through torchvision using the following command (both training and test data will be downloaded).
Both train and evaluation scripts assume that the data has already been downloaded.
```bash
python -c "from torchvision import datasets; datasets.MNIST(root='mnist_data', train=True, download=True); datasets.MNIST(root='mnist_data', train=False, download=True)"
```

## Training the model
The following command shows how to start the training process with torchrun. 
The training script has a command line argument for the number of training epochs. 
All internal weights from the model on rank 0 will be stored in a checkpoint directory.

```bash
OMP_NUM_THREADS=2 torchrun --nproc_per_node=3 train.py --n_epochs 3
```

## Evaluation on test data
The evaluation script requires that a model has already been trained for a given epoch, since the script will use those weights.
Model evaluation can be done using the command below. 
In contrast to the training script, inference will only be conducted on a single process without requiring any distributed communications.
```bash
python eval.py 3
```
