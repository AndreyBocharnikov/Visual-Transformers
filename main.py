import typing as tp
import os
from argparse import ArgumentParser, Namespace
import importlib
import time
from types import ModuleType

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim

from Datasets import ImageNetDataset


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("task_mode", help="Work in classification or semantic segmentation mode.")
    parser.add_argument("learning_mode", help="To train or get results on test set (one must provide weights).")
    parser.add_argument("data", help="path to the dataset.")
    parser.add_argument("--model", help="Provide ResNet18 or VT_ResNet18 for classification, ... or ... for semantic segmentation.")
    parser.add_argument("--weights", help="In case of learning_mode=Test you need to provide path to trained weights.")
    parser.add_argument("--from_pretrained", help="Path to weights to continue training from saved state dict,"
                                                  "if not provided, will be training from scratch. "
                                                  "Used only if learning_mode=Train")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--verbose_every", type=int, default=500, help="print loss and metrics every n batches.")
    parser.add_argument("--save_model_path", default="./state_dict", help="Dont add .pt, it will be added after epoch number")
    #parser.add_argument("--save_model_every", type=int, default=1000, help="save model weights and optimizer every n batches.")
    args = parser.parse_args()

    if args.task_mode not in ["classification", "semantic_segmentation"]:
        raise ValueError(f"task_model should be classification or semantic_segmentation not {args.task_model}")
    if args.learning_mode == "test" and args.weights is None:
        raise ValueError(f"provide weights to use model in test mode")

    if args.model is None:
        if args.task_mode == "classification":
            args.model = "VT_ResNet18"
        else:
            args.model = "" # TODO
    else:
        if args.task_mode == "classification":
            if args.model not in ["ResNet18", "VT_ResNet18"]:
                raise ValueError(f"Only ResNet18 and VT_ResNet18 models supported for classification, not {args.model}")
        else:
            if args.model not in ["", ""]: # TODO
                raise ValueError("") # TODO
    return args


def load_model(args: Namespace) -> nn.Module:
    module = importlib.import_module("models." + args.task_mode)
    model_class = getattr(module, args.model)
    model = model_class(n_classes=10)
    return model


def load_model_and_optimizer(args: Namespace) -> tp.Tuple[nn.Module, optim.SGD]:
    model = load_model(args)
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=4e-5, nesterov=True)
    if args.from_pretrained is not None:
        checkpoint = torch.load(args.from_pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.current_epoch = checkpoint['epoch']
        print(f"Training from {args.current_epoch} epoch.")
    else:
        print("Training from scratch.")
        args.current_epoch = 0
    return model, optimizer


def test(model: nn.Module, test_dataloader: DataLoader):
    accuracy = []
    with torch.no_grad():
        model.eval()
        for image, label in test_dataloader:
            logits = model(image)  # batch_size should be 1
            accuracy.append((torch.argmax(logits, dim=1) == label).numpy())
    print("Mean accuracy =", np.mean(accuracy))


def train(args: Namespace, model: nn.Module, optimizer: optim.SGD, train_dataloader: DataLoader, val_dataloader: DataLoader):
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracy = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            accuracy.append((torch.argmax(logits, dim=1) == labels).numpy().mean())
            if i % args.verbose_every == 0 or i + 1 == len(train_dataloader):
                print("Train loss: ", np.mean(losses))
                print("Accuracy: ", np.mean(accuracy))
                losses = []
                accuracy = []

        model.eval()
        with torch.no_grad():
            for images, labels in val_dataloader:
                logits = model(images)
                loss = criterion(logits, labels)
                current_accuracy = (torch.argmax(logits, dim=1) == labels).numpy().mean()

                accuracy.append(current_accuracy)
                losses.append(loss.item())

        print("Val loss: ", np.mean(losses))
        print("Val accuracy: ", np.mean(accuracy))
        current_epoch = epoch + args.current_epoch
        torch.save({'epoch': current_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, args.save_model_path + str(current_epoch) + ".pt")
    torch.save(model.state_dict(), "./weights" + args.model + ".pt")


def main(args: Namespace):
    if args.task_mode == "classification":
      train_dataloader = get_ImageNet_train()
      test_dataloader = get_ImageNet_val()
    else:
      train_dataloader = None # TODO
      test_dataloader = None # TODO
    if args.learning_mode == "test":
        model = load_model(args)
        model.load_state_dict(torch.load(args.weights))
        test(model, test_dataloader)
    else:
        model, optimizer = load_model_and_optimizer(args)
        train(args, model, optimizer, train_dataloader, test_dataloader)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)

# python main.py classification train
