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
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import torchvision.models as models

from data.classification import get_ImageNet_train, get_ImageNet_val
from data.semantic_segmentation import CocoStuff164k
from models.classification import ResNet18
from models.semantic_segmentation import ResNet50Backbone
from utils import change_names
from data.semantic_segmentation import pad_images_and_labels
from utils import mIOU, accuracy

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

    parser.add_argument("--device", default="cuda:0")
    #parser.add_argument("--epochs", type=int, default=15)
    #parser.add_argument("--batch_size", type=int, default=256)
    #parser.add_argument("--clip_grad_norm", type=float, default=1)
    #parser.add_argument("--verbose_every", type=int, default=100, help="print loss and metrics every n batches.")
    parser.add_argument("--save_model_path", default="/content/drive/MyDrive/ML/weights/semantic_segmentation/", help="Dont add .pt, it will be added after epoch number")
    #parser.add_argument("--save_model_every", type=int, default=1000, help="save model weights and optimizer every n batches.")
    args = parser.parse_args()

    if args.task_mode not in ["classification", "semantic_segmentation"]:
        raise ValueError(f"task_model should be classification or semantic_segmentation not {args.task_model}")
    if args.learning_mode not in ["train", "test"]:
        raise ValueError(f"learning_mode should be train or test not {args.task_model}")
    if args.learning_mode == "test" and args.weights is None:
        raise ValueError(f"provide weights to use model in test mode")

    if args.model is None:
        if args.task_mode == "classification":
            args.model = "VT_ResNet18"
        else:
            args.model = "VT_FPN"
    else:
        if args.task_mode == "classification":
            if args.model not in ["ResNet18", "VT_ResNet18"]:
                raise ValueError(f"Only ResNet18 and VT_ResNet18 models supported for classification, not {args.model}")
        else:
            if args.model not in ["PanopticFPN", "VT_FPN"]:
                raise ValueError("") # TODO
    if args.task_mode == "classification":
      args.ignore_index = None
      args.batch_size = 256
      args.lr = 0.1
      args.update_every = 1
      args.weight_decay = 4e-5
      args.nesterov = True
      args.epochs = 15
      args.metric = accuracy
      args.n_classes = 144
      args.verbose_every = 100
    else:
      args.ignore_index = 255 - 92
      args.batch_size = 8
      if args.model == "PanopticFPN":
        args.lr = 0.01
      else:
        args.lr = 0.04
      args.update_every = 2
      args.weight_decay = 1e-5
      args.nesterov = False
      args.epochs = 5
      args.metric = mIOU
      args.n_classes = 91
      args.verbose_every = 2000
    return args


def load_model(args: Namespace) -> nn.Module:
    module = importlib.import_module("models." + args.task_mode)
    model_class = getattr(module, args.model)
    model = model_class(args.n_classes)
    return model


def load_model_and_optimizer(args: Namespace) -> tp.Tuple[nn.Module, optim.SGD]:
    if args.from_pretrained is not None:
        checkpoint = torch.load(args.from_pretrained, map_location=torch.device('cuda:0'))
        args.current_epoch = checkpoint['epoch']
        print(f"Training from epoch number {args.current_epoch}.")
    else:
        print("Training from scratch.")
        args.current_epoch = 0
    
    model = load_model(args)
    if args.from_pretrained is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device=args.device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    #optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    if args.from_pretrained is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, scheduler


def test(model: nn.Module, test_dataloader: DataLoader):
    metrics = []
    with torch.no_grad():
        model.eval()
        for image, label in test_dataloader:
            label = label.to(device=args.device)
            image = image.to(device=args.device)
            logits = model(image)
            metrics.append(args.metric(logits, label))
    print("Mean metric =", np.mean(metrics))


def train(args: Namespace, model: nn.Module, optimizer: optim.SGD, scheduler, train_dataloader: DataLoader, val_dataloader: DataLoader):
    criterion = nn.CrossEntropyLoss(ignore_index=-100 if args.ignore_index is None else args.ignore_index, size_average=True)
    losses = []
    metrics = []
    print("Number of batches in training data", len(train_dataloader))
    for epoch in range(args.current_epoch + 1, args.epochs + 1):
        model.train()
        #print("very first model weights", torch.max(torch.abs(model.classification_head.fc.weight)))
        #start = time.time()
        for i, (images, labels) in enumerate(train_dataloader):
            #start = time.time()
            optimizer.zero_grad()
            images = images.to(device=args.device)
            labels = labels.to(device=args.device)
            logits = model(images)
            loss = criterion(logits, labels) / args.update_every
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            if (i + 1) % args.update_every == 0:
              optimizer.step()
            
            losses.append(loss.item())
            current_metric = args.metric(logits, labels, args.n_classes)
            metrics.append(current_metric)
            #print("one batch took", time.time() - start)
            #print("loss:", loss.item())
            #print("current_metric:", current_metric)
            start = time.time()
            #print("model weights", torch.max(torch.abs(model.classification_head.fc.weight)))
            #print("model grads", torch.max(torch.abs(model.classification_head.fc.weight.grad)))
            if (i % args.verbose_every == 0) or i + 1 == len(train_dataloader):
                print(f"Epoch = {epoch}, batches passed = {i}")
                print("Loss: ", np.mean(losses))
                print("Metric: ", np.mean(metrics))
                print()
                losses = []
                metrics = []
        
        model.eval()
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device=args.device)
                logits = model(images)
                loss = criterion(logits, labels.to(device=args.device)) / args.update_every
                current_metric = args.metric(logits, labels, args.n_classes)
                metrics.append(current_metric)
                losses.append(loss.item())
        scheduler.step()
        print("Val loss: ", np.mean(losses))
        print("Val metric: ", np.mean(metrics))
        print()
        metrics, losses = [], []
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, args.save_model_path + "state_dict_" + args.model + "_" + str(epoch) + ".pt")
    torch.save(model.state_dict(), args.save_model_path + "weights_" + args.model + ".pt")
    

def main(args: Namespace):
    if args.task_mode == "classification":
      train_dataloader = get_ImageNet_train(args)
      test_dataloader = get_ImageNet_val(args)
    else:
      train_dataset = CocoStuff164k(args.data, "train2017", ignore_index=args.ignore_index)
      train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=16,
                                shuffle=True) #collate_fn=pad_images_and_labels(args.ignore_index)
      test_dataset = CocoStuff164k(args.data, "val2017", ignore_index=args.ignore_index)
      test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=16, shuffle=False)
    if args.learning_mode == "test":
        model = load_model(args)
        model.load_state_dict(torch.load(args.weights))
        model.to(device=args.device)
        test(model, test_dataloader)
    else:
        model, optimizer, scheduler = load_model_and_optimizer(args)
        train(args, model, optimizer, scheduler, train_dataloader, test_dataloader)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)

    #pretrained_weights = models.resnet50(pretrained=True).state_dict()
    #pretrained_weights = change_names(pretrained_weights)
    #my_resnet = ResNet50Backbone()
    #my_resnet.load_state_dict(pretrained_weights)

    #state = models.resnet50(pretrained=True).state_dict()
    #my_resnet.load_state_dict(state)
# python main.py classification train
