def get_ImageNet_train(args):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dir = os.path.join(args.data, 'train')
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    return DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)


def get_ImageNet_val():
    val_dir = os.path.join(args.data, "val")
    val_dataset = datasets.ImageFolder(val_dir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    return DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)