import learn2learn as l2l
from torchvision import transforms
import os
import pandas as pd

# learn2learn is a library for few-shot learning tasks
# it will pre-split the dataset to have 64 base (train) classes, 16 validation classes,
# and 20 novel (test) classes.


def get_datasets(root="../code/data"):
    data_path = os.path.join(os.path.dirname(__file__), root)
    data_path = os.path.abspath(data_path)

    train_tf = transforms.Compose(
        [transforms.RandomCrop(84, padding=8), transforms.RandomHorizontalFlip()]
    )
    eval_tf = transforms.Compose([transforms.Resize(84), transforms.CenterCrop(84)])

    train = l2l.vision.datasets.MiniImagenet(
        root=root, mode="train", download=True, transform=train_tf
    )
    val = l2l.vision.datasets.MiniImagenet(
        root=root, mode="validation", transform=eval_tf
    )
    test = l2l.vision.datasets.MiniImagenet(root=root, mode="test", transform=eval_tf)

    # for ds, mode in [(train, "train"), (val, "validation"), (test, "test")]:
    #     csv = os.path.join(root, "miniImageNet", f"{mode}.csv")
    #     ds.labels = pd.read_csv(csv)["label"].tolist()

    return train, val, test


# NOTE: example on what the dataset looks like:
# img, label = train_dataset[0]
# print(img.shape)   # torch.Size([3, 84, 84])
# print(label)       # int (e.g., 0 to 63 in train, 0 to 15 in val, etc.)
