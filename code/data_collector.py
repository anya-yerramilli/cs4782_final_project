import learn2learn as l2l
from torchvision import transforms
import os

# learn2learn is a library for few-shot learning tasks
# it will pre-split the dataset to have 64 base (train) classes, 16 validation classes,
# and 20 novel (test) classes.


def get_datasets(root="../code/data"):
    data_path = os.path.join(os.path.dirname(__file__), root)
    data_path = os.path.abspath(data_path)

    transform = transforms.Compose([transforms.Resize(84), transforms.CenterCrop(84)])

    train = l2l.vision.datasets.MiniImagenet(
        root=data_path, mode="train", transform=transform
    )
    val = l2l.vision.datasets.MiniImagenet(
        root=data_path, mode="validation", transform=transform
    )
    test = l2l.vision.datasets.MiniImagenet(
        root=data_path, mode="test", transform=transform
    )

    return train, val, test


# NOTE: example on what the dataset looks like:
# img, label = train_dataset[0]
# print(img.shape)   # torch.Size([3, 84, 84])
# print(label)       # int (e.g., 0 to 63 in train, 0 to 15 in val, etc.)
