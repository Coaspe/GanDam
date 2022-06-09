from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import copy
import numpy as np


# [12871, 1212, 2702, 425, 353, 12939, 1231, 2590, 448, 355]

class DiabeticRetinopathy(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_csv_path, mode, transform):

        """Initialize and preprocess the dataset."""
        self.image_dir = image_dir
        self.attr_csv_path = attr_csv_path
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.preprocess()
        self.transform = transform

        # 중증 transformation
        transformForSevere = []
        transformForSevere.append(T.RandomVerticalFlip())
        transformForSevere.append(T.RandomResizedCrop(size=(200, 300), scale=(0.8, 1)))
        transformForSevere.append(T.ToTensor())
        self.transformForSevere = T.Compose(transformForSevere)

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the attribute file."""
        # 라벨링 csv 읽어오기
        lines = [line.rstrip() for line in open(self.attr_csv_path, 'r')]
        # 맨 위 이름 제외
        # 테스트 때문에 33까지, 실제는 1:로 사용
        lines = lines[1:]
        # 셔플
        random.seed(1234)
        random.shuffle(lines)
        # 각 라벨이 몇개 추가 되었는지 확인하는 List
        check = np.zeros((10,), dtype=int)
        # 이미지 이름에 라벨링
        for i, line in enumerate(lines):
            split = line.split(",")
            filename = f"{split[0]}.jpeg"
            value = int(split[1])
            # left 0~4 right 5~9 오른쪽이면 +5
            if filename.split("_")[1] == "right.jpeg":
                value += 5

            if check[value] >= 700:
                continue

            # 중증이면 무조건 추가
            if value in [3, 4, 8, 9]:
                self.train_dataset.append([filename, value])
                check[value] += 1
                continue

            # 7,000 개 넘기면 break
            self.train_dataset.append([filename, value])
            check[value] += 1
            if len(self.train_dataset) >= 7000:
                break;

            # Test, Train 데이터 나누기
            # if (i+1) < 10000:
            #     self.train_dataset.append([filename, value])
            # else:
            #     self.test_dataset.append([filename, value])

        print('Finished preprocessing the dataset...')
        print(f"Total train dataset: {len(self.train_dataset)}")

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        # ex) filename: 10_left.jpeg, label = 0
        filename, label = dataset[index]
        # 이미지 열고, transform 해서 label이랑 반환
        image = Image.open(os.path.join(self.image_dir, filename))

        # if 3, 4, 8, 9
        if int(label) in [3, 4, 8, 9]:
            returnedImage = self.transformForSevere(image)
        else:
            returnedImage = self.transform(image)

        return returnedImage, label

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path,
               batch_size=16, mode='train', num_workers=1, image_size=[200, 300]):
    """Build and return a data loader."""
    # Traditional augmentaion
    transform = []

    # # Augmentation
    # if mode == 'train':
    # # numpy image -> tensor image
    transform.append(T.ToTensor())
    transform = T.Compose(transform)

    # 데이터셋
    dataset = DiabeticRetinopathy(image_dir, attr_path, mode, transform)
    # 데이터 로드
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)

    return data_loader