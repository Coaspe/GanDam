from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


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
        lines = lines[1:33]
        # 셔플
        random.seed(1234)
        random.shuffle(lines)

        # 이미지 이름에 라벨링
        for i, line in enumerate(lines):
            split = line.split(",")
            filename = f"{split[0]}.jpeg"
            values = int(split[1])
            # left 0~4 right 5~9 오른쪽이면 +5
            if filename.split("_")[1] == "right.jpeg":
                values += 5

            # Test, Train 데이터 나누기
            if (i+1) < 5000:
                self.train_dataset.append([filename, values])
            else:
                self.test_dataset.append([filename, values])

        print('Finished preprocessing the dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        # ex) filename: 10_left.jpeg, label = 0
        filename, label = dataset[index]
        # 이미지 열고, transform 해서 label이랑 반환
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), label

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
    #     # 상하 반전
    #     transform.append(T.RandomVerticalFlip())
    # Resize 높이 200 너비 300
    # transform.append(T.Resize(image_size))
    # # numpy image -> tensor image
    transform.append(T.ToTensor())
    # 흑백 변환
    # transform.append(T.Grayscale())
    transform = T.Compose(transform)

    # 데이터셋
    dataset = DiabeticRetinopathy(image_dir, attr_path, mode, transform)
    # 데이터 로드
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
#
    return data_loader