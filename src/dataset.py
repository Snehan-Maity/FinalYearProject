import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class WHURS19FewShotDataset(Dataset):

    def __init__(self, root_dir, image_size=84):

        self.root_dir = root_dir

        self.classes = os.listdir(root_dir)

        self.class_to_images = {}

        for cls in self.classes:

            cls_path = os.path.join(root_dir, cls)

            images = [
                os.path.join(cls_path, img)
                for img in os.listdir(cls_path)
            ]

            self.class_to_images[cls] = images


        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])


    def sample_episode(
            self,
            n_way=5,
            k_shot=1,
            q_query=15
    ):

        selected_classes = random.sample(self.classes, n_way)

        support_images = []
        support_labels = []

        query_images = []
        query_labels = []


        for label, cls in enumerate(selected_classes):

            images = random.sample(
                self.class_to_images[cls],
                k_shot + q_query
            )

            support = images[:k_shot]
            query = images[k_shot:]


            for img_path in support:

                img = Image.open(img_path).convert("RGB")

                img = self.transform(img)

                support_images.append(img)

                support_labels.append(label)


            for img_path in query:

                img = Image.open(img_path).convert("RGB")

                img = self.transform(img)

                query_images.append(img)

                query_labels.append(label)


        support_images = torch.stack(support_images)
        query_images = torch.stack(query_images)

        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)


        return (
            support_images,
            support_labels,
            query_images,
            query_labels
        )