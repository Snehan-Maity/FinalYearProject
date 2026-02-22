import torch
import torch.nn as nn
import torch.nn.functional as F


class DBCLLoss(nn.Module):

    def __init__(self, temperature=0.1):

        super().__init__()

        self.temperature = temperature

        self.prototype_dict = {}


    def forward(self, features, labels):

        device = features.device

        loss = 0.0

        unique_labels = torch.unique(labels)

        prototypes = []

        proto_labels = []

        # compute prototypes
        for label in unique_labels:

            class_features = features[labels == label]

            prototype = class_features.mean(dim=0)

            prototypes.append(prototype)

            proto_labels.append(label)

            # store prototype in dictionary
            self.prototype_dict[int(label.item())] = prototype.detach()


        prototypes = torch.stack(prototypes)

        proto_labels = torch.tensor(proto_labels).to(device)


        # normalize
        features = F.normalize(features, dim=1)

        prototypes = F.normalize(prototypes, dim=1)


        # compute similarity
        logits = torch.matmul(features, prototypes.T) / self.temperature


        # ground truth indices
        target = torch.zeros(len(features), dtype=torch.long).to(device)

        for i in range(len(features)):

            label = labels[i]

            target[i] = torch.where(proto_labels == label)[0][0]


        loss = F.cross_entropy(logits, target)

        return loss