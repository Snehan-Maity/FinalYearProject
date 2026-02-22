# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from src.resnet12 import ResNet12



# class ConvBlock(nn.Module):

#     def __init__(self, in_channels, out_channels):

#         super().__init__()

#         self.block = nn.Sequential(

#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.MaxPool2d(2)

#         )

#     def forward(self, x):

#         return self.block(x)


# class FeatureExtractor(nn.Module):

#     def __init__(self):

#         super().__init__()

#         self.encoder = nn.Sequential(

#             ConvBlock(3, 64),
#             ConvBlock(64, 64),
#             ConvBlock(64, 64),
#             ConvBlock(64, 64)

#         )


#     def forward(self, x):

#         x = self.encoder(x)

#         x = x.view(x.size(0), -1)

#         return x



# #--------------------------------------------------------------





# class ProtoNet(nn.Module):

#     def __init__(self):

#         super().__init__()

#         self.encoder = ResNet12()


#     def forward(
#         self,
#         support_images,
#         support_labels,
#         query_images
#     ):

#         support_features = self.encoder(support_images)

#         query_features = self.encoder(query_images)

#         prototypes = []

#         for label in torch.unique(support_labels):

#             class_features = support_features[support_labels == label]

#             prototype = class_features.mean(0)

#             prototypes.append(prototype)


#         prototypes = torch.stack(prototypes)

#         distances = torch.cdist(query_features, prototypes)

#         logits = -distances

#         return logits


#--------------------------------------------------------------



import torch
import torch.nn as nn
from src.resnet12 import ResNet12


class ProtoNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = ResNet12()


    def forward(
        self,
        support_images,
        support_labels,
        query_images
    ):

        support_features = self.encoder(support_images)

        query_features = self.encoder(query_images)

        prototypes = []

        for label in torch.unique(support_labels):

            class_features = support_features[support_labels == label]

            prototype = class_features.mean(0)

            prototypes.append(prototype)


        prototypes = torch.stack(prototypes)

        distances = torch.cdist(query_features, prototypes)

        logits = -distances

        return logits, support_features, query_features

