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
        #prototypes = []   #changed
        #for label in torch.unique(support_labels):
        prototypes = []
        for label in torch.unique(support_labels):
          class_features = support_features[support_labels == label]
          # Query-guided attention
          q = query_features.unsqueeze(1)            # [Q,1,D]
          s = class_features.unsqueeze(0)            # [1,K,D]
          sim = torch.cosine_similarity(q, s, dim=2)  # [Q,K]
          weights = torch.softmax(sim, dim=1)         # attention weights
          weighted_proto = torch.bmm(
          weights.unsqueeze(1),
          s.expand(query_features.size(0), -1, -1)
          ).squeeze(1)

          prototypes.append(weighted_proto)
        # ✅ STACK AFTER LOOP
        prototypes = torch.stack(prototypes, dim=1)   # [Q, C, D]
        # Distance per query
        distances = torch.norm(
          query_features.unsqueeze(1) - prototypes,
          dim=2
        )
        logits = -distances
        return logits, support_features, query_features
