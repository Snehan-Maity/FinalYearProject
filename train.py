import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.dataset import WHURS19FewShotDataset
from src.model import ProtoNet
# SETTINGS
DATASET_PATH = "/content/drive/MyDrive/WHU-RS19"
EPISODES = 3000
MODEL_SAVE_PATH = "protonet_dbcl.pth"
TEMPERATURE = 0.1
CONTRAST_WEIGHT = 0.05
# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# DATASET

dataset = WHURS19FewShotDataset(DATASET_PATH)
# MODEL
model = ProtoNet().to(device)
model.train()
# OPTIMIZER
optimizer = optim.SGD(
    model.parameters(),
    lr=0.05,
    momentum=0.9,
    weight_decay=5e-4
)
# LR SCHEDULER
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=1000,
    gamma=0.5
)
# CROSS ENTROPY LOSS
criterion = nn.CrossEntropyLoss()
# TRAINING LOOP
for episode in range(EPISODES):
    # SAMPLE EPISODE
    support_x, support_y, query_x, query_y = dataset.sample_episode()
    support_x = support_x.to(device)
    support_y = support_y.to(device)
    query_x = query_x.to(device)
    query_y = query_y.to(device)
    # FORWARD PASS
    logits, support_features, query_features = model(
        support_x,
        support_y,
        query_x
    )
    # NORMALIZE FEATURES
    support_features = F.normalize(support_features, dim=1)
    query_features = F.normalize(query_features, dim=1)
    unique_labels = torch.unique(support_y)
    query_class_centers = []
    for label in unique_labels:
      class_q = query_features[query_y == label]
      center = class_q.mean(dim=0)
      query_class_centers.append(center)
    query_class_centers = torch.stack(query_class_centers)
    query_class_centers = F.normalize(query_class_centers, dim=1)
    support_class_centers = []
    for label in unique_labels:
      class_s = support_features[support_y == label]
      center = class_s.mean(dim=0)
      support_class_centers.append(center)
    support_class_centers = torch.stack(support_class_centers)
    support_class_centers = F.normalize(support_class_centers, dim=1)
    task_sim = torch.matmul(
      query_class_centers,
      support_class_centers.T
    ) / TEMPERATURE

    label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}   #added2
    mapped_query = torch.tensor(   
    [label_map[l.item()] for l in unique_labels],
    device=device
    )
    task_labels = mapped_query   #added2

    task_contrast_loss = F.cross_entropy(task_sim, task_labels)
    # COMPUTE SUPPORT PROTOTYPES 
    prototypes = []
    for label in unique_labels:
        class_features = support_features[support_y == label]
        weights = torch.softmax(torch.norm(class_features, dim=1), dim=0)    #new
        prototype = torch.sum(weights.unsqueeze(1) * class_features, dim=0)    #new
        prototypes.append(prototype)
    prototypes = torch.stack(prototypes)
    prototypes = F.normalize(prototypes, dim=1)
    proto_dist = torch.cdist(prototypes, prototypes)   #added
    mask = torch.eye(proto_dist.size(0), device=device).bool()
    inter_class_loss = torch.mean(proto_dist[~mask])      #added
    # CONTRASTIVE LOSS (QUERY vs SUPPORT PROTOTYPES)
    similarity = torch.matmul(query_features, prototypes.T) / TEMPERATURE
    # --- Prototype contrast ---        added
    similarity = torch.matmul(query_features, prototypes.T) / TEMPERATURE
    proto_contrast_loss = F.cross_entropy(similarity, query_y)
    # --- Pairwise contrast ---
    support_expanded = support_features.unsqueeze(0)
    query_expanded = query_features.unsqueeze(1)
    pair_sim = torch.cosine_similarity(query_expanded, support_expanded, dim=2) / TEMPERATURE
    label_matrix = query_y.unsqueeze(1) == support_y.unsqueeze(0)
    pair_sim = pair_sim.view(-1)
    labels = label_matrix.float().view(-1)
    pair_contrast_loss = F.binary_cross_entropy_with_logits(pair_sim, labels)
    #added
    # TOTAL LOSS
    ce_loss = criterion(logits, query_y)
    # loss = ce_loss + CONTRAST_WEIGHT * contrast_loss     change
    loss = (    #added
    ce_loss
    + 0.4 * proto_contrast_loss
    + 0.2 * pair_contrast_loss
    #- 0.1 * inter_class_loss    rmv2
    + 0.1 * (1 / (inter_class_loss + 1e-6))   #added2
    + 0.3 * task_contrast_loss
    )    #added
    # BACKPROPAGATION
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # ACCURACY
    pred = torch.argmax(logits, dim=1)
    acc = (pred == query_y).float().mean()
    # PRINT PROGRESS
    if episode % 50 == 0:
        print(f"Episode {episode} | Loss {loss.item():.4f} | Acc {acc.item():.4f}")
    # LR SCHEDULER STEP
    if episode % 1000 == 0 and episode != 0:
        scheduler.step()
# SAVE MODEL
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("\nTraining complete.")
print("Model saved at:", MODEL_SAVE_PATH)
