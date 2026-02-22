import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.dataset import WHURS19FewShotDataset
from src.model import ProtoNet


# ======================
# SETTINGS
# ======================

DATASET_PATH = "/content/drive/MyDrive/WHU-RS19"

EPISODES = 3000

MODEL_SAVE_PATH = "protonet_dbcl.pth"

TEMPERATURE = 0.1

CONTRAST_WEIGHT = 0.05


# ======================
# DEVICE
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ======================
# DATASET
# ======================

dataset = WHURS19FewShotDataset(DATASET_PATH)


# ======================
# MODEL
# ======================

model = ProtoNet().to(device)

model.train()


# ======================
# OPTIMIZER
# ======================

optimizer = optim.SGD(
    model.parameters(),
    lr=0.05,
    momentum=0.9,
    weight_decay=5e-4
)


# ======================
# LR SCHEDULER
# ======================

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=1000,
    gamma=0.5
)


# ======================
# CROSS ENTROPY LOSS
# ======================

criterion = nn.CrossEntropyLoss()


# ======================
# TRAINING LOOP
# ======================

for episode in range(EPISODES):

    # ======================
    # SAMPLE EPISODE
    # ======================

    support_x, support_y, query_x, query_y = dataset.sample_episode()

    support_x = support_x.to(device)
    support_y = support_y.to(device)

    query_x = query_x.to(device)
    query_y = query_y.to(device)


    # ======================
    # FORWARD PASS
    # ======================

    logits, support_features, query_features = model(
        support_x,
        support_y,
        query_x
    )


    # ======================
    # NORMALIZE FEATURES
    # ======================

    support_features = F.normalize(support_features, dim=1)

    query_features = F.normalize(query_features, dim=1)


    # ======================
    # COMPUTE SUPPORT PROTOTYPES
    # ======================

    unique_labels = torch.unique(support_y)

    prototypes = []

    for label in unique_labels:

        class_features = support_features[support_y == label]

        prototype = class_features.mean(dim=0)

        prototypes.append(prototype)

    prototypes = torch.stack(prototypes)

    prototypes = F.normalize(prototypes, dim=1)


    # ======================
    # CONTRASTIVE LOSS (QUERY vs SUPPORT PROTOTYPES)
    # ======================

    similarity = torch.matmul(query_features, prototypes.T) / TEMPERATURE

    contrast_loss = F.cross_entropy(similarity, query_y)


    # ======================
    # TOTAL LOSS
    # ======================

    ce_loss = criterion(logits, query_y)

    loss = ce_loss + CONTRAST_WEIGHT * contrast_loss


    # ======================
    # BACKPROPAGATION
    # ======================

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    # ======================
    # ACCURACY
    # ======================

    pred = torch.argmax(logits, dim=1)

    acc = (pred == query_y).float().mean()


    # ======================
    # PRINT PROGRESS
    # ======================

    if episode % 50 == 0:

        print(f"Episode {episode} | Loss {loss.item():.4f} | Acc {acc.item():.4f}")


    # ======================
    # LR SCHEDULER STEP
    # ======================

    if episode % 1000 == 0 and episode != 0:

        scheduler.step()


# ======================
# SAVE MODEL
# ======================

torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("\nTraining complete.")


print("Model saved at:", MODEL_SAVE_PATH)
