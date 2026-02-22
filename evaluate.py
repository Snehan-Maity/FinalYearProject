import torch
from src.dataset import WHURS19FewShotDataset
from src.model import ProtoNet

DATASET_PATH = "dataset/WHU-RS19"
MODEL_PATH = "protonet_dbcl.pth"

EPISODES = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = WHURS19FewShotDataset(DATASET_PATH)

model = ProtoNet().to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()

accuracies = []

with torch.no_grad():

    for episode in range(EPISODES):

        support_x, support_y, query_x, query_y = dataset.sample_episode()

        support_x = support_x.to(device)
        support_y = support_y.to(device)

        query_x = query_x.to(device)
        query_y = query_y.to(device)

        logits, _, _ = model(support_x, support_y, query_x)

        pred = torch.argmax(logits, dim=1)

        acc = (pred == query_y).float().mean()

        accuracies.append(acc.item())


mean_acc = sum(accuracies) / len(accuracies)

print(f"Average Accuracy over {EPISODES} episodes: {mean_acc:.4f}")