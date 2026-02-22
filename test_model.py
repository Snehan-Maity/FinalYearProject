#from src.dataset import WHURS19FewShotDataset
#from src.model import ProtoNet

#dataset = WHURS19FewShotDataset("dataset/WHU-RS19")

#support_x, support_y, query_x, query_y = dataset.sample_episode()

#model = ProtoNet()

#logits = model(support_x, support_y, query_x)
#print("Logits shape:", logits.shape)

#outputs = model(support_x, support_y, query_x)
#logits = outputs[0]
#print("Logits shape:", logits.shape)


import torch
from src.dataset import WHURS19FewShotDataset
from src.model import ProtoNet

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Google Drive dataset path
DATASET_PATH = "/content/drive/MyDrive/WHU-RS19"

# Load dataset
dataset = WHURS19FewShotDataset(DATASET_PATH)

# Sample one episode
support_x, support_y, query_x, query_y = dataset.sample_episode()

# Move data to GPU
support_x = support_x.to(device)
support_y = support_y.to(device)
query_x   = query_x.to(device)

# Load trained model
model = ProtoNet().to(device)
model.load_state_dict(torch.load("protonet_dbcl.pth", map_location=device))
model.eval()

# Inference
with torch.no_grad():
    logits, _ = model(support_x, support_y, query_x)

print("Logits shape:", logits.shape)

# Predictions
preds = torch.argmax(logits, dim=1)

# Accuracy
accuracy = (preds.cpu() == query_y).float().mean()

print("Test Episode Accuracy:", accuracy.item())
