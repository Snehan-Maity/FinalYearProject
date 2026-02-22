from src.dataset import WHURS19FewShotDataset
from src.model import ProtoNet

dataset = WHURS19FewShotDataset("dataset/WHU-RS19")

support_x, support_y, query_x, query_y = dataset.sample_episode()

model = ProtoNet()

#logits = model(support_x, support_y, query_x)
#print("Logits shape:", logits.shape)

outputs = model(support_x, support_y, query_x)
logits = outputs[0]
print("Logits shape:", logits.shape)

