from src.dataset import WHURS19FewShotDataset

dataset = WHURS19FewShotDataset("dataset/WHU-RS19")

support_x, support_y, query_x, query_y = dataset.sample_episode()

print("Support shape:", support_x.shape)
print("Query shape:", query_x.shape)