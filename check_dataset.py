import os

dataset_path = "dataset/WHU-RS19"

classes = os.listdir(dataset_path)

print("Total classes:", len(classes))

total_images = 0

for cls in classes:

    cls_path = os.path.join(dataset_path, cls)

    images = os.listdir(cls_path)

    print(cls, ":", len(images))

    total_images += len(images)

print("\nTotal images:", total_images)