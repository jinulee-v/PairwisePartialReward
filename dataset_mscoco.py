from cgi import test
import json
import datasets
import random


print("Loading MS")
mscoco = datasets.load_dataset(
    "merve/coco",
    revision="9e50abc",
    data_files={"train": "annotations/captions_train2017.json", "test": "annotations/captions_val2017.json"}
)
print(mscoco)

# Parse train set
raw_dataset = mscoco["train"]["annotations"][0]
dataset_dict = {}
for data in raw_dataset:
    if data["image_id"] not in dataset_dict:
        dataset_dict[data["image_id"]] = []
    dataset_dict[data["image_id"]].append(data["caption"])

dataset = []
for k, v in dataset_dict.items():
    dataset.append({
        "id": k,
        "paraphrases": v
    })

# Split dataset to train:dev=9:1
random.shuffle(dataset)
dev_set = dataset[:int(len(dataset)*0.05)]
train_set = dataset[int(len(dataset)*0.05):]

# Parse test set
raw_dataset = mscoco["test"]["annotations"][0]
dataset_dict = {}
for data in raw_dataset:
    if data["image_id"] not in dataset_dict:
        dataset_dict[data["image_id"]] = []
    dataset_dict[data["image_id"]].append(data["caption"])

dataset = []
for k, v in dataset_dict.items():
    dataset.append({
        "id": k,
        "paraphrases": v
    })
test_set = dataset

# Put to file
print("train, dev, test")
print(f"{len(train_set)}, {len(dev_set)}, {len(test_set)}")
for filename, set in zip(["data/mscoco_paraphrase_train.json", "data/mscoco_paraphrase_dev.json", "data/mscoco_paraphrase_test.json"], [train_set, dev_set, test_set]):
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(set, file, indent=4)
print("Complete!! Check data/ for datasets")