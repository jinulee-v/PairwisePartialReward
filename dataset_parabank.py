import random
import json

with open("../parabank/parabank2.tsv") as file:
    df = file.readlines()
    df = [d.split('\t')[1:] for d in df]
    df = [[d.strip() for d in row if len(d) < 200] for row in df]
    df = [row for row in df if len(row) >= 2]

dataset = []
for id, row in enumerate(df):
    dataset.append({
        "id": id,
        "paraphrases": row
    })

# Split dataset to train:dev=9:1
random.shuffle(dataset)
train_set = dataset[:int(len(dataset)*0.97)]
dev_set = dataset[int(len(dataset)*0.97):int(len(dataset)*0.98)]
test_set = dataset[int(len(dataset)*0.98):]

# Put to file
print("train, dev, test")
print(f"{len(train_set)}, {len(dev_set)}, {len(test_set)}")
for filename, set in zip(["data/parabank_train.json", "data/parabank_dev.json", "data/parabank_test.json"], [train_set, dev_set, test_set]):
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(set, file, indent=4)
print("Complete!! Check data/ for datasets")