import pandas as pd
import random
import json

df = pd.read_csv("data/questions.csv")

paragen_dataset = []
paraident_dataset = []
for _, row in df.iterrows():
    if row.is_duplicate:
        paragen_dataset.append({
            "id": row["id"],
            "paraphrases": [row.question1, row.question2] 
        })
    paraident_dataset.append({
        "sentence1": row.question1,
        "sentence2": row.question2,
        "is_duplicate": row.is_duplicate
    })

# Split dataset to train:dev=9:1
dataset = list(zip(paragen_dataset, paraident_dataset))
random.shuffle(dataset)
train_set = dataset[:int(len(dataset)*0.90)]
dev_set = dataset[int(len(dataset)*0.90):int(len(dataset)*0.95)]
test_set = dataset[int(len(dataset)*0.95):]

train_paragen_set, train_paraident_set = zip(*train_set)
dev_paragen_set, dev_paraident_set = zip(*dev_set)
test_paragen_set, test_paraident_set = zip(*test_set)

# Put to file
print("train, dev, test")
print(f"{len(train_paragen_set)}, {len(dev_paragen_set)}, {len(test_paragen_set)}")
for filename, set in zip(["data/qqp_paragen_train.json", "data/qqp_paragen_dev.json", "data/qqp_paragen_test.json"], [train_paragen_set, dev_paragen_set, test_paragen_set]):
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(set, file, indent=4)
for filename, set in zip(["data/qqp_paraident_train.json", "data/qqp_paraident_dev.json", "data/qqp_paraident_test.json"], [train_paraident_set, dev_paraident_set, test_paraident_set]):
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(set, file, indent=4)
print("Complete!! Check data/ for datasets")