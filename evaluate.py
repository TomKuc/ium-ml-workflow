from datasets import load_dataset, DatasetDict
import joblib
import pandas as pd

dataset = load_dataset('ag_news')
split = dataset["train"].train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict({
    "train": split["train"],
    "dev": split["test"],
    "test": dataset["test"]
})

X_test = dataset["test"]["text"]

model = joblib.load("mlp_model.joblib")
predictions = model.predict(X_test)

df = pd.DataFrame({"predicted_label": predictions})
df.to_csv("predictions.csv", index=False)