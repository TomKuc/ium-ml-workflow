from datasets import load_dataset, DatasetDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib

dataset = load_dataset('ag_news')
split = dataset["train"].train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict({
    "train": split["train"],
    "dev": split["test"],
    "test": dataset["test"]
})

X_train = dataset["train"]["text"]
y_train = dataset["train"]["label"]

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, random_state=42))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "mlp_model.joblib")