from collections import Counter
from itertools import combinations
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/wdbc.csv")

# Exclude the identifier and diagnosis from the model inputs.
x = df.iloc[:, 2:]
y = df["diagnosis"].map({"benign": 0, "malignant": 1})

seed = 21

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=seed,
)


def weighted_voting(patient):
    score = 0
    for i in selected_indices:
        score += weights[i] * (patient.iloc[i] - ((malignant_means[i] + benign_means[i]) / 2))
    return score


selected_features = []
selections = []
golub_results = []

for fold_number, (train_idx, val_idx) in enumerate(cv.split(x, y), start=1):
    x_train = x.iloc[train_idx]
    y_train = y.iloc[train_idx]

    x_val = x.iloc[val_idx]
    y_val = y.iloc[val_idx]

    weights = []
    malignant_means = []
    benign_means = []

    # Estimate all parameters using training patients only to avoid leakage.
    for column in x_train:
        malignant = x_train[column].where(y_train == 1)
        benign = x_train[column].where(y_train == 0)

        malignant_mean = malignant.mean()
        benign_mean = benign.mean()

        malignant_std = malignant.std()
        benign_std = benign.std()

        weights.append((malignant_mean - benign_mean) / (malignant_std + benign_std))
        malignant_means.append(malignant_mean)
        benign_means.append(benign_mean)

    # Select the ten largest absolute weights within this training fold.
    selected_indices = sorted(range(len(weights)), key=lambda i: abs(weights[i]), reverse=True)[:10]

    # Store feature names for selection frequencies and Jaccard comparisons.
    feature_names = [x.columns[i] for i in selected_indices]
    selected_features.extend(feature_names)
    selections.append(set(feature_names))

    scores = []
    predictions = []

    for _, patient in x_val.iterrows():
        score = weighted_voting(patient)
        scores.append(score)
        predictions.append(1 if score >= 0 else 0)

    # Presentation of the resultats 
    auc = roc_auc_score(y_val, scores)
    balanced_accuracy = balanced_accuracy_score(y_val, predictions)

    golub_results.append({
        "evaluation": fold_number,
        "auc": auc,
        "balanced_accuracy": balanced_accuracy,
    })

    # Use the first fold as the detailed example
    if fold_number == 1:
        print("Golub — first evaluation")
        print("Confusion matrix")
        print("Label order: benign (0), malignant (1)")
        print(confusion_matrix(y_val, predictions, labels=[0, 1]))
        print("AUC:", auc)
        print("Balanced accuracy:", balanced_accuracy)

golub_results = pd.DataFrame(golub_results)

print("\nGolub — mean performance")
print(golub_results[["auc", "balanced_accuracy"]].mean())

# Iterate over all 30 features so that zero-frequency features are included.
counts = Counter(selected_features)
n_evaluations = len(selections)

print("\nFeature selection frequencies")
for feature in x.columns:
    count = counts[feature]
    frequency = count / n_evaluations * 100
    print(f"{feature}: {count}/{n_evaluations} ({frequency:.1f}%)")

# Compare every distinct pair of selected sets
jaccards = [
    len(a & b) / len(a | b)
    for a, b in combinations(selections, 2)
]

print("\nMean pairwise Jaccard similarity:", sum(jaccards) / len(jaccards))


logistic_results = []

# Reusing the splitter with the same integer seed reproduces the same splits.
for fold_number, (train_idx, val_idx) in enumerate(cv.split(x, y), start=1):
    x_train = x.iloc[train_idx]
    y_train = y.iloc[train_idx]

    x_val = x.iloc[val_idx]
    y_val = y.iloc[val_idx]

    # The pipeline fits scaling on training data only, using all 30 features.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="liblinear", C=1.0, max_iter=1000),
    )

    model.fit(x_train, y_train)

    # Use the default decision threshold and malignant-class probabilities.
    predictions = model.predict(x_val)
    scores = model.predict_proba(x_val)[:, 1]

    auc = roc_auc_score(y_val, scores)
    balanced_accuracy = balanced_accuracy_score(y_val, predictions)

    logistic_results.append({
        "evaluation": fold_number,
        "auc": auc,
        "balanced_accuracy": balanced_accuracy,
    })

logistic_results = pd.DataFrame(logistic_results)

print("\nLogistic regression — mean performance")
print(logistic_results[["auc", "balanced_accuracy"]].mean())