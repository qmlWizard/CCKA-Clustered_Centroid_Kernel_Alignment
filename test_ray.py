
import ray
from ray import tune
from ray.air import session
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

def tune_svm(config):
    # Train a model with the given hyperparameters from Ray Tune
    model = SVC(C=config["C"], gamma=config["gamma"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Report accuracy to Ray Tune
    session.report({"accuracy": accuracy})

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Define the search space
search_space = {
    "C": tune.grid_search([0.1, 0.5, 1.0, 1.5, 2.0]),
    "gamma": tune.grid_search(["scale", "auto"])
}

# Run the hyperparameter search
analysis = tune.run(
    tune.with_parameters(tune_svm),
    config=search_space,
    resources_per_trial={"gpu": 1},
    metric="accuracy",
    mode="max"
)

# Get the best hyperparameters
best_config = analysis.get_best_config(metric="accuracy", mode="max")
print(f"Best hyperparameters found: {best_config}")