import pandas as pd


# Load CSVs
training_data = pd.read_csv("/content/drive/MyDrive/Colab/training_data.csv")
validation_data = pd.read_csv("/content/drive/MyDrive/Colab/validation_data.csv")
test_data = pd.read_csv("/content/drive/MyDrive/Colab/test_data.csv")


# Extract features and labels
X_train = training_data[["laterality", "view_position"]]
y_train_birads = training_data["breast_birads"]
y_train_density = training_data["breast_density"]

X_val = validation_data[["laterality", "view_position"]]
y_val_birads = validation_data["breast_birads"]
y_val_density = validation_data["breast_density"]