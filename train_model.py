import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import pickle

print("🚀 Training started...")

data = pd.read_csv("dataset2.csv")

desc_data = pd.read_csv("description.csv")

# Clean data
desc_data["Disease"] = desc_data["Disease"].str.lower().str.strip()
desc_data["Description"] = desc_data["Description"].str.strip()

data.columns = data.columns.str.strip()


data = data.apply(lambda x: x.str.lower().str.strip() if x.dtype == "object" else x)
data = data.fillna("")

symptom_columns = data.columns[1:]

symptoms = data[symptom_columns].values.tolist()

# Remove empty values
symptoms = [
    [s for s in row if s != ""]
    for row in symptoms
]

# Output labels
y = data["Disease"]


mlb = MultiLabelBinarizer()
X = mlb.fit_transform(symptoms)

print("✅ Total symptoms:", len(mlb.classes_))

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(mlb, open("encoder.pkl", "wb"))

print("🎉 Model trained and saved successfully!")