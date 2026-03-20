from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
import pandas as pd
from chatbot import chatbot_response

# Load extra data
desc_data = pd.read_csv("description.csv")

# Load model and encoder
model = pickle.load(open("model.pkl", "rb"))
mlb = pickle.load(open("encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    symptoms = [s.strip() for s in request.form["symptoms"].lower().split(",")]

    input_data = mlb.transform([symptoms])
    result = model.predict(input_data)[0]

   #get description
    row = desc_data[desc_data["Disease"] == result]

    if not row.empty:
        description = row.iloc[0]["Description"]
    else:
        description = "No description available"




    return render_template(
        "index.html",
        prediction=result,
        description=description,
       
    )

if __name__ == "__main__":
    app.run(debug=True)