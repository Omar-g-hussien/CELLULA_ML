import numpy as np
from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import LabelEncoder

# Create the application object
app = Flask(__name__)
file_path_m = r'D:\CELLULA\app\model\model.pkl'
file_path_s = r'D:\CELLULA\app\model\scaler.pkl'
file_path_l = r'D:\CELLULA\app\model\label.pkl'

# Load the model and label encoder
try:
    model = pickle.load(open(file_path_m, 'rb'))
    scaler = pickle.load(open(file_path_s, 'rb'))
    with open(file_path_l, 'rb') as f:
        label_encoder = pickle.load(f)
        label_encoder.set_params(handle_unknown='ignore')  # Handle unseen labels
except Exception as e:
    print(f"Error loading the model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    features = [int(x) if x.isdigit() else x for x in request.form.values()]
    # Apply preprocessing
    features_encoded = [label_encoder.transform([x]) if isinstance(x, str) else x for x in features]
    features_normalized = scaler.transform([features_encoded])
    # Make predictions
    pred = model.predict(features_normalized)

    prediction = "Canceled" if pred[0] == 1 else "Not Canceled"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
