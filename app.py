from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("models/random_forest.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get form data
        form_data = {key: float(value) for key, value in request.form.items()}
        
        # Convert to DataFrame
        input_data = pd.DataFrame([form_data])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return render_template("index.html", prediction=prediction)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
