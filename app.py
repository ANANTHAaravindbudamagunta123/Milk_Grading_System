from flask import Flask, render_template, request
import numpy as np
import pickle
import mysql.connector
import os

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), "milkgrade1.pkl")
model = pickle.load(open(model_path, "rb"))

# MySQL connection using environment variables (suitable for Render/Cloud)
db = mysql.connector.connect(
    host=os.environ.get("DB_HOST"),
    user=os.environ.get("DB_USER"),
    password=os.environ.get("DB_PASSWORD"),
    database=os.environ.get("DB_NAME"),
    ssl_ca=os.environ.get("DB_SSL_CA")  # Only needed for PlanetScale or secure remote DBs
)
cursor = db.cursor()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

@app.route("/pred", methods=["POST"])
def predict():
    try:
        # Collect form data
        ph = float(request.form["pH"])
        temperature = float(request.form["Temperature"])
        taste = float(request.form["Taste"])
        odor = float(request.form["Odor"])
        fat = float(request.form["Fat"])
        turbidity = float(request.form["Turbidity"])
        colour = float(request.form["Colour"])

        # Predict
        features = np.array([[ph, temperature, taste, odor, fat, turbidity, colour]])
        prediction = model.predict(features)[0]

        # Insert into DB
        sql = """
        INSERT INTO milk_data (ph, temperature, taste, odor, fat, turbidity, colour, quality)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (ph, temperature, taste, odor, fat, turbidity, colour, prediction)
        cursor.execute(sql, values)
        db.commit()

        return render_template("submit.html", prediction_text=f"{prediction}")
    except Exception as e:
        return f"Something went wrong: {e}"

if __name__ == "__main__":
    app.run(debug=True, port=7000)
