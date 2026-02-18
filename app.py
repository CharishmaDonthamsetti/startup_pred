from flask import Flask, render_template, request
import joblib
import numpy as np

# Load saved model
model = joblib.load("random_forest_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(request.form)  # Debugging (optional)

        # ✅ Numerical Inputs
        age_first_funding_year = float(request.form['age_first_funding_year'])
        age_last_funding_year = float(request.form['age_last_funding_year'])
        age_first_milestone_year = float(request.form['age_first_milestone_year'])
        age_last_milestone_year = float(request.form['age_last_milestone_year'])
        relationships = float(request.form['relationships'])
        funding_rounds = float(request.form['funding_rounds'])
        funding_total_usd = float(request.form['funding_total_usd'])
        milestones = float(request.form['milestones'])
        avg_participants = float(request.form['avg_participants'])

        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])

        # ✅ Checkbox Handling (CRITICAL FIX)
        is_CA = 1 if request.form.get("is_CA") else 0
        is_NY = 1 if request.form.get("is_NY") else 0
        is_MA = 1 if request.form.get("is_MA") else 0
        is_TX = 1 if request.form.get("is_TX") else 0
        is_otherstate = 1 if request.form.get("is_otherstate") else 0

        has_VC = 1 if request.form.get("has_VC") else 0
        has_angel = 1 if request.form.get("has_angel") else 0
        has_roundA = 1 if request.form.get("has_roundA") else 0
        has_roundB = 1 if request.form.get("has_roundB") else 0
        has_roundC = 1 if request.form.get("has_roundC") else 0
        has_roundD = 1 if request.form.get("has_roundD") else 0

        is_top500 = 1 if request.form.get("is_top500") else 0

        founded_year = float(request.form['founded_year'])

        # ✅ Feature Order MUST Match Training Data
        input_data = np.array([[
            latitude,
            longitude,
            age_first_funding_year,
            age_last_funding_year,
            age_first_milestone_year,
            age_last_milestone_year,
            relationships,
            funding_rounds,
            funding_total_usd,
            milestones,
            is_CA,
            is_NY,
            is_MA,
            is_TX,
            is_otherstate,
            has_VC,
            has_angel,
            has_roundA,
            has_roundB,
            has_roundC,
            has_roundD,
            avg_participants,
            is_top500,
            founded_year
        ]])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "Startup is likely Acquired ✅"
        else:
            result = "Startup is likely Closed ❌"

        return render_template("result.html", result=result)

    except Exception as e:
        return render_template("result.html", result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
