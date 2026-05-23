from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model and preprocessing objects
model = joblib.load("model/model.pkl")

scaler = joblib.load("model/scaler.pkl")

le_extra = joblib.load("model/le_extra.pkl")

le_training = joblib.load("model/le_training.pkl")

le_target = joblib.load("model/le_target.pkl")

feature_columns = joblib.load(
    "model/feature_columns.pkl"
)

@app.route('/')
def home():

    return render_template(
        'index.html',
        prediction_text=None,
        form_data=None
    )


@app.route('/predict', methods=['POST'])
def predict():

    try:

        # -----------------------------
        # Get Input Values
        # -----------------------------

        cgpa = float(request.form['CGPA'])

        internships = int(request.form['Internships'])

        projects = int(request.form['Projects'])

        workshops = int(
            request.form['Workshops_Certifications']
        )

        aptitude = int(
            request.form['AptitudeTestScore']
        )

        softskills = float(
            request.form['SoftSkillsRating']
        )

        extracurricular = request.form[
            'ExtracurricularActivities'
        ]

        training = request.form[
            'PlacementTraining'
        ]

        ssc = int(request.form['SSC_Marks'])

        hsc = int(request.form['HSC_Marks'])

        # -----------------------------
        # Encode categorical features
        # -----------------------------

        extracurricular_encoded = le_extra.transform(
            [extracurricular]
        )[0]

        training_encoded = le_training.transform(
            [training]
        )[0]

        # -----------------------------
        # Create Feature Array
        # -----------------------------

        features = pd.DataFrame([{

            "CGPA": cgpa,

            "Internships": internships,

            "Projects": projects,

            "Workshops/Certifications": workshops,

            "AptitudeTestScore": aptitude,

            "SoftSkillsRating": softskills,

            "ExtracurricularActivities":
                extracurricular_encoded,

            "PlacementTraining":
                training_encoded,

            "SSC_Marks": ssc,

            "HSC_Marks": hsc
        }])

        features = features[feature_columns]

        # -----------------------------
        # Scale Features
        # -----------------------------

        features_scaled = scaler.transform(features)

        # -----------------------------
        # Prediction
        # -----------------------------

        prediction = model.predict(
            features_scaled
        )

        probabilities = model.predict_proba(
            features_scaled
        )

        confidence = round(
            np.max(probabilities) * 100,
            2
        )

        result = le_target.inverse_transform(
            prediction
        )[0]

        # -----------------------------
        # Save Prediction History
        # -----------------------------

        prediction_record = {

            "CGPA": cgpa,

            "Internships": internships,

            "Projects": projects,

            "Workshops": workshops,

            "AptitudeScore": aptitude,

            "SoftSkills": softskills,

            "ExtracurricularActivities":
                extracurricular,

            "PlacementTraining":
                training,

            "SSC_Marks": ssc,

            "HSC_Marks": hsc,

            "Prediction": result,

            "Confidence": confidence
        }

        history_file = (
            "history/prediction_history.csv"
        )

        os.makedirs("history", exist_ok=True)

        new_df = pd.DataFrame(
            [prediction_record]
        )

        if os.path.exists(history_file):

            new_df.to_csv(
                history_file,
                mode='a',
                header=False,
                index=False
            )

        else:

            new_df.to_csv(
                history_file,
                index=False
            )

        # -----------------------------
        # Return Result
        # -----------------------------

        return render_template(

            'index.html',

            prediction_text=f"""
            Prediction: {result}
            | Confidence: {confidence}%
            """,

            form_data=request.form
        )

    except Exception as e:

        return render_template(

            'index.html',

            prediction_text=f"Error: {str(e)}",

            form_data=request.form
        )


@app.route('/history')
def history():

    history_file = (
        "history/prediction_history.csv"
    )

    if os.path.exists(history_file):

        df = pd.read_csv(history_file)

        return render_template(

            'history.html',

            columns=df.columns,

            rows=df.values.tolist()
        )

    return "No prediction history found."


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)

#checking if it works