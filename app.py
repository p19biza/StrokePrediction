from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
def load_model():
    with open('stroke_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form.get('gender')
    age = float(request.form.get('age'))
    avg_glucose_level = float(request.form.get('avg_glucose_level'))
    heart_disease = request.form.get('heart_disease')
    hypertension = request.form.get('hypertension')
    bmi = float(request.form.get('bmi'))
    work_type = request.form.get('work_type')
    smoking_status = request.form.get('smoking_status')
    residence_type = request.form.get('residence_type')
    ever_married = request.form.get('ever_married')
    
    # Convert categorical variables to numeric
    gender = 1 if gender == 'male' else 0
    heart_disease = 1 if heart_disease == 'yes' else 0
    hypertension = 1 if hypertension == 'yes' else 0
    ever_married = 1 if ever_married == 'yes' else 0
    residence_type = 1 if residence_type == 'urban' else 0
    work_type_mapping = {'private': 0, 'Self-employed': 1, 'children': 2, 'Govt_job': 3, 'Never_worked': 4}
    smoking_status_mapping = {'never smoked': 0, 'Unknown': 1, 'formerly smoked': 2, 'smokes': 3}
    
    work_type = work_type_mapping[work_type]
    smoking_status = smoking_status_mapping[smoking_status]

    # DataFrame for the model
    input_data = pd.DataFrame([[
        gender, age, hypertension, heart_disease, ever_married, work_type, 
        residence_type, avg_glucose_level, bmi, smoking_status
    ]], columns=[
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
    ])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "Υπάρχει κίνδυνος εγκεφαλικού" if prediction == 1 else "Δεν υπάρχει κίνδυνος εγκεφαλικού"

    return jsonify({'stroke_prediction': result})

if __name__ == '__main__':
    app.run(debug=True, threaded=False)
