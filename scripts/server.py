import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

filename = "../saved models/final_model.pickle"

# Load the trained model
model = pickle.load(open(filename, "rb"))

def predict_disease(symptoms_list, model=model):
    # Read the symptom severity CSV file
    df1 = pd.read_csv('../datasets/Symptom-severity.csv')
    
    # Get the list of all available symptoms
    all_symptoms = df1['Symptom'].unique()
    
    # Create a dictionary of symptom weights
    symptom_weights = dict(zip(df1.Symptom, df1.weight))
    
    # Convert the user input list of symptoms to severity scores
    symptom_scores = [symptom_weights[symptom] for symptom in symptoms_list if symptom in all_symptoms]
    
    # Pad the symptom scores with 0's to match the number of symptoms in the encoded dataset
    max_num_symptoms = 17
    symptom_scores = symptom_scores + [0]*(max_num_symptoms - len(symptom_scores))
    
    # Reshape the symptom scores as a numpy array to match the input format of the SVC model
    symptom_scores = np.array(symptom_scores).reshape(1, -1)
    
    # Make a prediction and obtain the probability estimates for each of the possible classes
    predicted_disease = model.predict(symptom_scores)[0]
    prob_estimates = model.predict_proba(symptom_scores)[0]
    
    # Create a dictionary of disease names and their corresponding index in the probability estimates array
    disease_names = dict(zip(range(len(model.classes_)), model.classes_))
    
    # Create a list of tuples containing the disease name and its corresponding probability estimate
    disease_probs = [(disease_names[i], prob_estimates[i]) for i in range(len(prob_estimates))]
    
    # Sort the disease probability list in descending order of probability estimates
    disease_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Create a formatted string that includes the predicted disease and the list of disease probabilities
    disease_array = []
    for disease, prob in disease_probs[:3]:
        disease_array.append(f"{disease}: {prob * 100:.1f}%\n")
    
    return disease_array

app = Flask(__name__)

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    symptoms_array = data['symptoms']
    disease_probs = predict_disease(symptoms_array)
    print(symptoms_array)
    return jsonify(disease_probs)

if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")