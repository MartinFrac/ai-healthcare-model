Disease Diagnosis AI
This is a machine learning model that predicts the most likely disease based on a given set of symptoms. The model was trained on a dataset of symptoms and their corresponding diseases using a Support Vector Classifier (SVC) algorithm.

Usage
To use the model, you can call the predict_disease function with a list of symptoms as its argument. The function returns a tuple with the predicted disease and a formatted string containing the top 3 disease probabilities.

Example run:
symptoms_list = ['fatigue', 'muscle_wasting', 'weight_gain', 'irritability']
predicted_disease, disease_probs = predict_disease(symptoms_list)

print(predicted_disease)
print(disease_probs)

Output:
Predicted Disease: Paralysis (brain hemorrhage)

Top 3 Disease Probabilities:
Paralysis (brain hemorrhage): 66.6%
Allergy: 13.1%
Impetigo: 4.2%

Author
This code was written by Nikita Sisikin.
