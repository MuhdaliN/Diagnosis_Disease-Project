import joblib
import numpy as np
import pandas as pd
from django.shortcuts import render, redirect
from django.conf import settings
from django.urls import reverse
import os

# Load artifacts at startup
model_path = os.path.join(settings.BASE_DIR, 'ml_models', 'disease_prediction_ensemble.pkl')
symptom_cols_path = os.path.join(settings.BASE_DIR, 'ml_models', 'symptom_columns.pkl')
selected_features_path = os.path.join(settings.BASE_DIR, 'ml_models', 'selected_features.pkl')
label_encoder_path = os.path.join(settings.BASE_DIR, 'ml_models', 'label_encoder.pkl')

model = joblib.load(model_path)
all_symptoms_raw = joblib.load(symptom_cols_path)
selected_features = joblib.load(selected_features_path)
label_encoder = joblib.load(label_encoder_path)

# Prettify symptom names for display
def prettify_symptom(name):
    return name.replace('_', ' ').title()

all_symptoms = [prettify_symptom(s) for s in all_symptoms_raw]
# Keep mapping from display name to raw feature name
symptom_display_to_raw = {prettify_symptom(s): s for s in all_symptoms_raw}

def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def predict(request):
    """GET: display symptom form"""
    return render(request, 'predict.html', {'all_symptoms': all_symptoms})

def result(request):
    """POST: run prediction and show results with chart"""
    if request.method != 'POST':
        return redirect('predict')
    
    # Build input vector with selected features
    input_dict = {feat: 0 for feat in selected_features}
    selected_display = request.POST.getlist('symptoms')
    
    # Map display names back to raw feature names
    selected_raw = [symptom_display_to_raw.get(name, name) for name in selected_display]
    
    for sym in selected_raw:
        # Clean exactly as in training
        sym_clean = sym.strip().replace(' ', '_').replace('(', '').replace(')', '')
        if sym_clean in input_dict:
            input_dict[sym_clean] = 1
    
    # Create DataFrame with correct column order
    df_input = pd.DataFrame([input_dict])[selected_features]
    
    # Predict probabilities
    proba = model.predict_proba(df_input)[0]
    pred_class_idx = np.argmax(proba)
    disease = label_encoder.inverse_transform([pred_class_idx])[0]
    confidence = proba[pred_class_idx] * 100
    
    # Top 3 diseases
    top3_idx = np.argsort(proba)[-3:][::-1]
    top3_diseases = label_encoder.inverse_transform(top3_idx)
    top3_conf = proba[top3_idx] * 100
    top3 = list(zip(top3_diseases, top3_conf))
    
    # Prepare data for chart (top 10)
    all_classes = label_encoder.classes_
    # Sort by probability descending
    sorted_indices = np.argsort(proba)[::-1]
    top10_indices = sorted_indices[:10]
    chart_labels = all_classes[top10_indices].tolist()
    chart_values = (proba[top10_indices] * 100).tolist()
    
    context = {
        'disease': disease,
        'confidence': f"{confidence:.2f}",
        'top3': top3,
        'chart_labels': chart_labels,
        'chart_values': chart_values,
        'all_symptoms': all_symptoms,
        'selected': selected_display,  # to re-check checkboxes if needed
    }
    return render(request, 'result.html', context)