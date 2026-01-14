import gradio as gr
import joblib
import pandas as pd
import json

# 1. Load the exported artifacts
model_pipeline = joblib.load('./models/logistic_regression_pipeline.joblib')
encoders = joblib.load('./models/label_encoders.joblib')
with open('feature_metadata.json', 'r') as f:
    metadata = json.load(f)

feature_names = metadata['feature_names']

# --- MAPPING DICTIONARIES ---
# Maps descriptive text from dropdowns to numerical values (0-5) for the ML model
pressure_map = {
    "0 = No pressure": 0, "1 = Little": 1, "2 = Average": 2, 
    "3 = High": 3, "4 = Very high": 4, "5 = Very very high": 5
}

satisfaction_map = {
    "0 = Very Dissatisfied": 0, "1 = Dissatisfied": 1, "2 = Neutral": 2, 
    "3 = Satisfied": 3, "4 = Very Satisfied": 4, "5 = Extremely Satisfied": 5
}

financial_stress_map = {
    "1 = No Stress": 1, "2 = Low": 2, "3 = Moderate": 3, "4 = High": 4, "5 = Extreme": 5
}

def predict_depression(
    name, phone, gender, age, status, profession, acad_press_label, work_press_label, 
    cgpa, study_sat_label, job_sat_label, sleep, diet, suicide, 
    hours, finance_stress_label, family_hist
):
    # --- FORM VALIDATION SECTION ---
    if not gender or not status or not sleep or not diet or not suicide or not family_hist:
        raise gr.Error("Please fill in all required fields before submitting.")

    # Convert descriptive labels back to numerical integers for the model
    acad_press = pressure_map.get(acad_press_label, 0)
    work_press = pressure_map.get(work_press_label, 0)
    study_sat = satisfaction_map.get(study_sat_label, 0)
    job_sat = satisfaction_map.get(job_sat_label, 0)
    finance_stress = financial_stress_map.get(finance_stress_label, 1)

    # Logical Validation: Student check
    if status == "Student" and (cgpa is None or cgpa < 0):
        raise gr.Error("Please provide a valid CGPA for Student status.")
    
    # Logical Validation: Profession check
    if status == "Working Professional" and (not profession or len(profession.strip()) < 2):
        raise gr.Error("Please specify your Profession.")

    # 2. Create a dictionary from inputs
    input_data = {
        'Gender': gender,
        'Age': age,
        'Working Professional or Student': status,
        'Profession': profession if status == 'Working Professional' else 'Student',
        'Academic Pressure': acad_press if status == 'Student' else 0,
        'Work Pressure': work_press if status == 'Working Professional' else 0,
        'CGPA': cgpa if status == 'Student' else 0,
        'Study Satisfaction': study_sat if status == 'Student' else 0,
        'Job Satisfaction': job_sat if status == 'Working Professional' else 0,
        'Sleep Duration': sleep,
        'Dietary Habits': diet,
        'Have you ever had suicidal thoughts ?': suicide,
        'Work/Study Hours': hours,
        'Financial Stress': finance_stress,
        'Family History of Mental Illness': family_hist
    }

    try:
        # 3. Preprocess and Align
        df_input = pd.DataFrame([input_data])
        
        for col in ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']:
            df_input[col] = encoders[col].transform(df_input[col].astype(str))
        
        df_encoded = pd.get_dummies(df_input)
        
        for col in feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_final = df_encoded[feature_names]

        # 4. Predict Probability
        prob = model_pipeline.predict_proba(df_final)[0][1]
        
        # 5. Intervention Logic
        if prob < 0.3:
            level, color = "LOW RISK", "🟢"
            advice = "Continue your healthy habits! Regular exercise and mindfulness can maintain this state."
        elif prob < 0.7:
            level, color = "MODERATE RISK", "🟡"
            advice = "You are showing some signs of stress. Consider a wellness check-in or guided meditation."
        else:
            level, color = "HIGH RISK", "🔴"
            advice = "Risk detected. We recommend speaking with a mental health professional or using a support hotline."

        return f"{color} {level}", f"{prob:.2%}", advice

    except Exception as e:
        raise gr.Error(f"An error occurred during prediction: {str(e)}")

# UI Component Options
pressure_options = list(pressure_map.keys())
sat_options = list(satisfaction_map.keys())
finance_options = list(financial_stress_map.keys())

# 6. Define Gradio Interface
interface = gr.Interface(
    fn=predict_depression,
    inputs=[
        gr.Textbox(label="Name"),
        gr.Textbox(label="Phone Number"),
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Slider(18, 65, step=1, label="Age"),
        gr.Radio(["Student", "Working Professional"], label="Status"),
        gr.Textbox(label="Profession (if applicable)"),
        gr.Dropdown(pressure_options, label="Academic Pressure Level"), # Changed to Dropdown
        gr.Dropdown(pressure_options, label="Work Pressure Level"),     # Changed to Dropdown
        gr.Number(label="CGPA (Students only)"),
        gr.Dropdown(sat_options, label="Study Satisfaction Level"),      # Changed to Dropdown
        gr.Dropdown(sat_options, label="Job Satisfaction Level"),        # Changed to Dropdown
        gr.Dropdown(["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"], label="Sleep Duration"),
        gr.Dropdown(["Healthy", "Moderate", "Unhealthy"], label="Dietary Habits"),
        gr.Radio(["No", "Yes"], label="Have you ever had suicidal thoughts?"),
        gr.Slider(0, 15, step=1, label="Daily Work/Study Hours"),
        gr.Dropdown(finance_options, label="Financial Stress Level"),   # Changed to Dropdown
        gr.Radio(["No", "Yes"], label="Family History of Mental Illness")
    ],
    outputs=[
        gr.Textbox(label="Assessment"),
        gr.Textbox(label="Risk Probability"),
        gr.Textbox(label="Recommendation")
    ],
    title="AI Depression Risk Screener",
    description="Welcome to a secure, AI-powered platform designed to assess depression risk using validated lifestyle and behavioral indicators. \n\n Your privacy is our priority. All information provided is handled securely, anonymized, and never shared with third parties. \n\n By using this platform, you provide informed consent for your data to be processed solely for depression risk prediction and research-driven improvement of the system."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)