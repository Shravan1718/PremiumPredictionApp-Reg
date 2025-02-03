
import pandas as pd
from joblib import load

model = load("artifacts/model.joblib")
scaler_c_m = load("artifacts/scaler_with_cols_and_maps.joblib")

def scale_num_cols(df_values):
    df_values['income_level'] = 0
    
    cols_to_scale = scaler_c_m['cols_to_scale']
    scaler_model = scaler_c_m['scaler']
    df_values[cols_to_scale] = scaler_model.transform(df_values[cols_to_scale])
    df_values.drop('income_level',axis=1,inplace=True)
    return df_values

def lifestyle_score_cal(df_values,input_dict):
    physical_activity_encode = scaler_c_m['mapping_physical_activity']
    stress_level_encode = scaler_c_m['mapping_stress_level']
    lifestyle_score = physical_activity_encode.get(input_dict['physical_activity'],0) + stress_level_encode.get(input_dict['stress_level'],0)
    return lifestyle_score


def med_score_cal(m_history):
    risk_score_encode = scaler_c_m['mapping_risk_score']
    
    d1, d2 = (m_history.lower().split(' & ') + ['none'])[:2]
    total_risk_score = risk_score_encode.get(d1, 0) + risk_score_encode.get(d2, 0)
    return total_risk_score

def preprocess_input(input_dict):
    expected_columns = ['age', 'number_of_dependants', 'income_lakhs', 'insurance_plan',
       'gender_Male', 'region_Northwest', 'region_Southeast',
       'region_Southwest', 'marital_status_Unmarried', 'bmi_category_Obesity',
       'bmi_category_Overweight', 'bmi_category_Underweight',
       'smoking_status_Occasional', 'smoking_status_Regular',
       'employment_status_Salaried', 'employment_status_Self-Employed',
       'med_risk_score', 'lifestyle_risk_score']
    
    df_values = pd.DataFrame(0,columns=expected_columns,index=[0]) 
    
    insurance_plan_encode = scaler_c_m['mapping_insurance_plan']

    for key, val in input_dict.items():
        if key == 'age':
            df_values['age'] = val
        elif key == 'num_dependencies':
            df_values['number_of_dependants'] = val
        elif key == 'income_lakhs':
            df_values['income_lakhs'] = val
        elif key == 'insurance_plan':
            df_values['insurance_plan'] = insurance_plan_encode[val]
        elif key == 'gender' and val == 'Male':
            df_values['gender_Male'] = 1
        elif key == 'region':
            if val in ['Northwest', 'Southeast', 'Southwest']:
                df_values[f'region_{val}'] = 1
        elif key == 'marital_status' and val == 'Unmarried':
            df_values['marital_status_Unmarried'] = 1
        elif key == 'bmi_category':
            if val in ['Underweight', 'Overweight', 'Obesity']:
                df_values[f'bmi_category_{val}'] = 1
        elif key == 'smoking_status':
            if val in ['Occasional', 'Regular']:
                df_values[f'smoking_status_{val}'] = 1
        elif key == 'employment_status':
            if val in ['Salaried', 'Self-Employed']:
                df_values[f'employment_status_{val}'] = 1


    m_history = input_dict['medical_history']
    df_values['med_risk_score'] = med_score_cal(m_history)

    df_values['lifestyle_risk_score'] = lifestyle_score_cal(df_values,input_dict)

    df_scaled = scale_num_cols(df_values)
    return df_scaled

def predict(input_dict):
    final_df = preprocess_input(input_dict)
    return model.predict(final_df)