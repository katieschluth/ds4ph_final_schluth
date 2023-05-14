import streamlit as st
import pandas as pd
from models import los_model, charges_model

#los_model = joblib.load("los_model.pkl")
#charges_model = joblib.load("charges_model.pkl")

def main():
    st.title("Estimate length of hospital stay and total charges based on location and demographic variables")
    
    # Create dropdown menus for user input
    input_features = ["Hospital County", "Age Group", "Sex", "Race", "Ethnicity", "Type of Admission", "Form of Payment"]
    options = {
        "Hospital County": ['Bronx', 'Manhattan', 'Kings', 'Queens', 'Rockland', 'Westchester', 'Onondaga', 'Nassau', 'Otsego', 'Delaware', 'Sullivan', 'Orange', 'Monroe', 'Ontario', 'Columbia', 'Albany', 'Steuben', 'Cayuga'],
        "Age Group": ['0 to 17',"18 to 29", "30 to 49", "50 to 69", "70 or Older"],
        "Sex": ["M", "F"],
        "Race": ['White', 'Black/African American', 'Multi-racial','Other Race'],
        "Ethnicity": ['Spanish/Hispanic','Not Span/Hispanic', 'Multi-ethnic'],
        "Type of Admission": ['Emergency', 'Newborn', 'Elective', 'Urgent', 'Trauma'],
        "Form of Payment": ['Medicare', 'Private Health Insurance', 'Medicaid','Blue Cross/Blue Shield', 'Self-Pay', 'Miscellaneous/Other','Managed Care, Unspecified', 'Department of Corrections','Federal/State/Local/VA']
    }
    user_inputs = [st.selectbox(f"Select {feature}", options=options[feature], key=feature) for feature in input_features]
    
    # Process user input and run the regression model
    if st.button("Predict Length of Stay and Total Charges"):
        # Convert user inputs to a DataFrame or the required format for the model
        input_data = pd.DataFrame([user_inputs], columns=["hospital_county", "age_group", "gender", "race", "ethnicity", "type_of_admission", "payment_typology_1"])
        
        # Make predictions using the pre-trained model
        predict_stay = los_model.predict(input_data)
        predict_charges = charges_model.predict(input_data)
        
        # Display the predictions
        st.write("Predicted Length of Stay:")
        st.write(predict_stay)
        st.write("Predicted Total Charges:")
        st.write(predict_charges)
        
        # You can also display additional information or visualizations based on the predictions
        
# Run the app
if __name__ == "__main__":
    main()