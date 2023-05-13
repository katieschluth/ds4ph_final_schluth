import pandas as pd
from sodapy import Socrata
import statsmodels.formula.api as smf
import pgeocode

client = Socrata("health.data.ny.gov", None)

# First 50,000 results, returned as JSON from API / converted to Python list of
# dictionaries by sodapy.
results = client.get("tg3i-cinn", limit=50000)

# Convert to pandas DataFrame
nyd2019_50k = pd.DataFrame.from_records(results)
nyd2019_50k.shape

#go from county to latitiude longitude
nomi = pgeocode.Nominatim('us')

nomi.query_location("Bronx", top_k=3)
## might need to go from inputted lat/long in the webapp to county name

# predict total charges and length of stay from demographic variables

## length of stay

#remove rows where length of stay >120
nyd2019_los = nyd2019_50k[~(nyd2019_50k['length_of_stay'] == '120 +')]
#make length of stay variable numeric
nyd2019_los['length_of_stay'] = pd.to_numeric(nyd2019_los['length_of_stay'])

los_model = smf.ols(formula='length_of_stay ~ C(hospital_county) + C(age_group) + C(gender) + C(race) + C(ethnicity) + C(type_of_admission) + C(payment_typology_1)', data=nyd2019_los).fit()

##total charges

#converting total charges column to numeric values
nyd2019_50k['total_charges'] = pd.to_numeric(nyd2019_50k['total_charges'])

charges_model = smf.ols(formula='total_charges ~ C(hospital_county) + C(age_group) + C(gender) + C(race) + C(ethnicity) + C(type_of_admission) + C(payment_typology_1)', data=nyd2019_50k).fit()

import joblib

#joblib.dump(los_model, "los_model.pkl")
#joblib.dump(charges_model, "charges_model.pkl")

##I started on this but don't know if it works

import streamlit as st

#los_model = joblib.load("los_model.pkl")
#charges_model = joblib.load("charges_model.pkl")

def main():
    st.title("Estimate length of hospital stay and total charges based on location and demographic variables")
    
    # Create dropdown menus for user input
    input_features = ["Hospital County", "Age Group", "Sex", "Race", "Ethnicity", "Type of Admission", "Form of Payment"]
    options = {
        "Hospital County": ["Bronx", "Cayuga", "Columbia", "Delaware", "Kings", "Manhattan", "Monroe", "Nassau", "Onondaga", "Ontario", "Orange", "Otsego", "Queens", "Rockland", "Steuben", "Sullivan", "Westchester"],
        "Age Group": ["18 to 29", "30 to 49", "50 to 69", "70 to 79"],
        "Sex": ["Male", "Female"],
        "Race": ["Multi-racial", "Other Race", "White"],
        "Ethnicity": ["Not Spanish/Hispanic", "Spanish/Hispanic", "Unknown"],
        "Type of Admission": ["Emergency", "Newborn", "Not Available", "Trauma", "Urgent"],
        "Form of Payment": ["Department of Corrections", "Federal/State/Local/VA", "Managed Care, Unspecified", "Medicaid", "Medicare", "Miscellaneous/Other", "Private Health Insurance", "Self-Pay"]
    }
    user_inputs = [st.selectbox(f"Select {feature}", options=options[feature], key=feature) for feature in input_features]
    
    # Process user input and run the regression model
    if st.button("Predict Length of Stay and Total Charges"):
        # Convert user inputs to a DataFrame or the required format for the model
        input_data = pd.DataFrame([user_inputs], columns=input_features)
        
        # Apply any necessary preprocessing to the input data
        
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