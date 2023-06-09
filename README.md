__Names : Katie Schluth, Jamie Vu, Madiha Shafquat__

This is a web app that allows users from New York state to enter information about their county, demographics, and hospital stay and outputs an estimate of their length of stay and total charges. 

### NY Dataset Description
The data for this project is published by the New York State Department of Health, and includes de-identified patient-level data on hospital in-patient discharges in New York State in 2019, including patient characteristics, diagnoses, treatments, services, and charges. The full dataset contains over 2.3 million records; for the purposes of our analysis, we have only used the first 50,000 records.

More information can be found at: https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/4ny4-j5zv  

### App Functionality
The app takes in user inputs and provides information about the following:   
   
__Length of Stay__  
_Predicted length of stay:_ Based on linear regression model trained on the NY dataset with covariates county, age, sex, race, ethnicity, type of admission, and form of payment. 

_Comparison to county:_ Provides context of whether estimate is 'high', 'typical', or 'low' relative to their __admission type__ and shows boxplot visual of where their predicted length of stay falls relative to admission type distribution.

__Total Charges__  
_Predicted total charges:_ Based on linear regression model trained on the NY dataset with covariates county, age, sex, race, ethnicity, type of admission, and form of payment. 

_Comparison to county:_ Provides context of whether predicted charges are 'high', 'typical', or 'low' relative to their __county__ and shows boxplot visual of where their predicted charges fall relative to county distribution.

### App Demonstration
Link: https://www.loom.com/share/5dbec227abd04d78b0e55b11f9249d6e
