__Name : Madiha Shafquat__

Data downloaded from: https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/4ny4-j5zv

Purpose is to build web app where users can input their county, some demographic info, and predict their length of stay and total charges.

To Do:
[x] Import data
[x] Length of stay regression model
[] Total charges regression model
[] Streamlit app
    [] basic app with inputs
        [] figure out how to make latitude and longitude input into county input for regression model 
            - maybe using the pgeocode library?
    [] feed inputs into regression models to display estimated total charges and length of stay
    [] (maybe) feature that says 'Your estimated total charges are low/typical/high for your county'