import streamlit as st
import pandas as pd
from models import los_model, charges_model, compare_los, compare_charges
import plotly.express as px
from sodapy import Socrata

def main():
    st.subheader("Enter a few things about yourself and your hospital stay")
    
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
        predict_stay = los_model.predict(input_data).iloc[0]
        predict_charges = charges_model.predict(input_data).iloc[0]
        
        # Display the predictions
        st.subheader("Predicted Length of Stay:")
        st.write(round(predict_stay)," days")
        st.write("This is ",compare_los(predict_stay,input_data['hospital_county'].iloc[0]),"for your county")
        
        #compare to state
        st.subheader("Predicted Total Charges:")
        st.write("$",round(predict_charges))
        st.write("This is ",compare_charges(predict_charges,input_data['hospital_county'].iloc[0]),"for your county")
    
        # You can also display additional information or visualizations based on the predictions
        import plotly.express as px
        import plotly.graph_objects as go
        # call in data for county to visualize
        nyd2019_50k = pd.read_csv('total_charges.csv')
        # subsetting only the data from the county selected
        county_df = nyd2019_50k[nyd2019_50k["hospital_county"] == input_data['hospital_county'].iloc[0]]
        tot_charges_df = county_df['total_charges']

        fig_charge = go.Figure()
        fig_charge.add_trace(go.Box(x=tot_charges_df, orientation="h"))
        fig_charge.add_vline(x=predict_charges, line_width=3, line_dash="dash", line_color="red", annotation=dict(text="Your projected total charges"))
        fig.update_layout(xaxis_range=[0, 1500000])
        fig_charge.update_yaxes(showticklabels=False)
        
        st.plotly_chart(fig_charge, use_container_width=True)

        
# Run the app
if __name__ == "__main__":
    main()