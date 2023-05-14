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