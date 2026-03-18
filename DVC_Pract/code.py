import pandas as pd
import os

# Create a small DataFrame with COls names
data = {
    "Name": ['Alice','Bob','Charlie'],
    "Age" : [25,30,35],
    "City" : ["New York","Los Angeles","San Diego"]
}

df = pd.DataFrame(data)

new_role_loc = {'Name':'GF1','Age':20,'City':'City2'}
df.loc[len(df.index)] = new_role_loc

# Adding New Row to DF for V2
# New Row {"Name":'v2',"age":20,'City':'City'}
# df.iloc[len(df.index)] = new_roc_loc2

# # Adding new row to df for V1
# new_row_loc2 = {"Name": 'V3', "Age": 30, "city":"City1"}
# df.loc[len(df.index8)] = new_row_loc2

# Ensure the 'Data directory exists at the root level
data_dir = 'data'
os.makedirs(data_dir,exist_ok=True)

# Adding new role to df for v2



# Define the Path
file_path = os.path.join(data_dir,'sample_data.csv')

# Save the DataFrame to a CSV File, including column names
df.to_csv(file_path,index=False)

print(f'CSV file saved to {file_path}')