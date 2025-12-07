# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/jarpan03/Tourism-Package-Prediction/tourism.csv"
dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',                         # Age of the customer
    'CityTier',                    # City category (1, 2, or 3)
    'DurationOfPitch',             # Duration of the sales pitch (minutes)
    'NumberOfPersonVisiting',      # Number of people accompanying the customer
    'NumberOfFollowups',           # Number of follow-up attempts
    'PreferredPropertyStar',       # Preferred hotel star rating
    'NumberOfTrips',               # Annual number of trips taken
    'Passport',                    # Passport status (0: No, 1: Yes)
    'PitchSatisfactionScore',      # Customer's pitch satisfaction score
    'OwnCar',                      # Car ownership (0: No, 1: Yes)
    'NumberOfChildrenVisiting',    # Number of children below age 5 on the trip
    'MonthlyIncome'                # Customer's monthly income

]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',     # How the customer was contacted (Company Invited / Self Inquiry)
    'Occupation',        # Customer's occupation (Salaried, Freelancer, etc.)
    'Gender',            # Gender of the customer (Male / Female)
    'ProductPitched',    # Product/package pitched to the customer
    'MaritalStatus',     # Marital status (Single, Married, Divorced)
    'Designation'        # Customer's designation/role in their organization
]

# Define predictor matrix (X) using selected numeric and categorical features
X = dataset[numeric_features + categorical_features]

# Define target variable
y = dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="jarpan03/Tourism-Package-Prediction",
        repo_type="dataset",
    )
