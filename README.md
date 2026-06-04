## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    OrdinalEncoder,
    LabelEncoder,
    OneHotEncoder,
    PowerTransformer
)
import category_encoders as ce

# STEP 1: Read dataset
df = pd.read_csv("C:\\Users\\acer\\Downloads\\Data_to_Transform.csv")

# STEP 2: Data Cleaning
df = df.drop_duplicates()
df = df.dropna()

print("Original Dataset")
print(df.head())

```
<img width="487" height="191" alt="image" src="https://github.com/user-attachments/assets/76a63911-81a2-40a7-9b91-a2ae4698ec90" />
```


# STEP 3: Feature Encoding (Example categorical column)

df["Category"] = ["Low","Medium","High"]*(len(df)//3)+["Low"]*(len(df)%3)

# Ordinal Encoding
ord_enc = OrdinalEncoder()
df["Ordinal_Encoded"] = ord_enc.fit_transform(df[["Category"]])

# Label Encoding
lab = LabelEncoder()
df["Label_Encoded"] = lab.fit_transform(df["Category"])

# Binary Encoding
binary = ce.BinaryEncoder(cols=["Category"])
binary_df = binary.fit_transform(df[["Category"]])

# One Hot Encoding
onehot = pd.get_dummies(df["Category"])

# STEP 4: Feature Transformation

# Log Transformation
df["Log"] = np.log(df["Moderate Positive Skew"])

# Reciprocal Transformation
df["Reciprocal"] = 1/(df["Moderate Positive Skew"])

# Square Root Transformation
df["Sqrt"] = np.sqrt(df["Moderate Positive Skew"])

# Square Transformation
df["Square"] = df["Moderate Positive Skew"]**2

# Box-Cox Transformation
boxcox = PowerTransformer(method="box-cox")
df["BoxCox"] = boxcox.fit_transform(
    df[["Moderate Positive Skew"]]
)

# Yeo-Johnson Transformation
yeo = PowerTransformer(method="yeo-johnson")
df["YeoJohnson"] = yeo.fit_transform(
    df[["Moderate Positive Skew"]]
)

# Combine encoded data
final_df = pd.concat(
    [df, binary_df, onehot],
    axis=1
)

# STEP 5: Save file
final_df.to_csv(
    "Transformed_Output.csv",
    index=False
)

print("\nTransformation Completed")
print("Saved as Transformed_Output.csv")
print(final_df.head())
```
<img width="721" height="526" alt="image" src="https://github.com/user-attachments/assets/bfaa1d39-a8b0-4dea-8c5b-8302bb5bf960" />

# RESULT:
The code was executed and verified succesfully.

       
