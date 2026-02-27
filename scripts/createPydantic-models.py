#!/usr/bin/env python
# coding: utf-8

# In[4]:

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # locate TabArenaIterator

from datetime import datetime
import pandas as pd
from pydantic import BaseModel, PositiveInt


class User(BaseModel):
    id: int  
    name: str = 'John Doe'  
    signup_ts: datetime | None  
    tastes: dict[str, PositiveInt]  


external_data = {
    'id': 123,
    'signup_ts': '2019-06-01 12:22',  
    'tastes': {
        'wine': 9,
        b'cheese': 7,  
        'cabbage': '1',  
    },
}

user = User(**external_data)  

print(user.id)
print(user.model_dump())  


# In[5]:


# continuing the above example...

from datetime import datetime
from pydantic import BaseModel, PositiveInt, ValidationError


class User(BaseModel):
    id: int
    name: str | None
    signup_ts: datetime | None
    tastes: dict[str, PositiveInt]

external_data = {
    'id': 123,
    'name' : 'Yu Ye',
    'signup_ts': '2025-06-01',  
    'tastes': {
        'wine': 9,
        'cheese': 7,  
        'cabbage': '1',  
    },
}



try:
    user = User(**external_data)  
except ValidationError as e:
    print(e.errors())

user.model_dump()


# # Autogluon dataset

# In[6]:


from autogluon.tabular import TabularDataset, TabularPredictor


# In[7]:


data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
train_data = TabularDataset(f'{data_url}train.csv')
train_data.head()


# In[8]:


from autogluon.tabular import TabularDataset, TabularPredictor

df = TabularDataset("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
df


# In[9]:


df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
       'marital_status', 'occupation', 'relationship', 'race', 'sex',
       'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
       'class_']
df


# In[10]:


for c in df.columns:
    print('***', c)
    print(df[c].unique())


# In[11]:


from pydantic import NonNegativeInt, Field
from typing import Annotated, Literal


class Person(BaseModel):
    age: Annotated[PositiveInt, Field(strict=True, lt=150)]
    workclass: Literal[' Private', ' State-gov', ' Local-gov', ' Self-emp-not-inc', 
                       ' Self-emp-inc', ' Federal-gov', ' Never-worked', ' Without-pay'] | None
    fnlwgt: PositiveInt 
    education: Literal[' Bachelors', ' 5th-6th', ' HS-grad', ' 7th-8th', ' Some-college',
                       ' Assoc-voc', ' Masters', ' Assoc-acdm', ' 11th', ' Prof-school',
                       ' Doctorate', ' 12th', ' 10th', ' 9th', ' Preschool', ' 1st-4th']
    education_num : Annotated[PositiveInt, Field(lt=20)] # if strict is not set, automatic conversion will be done
    marital_status : Literal[' Never-married', ' Married-civ-spouse', ' Divorced', ' Separated',
                             ' Widowed', ' Married-spouse-absent', ' Married-AF-spouse']
    occupation: str | None
    relationship: Literal[' Own-child', ' Not-in-family', ' Husband', ' Wife', ' Unmarried',' Other-relative']
    race : Literal[' White', ' Asian-Pac-Islander', ' Other', ' Black', ' Amer-Indian-Eskimo']
    sex : Literal[' Female', ' Male']
    capital_gain: NonNegativeInt
    capital_loss : NonNegativeInt
    hours_per_week: Annotated[PositiveInt, Field(strict=True, lt=100)] # if strict = true, automatic conversion will not be done
    native_country : str | None
    class_ : Literal[' <=50K', ' >50K']


# In[12]:


external_data = {'age': 25, 
                 'workclass': ' Local-gov', 
                 'fnlwgt': '178478', 
                 'education': ' Bachelors', 
                 'education_num': '13', 
                 'marital_status': ' Never-married', 
                 'occupation': ' Tech-support', 
                 'relationship': ' Own-child', 
                 'race': ' White', 
                 'sex': ' Female', 
                 'capital_gain': 0, 
                 'capital_loss': 0, 
                 'hours_per_week': 40, 
                 'native_country': ' United-States', 
                 'class_': ' <=50K'}

try:
    person = Person(**external_data)  
    print('ok.')
    print(repr(person))
except ValidationError as e:
    print(e.errors())



# In[13]:


external_data = {'age': 25, 
                 'workclass': ' Local-gov', 
                 'fnlwgt': '178478', 
                 'education': ' Bachelors', 
                 'education_num': '13', 
                 'marital_status': ' Never-married', 
                 'occupation': ' Tech-support', 
                 'relationship': ' Own-child', 
                 'race': ' White', 
                 'sex': ' Fema', 
                 'capital_gain': 0, 
                 'capital_loss': 0, 
                 'hours_per_week': 40, 
                 'native_country': ' United-States', 
                 'class_': ' <=50K'}

try:
    person = Person(**external_data)  
    print('ok.')
    print(repr(person))
except ValidationError as e:
    print(e.errors())



# In[14]:


df_replaced_none = df.replace(' ?', None)
df_replaced_none


# In[15]:


import pandas as pd
from pydantic import BaseModel, ValidationError


# Validate each row
validated_rows = []
errors = []

for index, row in df_replaced_none.iterrows():
    try:
        person = Person(**row.to_dict())
        validated_rows.append(person)
    except ValidationError as e:
        errors.append((index, e))

# Results
print("Valid rows:", len(validated_rows))
print("Errors:", len(errors))


# In[16]:


validated_rows[5]


# # Download datasets from openml

# In[2]:


import openml

df_openml = openml.datasets.list_datasets(output_format='dataframe')
df_openml


# In[11]:


df_openml_with_missing = df_openml[df_openml['NumberOfMissingValues'] > 0]
df_openml_with_missing[df_openml_with_missing['did'] > 46904]


# In[15]:


df_openml[df_openml['did'] == 46927] #46939 all vars


# In[3]:


dataset = openml.datasets.get_dataset(46927)
dataset


# In[4]:


# Get the data itself as a dataframe (or otherwise)
X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe")
X


# In[18]:


X.to_csv('../data/exampleSparseDataset.csv', index=False)


# In[7]:


X.dtypes


# In[21]:


y


# In[22]:


categorical_indicator


# In[9]:


attribute_names


# ## Easy example

# In[ ]:


import pandas as pd
from pydantic import create_model
from enum import Enum
from typing import Optional
import datetime
import numpy as np


# Sample unknown dataset
data = {
    "name": ["Alice", "Bob", "Charlie", "Alex"],
    "age": [30, 28, 'nan', 25],
    "score": [85.5, 88, 90.0, None],
    "passed": [True, True, False, True],
    "registered_on": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-05"), pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03")],
    "role": pd.Series(["admin", "Student", "user", "nan"], dtype="category")
}

df = pd.DataFrame(data)


df = df.replace('nan', None)
df = df.infer_objects()

# Build fields for Pydantic model
fields = {
    col: (map_dtype(col, dtype, df), ...)
    for col, dtype in df.dtypes.items()
}

# Create dynamic Pydantic model
DynamicModel = create_model("DynamicModel", **fields)

df_replaced_nan_for_none = df.astype(object).where(pd.notnull(df), None)

# Validate each row
validated = [DynamicModel(**row.to_dict()) for _, row in df_replaced_nan_for_none.iterrows()]

# Print validated models
for item in validated:
    print(item)



# In[ ]:


df


# In[ ]:


df_replaced_nan_for_none


# In[ ]:


external_data = {'name': 'Alice', 'age': None, 'score': 85.5, 'passed': True, 'registered_on': '2023-01-01 05:00:00', 'role': 'user'}
try:
    dynamicModel = DynamicModel(**external_data)  
    print('ok.')
    print(repr(dynamicModel))
except ValidationError as e:
    print(e.errors())


# In[ ]:


fields


# ## Tabarena example

# In[ ]:


for col in X.select_dtypes(include='category').columns:
    nunique = X[col].nunique()
    X[col] = X[col].astype("str").replace('nan', None)
    if nunique > CATEGORY_UNIQUE_VALUES_THRESHOLD:
        #X[col] = X[col].astype('string')
        #converted_columns.append(col)
        print(col, 'has', nunique, 'unique values. Changed to str dtype.')
    else:
        X[col] = X[col].astype("category")

X = X.replace('nan', None)
X = X.infer_objects()

# Build fields for Pydantic model
fields = {
    col: (map_dtype(col, dtype, X), ...)
    for col, dtype in X.dtypes.items()
}

# Create dynamic Pydantic model
DynamicModel = create_model("DynamicModel", **fields)

# Replace the nan values for None
X_replaced_nan_for_none = X.astype(object).where(pd.notnull(X), None)

# Validate each row
validated = [DynamicModel(**row.to_dict()) for _, row in X_replaced_nan_for_none.iterrows()]

# Print validated models
for item in validated:
    print(item)


# In[ ]:


fields


# In[ ]:


external_data = {'family': 'not_applicable', 
                 'product-type': 'C', 
                 'steel': 'M', 
                 'carbon': 0, 
                 'hardness': 0, 
                 'temper_rolling': 'not_applicable', 
                 'condition': 'not_applicable', 
                 'formability': 'not_applicable', 
                 'strength': 350, 
                 'non-ageing': 'not_applicable', 
                 'surface-finish': 'not_applicable', 
                 'surface-quality': 'G', 
                 'enamelability': 'not_applicable', 
                 'bc': 'not_applicable', 
                 'bf': 'not_applicable', 
                 'bt': 'not_applicable', 
                 'bw_me': 'not_applicable', 
                 'bl': 'not_applicable', 
                 'm': 'not_applicable', 
                 'chrom': 'not_applicable', 
                 'phos': 'not_applicable', 
                 'cbond': 'not_applicable', 
                 'marvi': 'not_applicable', 
                 'exptl': 'not_applicable', 
                 'ferro': 'not_applicable', 
                 'corr': 'not_applicable', 
                 'blue_bright_varn_clean': 'not_applicable', 
                 'lustre': 'not_applicable', 
                 'jurofm': 'not_applicable', 
                 's': 'not_applicable', 'p': 'not_applicable', 'shape': 'COIL', 'thick': 1.601, 'width': 609.9, 'len': 0, 'oil': 'not_applicable', 'bore': '0', 'packing': 'not_applicable', 'classes': '3'}


try:
    dynamicModel = DynamicModel(**external_data)  
    print('ok.')
    print(repr(dynamicModel))
except ValidationError as e:
    print(e.errors())


# In[ ]:


DynamicModel.model_json_schema()


# In[ ]:


fields


# # Try all datasets from tabarena (openML)

# In[ ]:


import pandas as pd
df_tabarena = pd.read_csv('https://raw.githubusercontent.com/TabArena/tabarena_dataset_curation/refs/heads/main/dataset_creation_scripts/metadata/tabarena_dataset_metadata.csv')
df_tabarena


# In[ ]:


print(df_tabarena.iloc[0].dataset_id)


# In[ ]:


# Create Enum from category values
def create_enum_from_categories(name, categories):
    return Enum(name, {str(cat): str(cat) for cat in categories})

CATEGORY_UNIQUE_VALUES_THRESHOLD = 50

# Map pandas dtype to Python type
def map_dtype(col, dtype, df):
    dtype_str = str(dtype)
    if dtype_str == "int64":
        return Optional[int] if df[col].isnull().any() else int
    elif dtype_str == "uint8":
        return Optional[Annotated[NonNegativeInt, Field(lt=255)]] if df[col].isnull().any() else Annotated[NonNegativeInt, Field(lt=255)] 
    elif dtype_str == "float64":
        return Optional[float] if df[col].isnull().any() else float
    elif dtype_str == "bool":
        return Optional[bool] if df[col].isnull().any() else bool
    elif dtype_str == "object":
        return Optional[str] if df[col].isnull().any() else str
    elif dtype_str == "datetime64[ns]":
        return Optional[datetime.datetime] if df[col].isnull().any() else datetime.datetime
    elif dtype_str == "timedelta64[ns]":
        return Optional[datetime.timedelta] if df[col].isnull().any() else datetime.timedelta
    elif dtype_str == "category":
        enum_type = create_enum_from_categories(f"{col.capitalize()}Enum", df[col].cat.categories)
        return Optional[enum_type] if df[col].isnull().any() else enum_type
    else:
        return Optional[str]


def checkAllTabarenaDatasets():
    # Iterare all datasets
    for id in df_tabarena['dataset_id']:
        print(id)

        # Get the dataset by ID
        dataset = openml.datasets.get_dataset(id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe")
        print('Number of rows:', len(X))

        # Check if the number unique values of a category column exceeds the threshold -> if so, change it to str type
        # Also replace 'nan' (in str) to None
        for col in X.select_dtypes(include='category').columns:
            nunique = X[col].nunique()
            X[col] = X[col].astype("str").replace('nan', None)
            if nunique > CATEGORY_UNIQUE_VALUES_THRESHOLD:
                print(col, 'has', nunique, 'unique values. Changed to str dtype.')
            else:
                X[col] = X[col].astype("category")

        # Replace all 'nan' (as string) values to None and infer the dtypes of each column
        X = X.replace('nan', None)
        X = X.infer_objects()

        # Build fields for Pydantic model
        fields = {
            col: (map_dtype(col, dtype, X), ...)
            for col, dtype in X.dtypes.items()
        }

        # Create dynamic Pydantic model
        DynamicModel = create_model("DynamicModel", **fields)

        # Replace the nan values for None
        X_replaced_nan_for_none = X.astype(object).where(pd.notnull(X), None)

        # Validate each row
        validated = [DynamicModel(**row.to_dict()) for _, row in X_replaced_nan_for_none.iterrows()]
        print('Number of validated rows:', len(validated))
        print(repr(validated[0]))
        print()
        # Print validated models
        #for item in validated:
        #    print(item)


# In[ ]:


checkAllTabarenaDatasets()


# In[ ]:


fields


# In[ ]:


# Create Enum from category values
def create_enum_from_categories(name, categories):
    return Enum(name, {str(cat): str(cat) for cat in categories})

CATEGORY_UNIQUE_VALUES_THRESHOLD = 50

# Map pandas dtype to Python type
def map_dtype(col, dtype, df):
    dtype_str = str(dtype)
    if dtype_str == "int64":
        return Optional[int] if df[col].isnull().any() else int
    elif dtype_str == "uint8":
        return Optional[Annotated[NonNegativeInt, Field(lt=255)]] if df[col].isnull().any() else Annotated[NonNegativeInt, Field(lt=255)] 
    elif dtype_str == "float64":
        return Optional[float] if df[col].isnull().any() else float
    elif dtype_str == "bool":
        return Optional[bool] if df[col].isnull().any() else bool
    elif dtype_str == "object":
        return Optional[str] if df[col].isnull().any() else str
    elif dtype_str == "datetime64[ns]":
        return Optional[datetime.datetime] if df[col].isnull().any() else datetime.datetime
    elif dtype_str == "timedelta64[ns]":
        return Optional[datetime.timedelta] if df[col].isnull().any() else datetime.timedelta
    elif dtype_str == "category":

        # If the number of unique values exceeds the threshold, set the type as str
        #if df[col].nunique() > CATEGORY_UNIQUE_VALUES_THRESHOLD:
        #    return Optional[str] if df[col].isnull().any() else str

        enum_type = create_enum_from_categories(f"{col.capitalize()}Enum", df[col].cat.categories)
        return Optional[enum_type] if df[col].isnull().any() else enum_type
    else:
        return Optional[str]


def checkAllTabarenaDatasets():
    # Iterare all datasets
    for id in df_tabarena['dataset_id']:
        print(id)

        # Get the dataset by ID
        dataset = openml.datasets.get_dataset(id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe")
        print('Number of rows:', len(X))

        # Check if the number unique values of a category column exceeds the threshold -> if so, change it to str type
        # Also replace 'nan' (in str) to None
        for col in X.select_dtypes(include='category').columns:
            nunique = X[col].nunique()
            X[col] = X[col].astype("str").replace('nan', None)
            if nunique > CATEGORY_UNIQUE_VALUES_THRESHOLD:
                print(col, 'has', nunique, 'unique values. Changed to str dtype.')
            else:
                X[col] = X[col].astype("category")

        # Replace all 'nan' values to None and infer the dtypes of each column
        X = X.replace('nan', None)
        X = X.infer_objects()

        # Build fields for Pydantic model
        fields = {
            col: (map_dtype(col, dtype, X), ...)
            for col, dtype in X.dtypes.items()
        }

        # Create dynamic Pydantic model
        DynamicModel = create_model("DynamicModel", **fields)

        # Replace the nan values for None
        X_replaced_nan_for_none = X.astype(object).where(pd.notnull(X), None)

        # Validate each row
        validated = [DynamicModel(**row.to_dict()) for _, row in X_replaced_nan_for_none.iterrows()]
        print('Number of validated rows:', len(validated))
        print(repr(validated[0]))
        print()
        # Print validated models
        #for item in validated:
        #    print(item)


# # Import iterators and create_model function

# In[1]:


from TabArenaIterator import TabArenaIterator
from pydantic import BaseModel, ValidationError
import pydantic_create_model as pdcm

pydantic_models: dict[str, BaseModel] = {}
tabArenaURL = 'https://raw.githubusercontent.com/TabArena/tabarena_dataset_curation/refs/heads/main/dataset_creation_scripts/metadata/tabarena_dataset_metadata.csv'
iterator = TabArenaIterator(tabArenaURL)
for row, df in iterator:
    model = pdcm.create_pydantic_model(str(row.dataset_id), df)
    pydantic_models[str(row.dataset_id)] = model


# In[5]:


pydantic_models


# In[6]:


type(pydantic_models.get('46906'))


# In[10]:


X.dtypes


# In[10]:


external_data = {'family': 'not_applicable', 
                 'product-type': 'g', 
                 'steel': 'M', 
                 'carbon': -5, 
                 'hardness': 0, 
                 'temper_rolling': 'not_applicable', 
                 'condition': 'not_applicable', 
                 'formability': 'not_applicable', 
                 'strength': 350, 
                 'non-ageing': 'not_applicable', 
                 'surface-finish': 'not_applicable', 
                 'surface-quality': 'G', 
                 'enamelability': 'not_applicable', 
                 'bc': 'not_applicable', 
                 'bf': 'not_applicable', 
                 'bt': 'not_applicable', 
                 'bw_me': 'not_applicable', 
                 'bl': 'not_applicable', 
                 'm': 'not_applicable', 
                 'chrom': 'not_applicable', 
                 'phos': 'not_applicable', 
                 'cbond': 'not_applicable', 
                 'marvi': 'not_applicable', 
                 'exptl': 'not_applicable', 
                 'ferro': 'not_applicable', 
                 'corr': 'not_applicable', 
                 'blue_bright_varn_clean': 'not_applicable', 
                 'lustre': 'not_applicable', 
                 'jurofm': 'not_applicable', 
                 's': 'not_applicable', 'p': 'not_applicable', 'shape': 'COIL', 'thick': 1.601, 'width': 609.9, 'len': 0, 'oil': 'not_applicable', 'bore': '0', 'packing': 'not_applicable', 'classes': '3'}


try:
    dynamicModel = pydantic_models.get('46906')(**external_data)  
    print('ok.')
    print(repr(dynamicModel))
except ValidationError as e:
    print(e.errors())


# In[ ]:


pydantic_models.get('46906').model_json_schema()


# In[ ]:


import pandas as pd
series = [
    ('a@a.com','Bill', 'Schneider', 123, 321, 20190502),
    ('a@a.com', 'Damian', 'Schneider', 124, 231, 20190502),
    ('b@b.com', 'Bill', 'Schneider',164, 313, 20190503),
    ('a@a.com','Bill', 'Schneider', 123, 321, 20190502),
    ('b@b.com', 'Bill', 'Schneider',164, 313, 20190503),
    ]

# Create a DataFrame object
df = pd.DataFrame(series, columns=['email', 'first name', 'last name', 'C_ID', 'A_ID', 'CreatedDate'])

# Find duplicate rows
df_duplicates = df[df.duplicated()]
print(df_duplicates)

