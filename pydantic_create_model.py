import pandas as pd
from pydantic import BaseModel, NonNegativeInt, ValidationError, Field, create_model
from typing import Annotated, Optional
from enum import Enum
import datetime 


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




def create_pydantic_model(model_name,df):
    print(model_name)
    # Check if the number unique values of a category column exceeds the threshold -> if so, change it to str type
    # Also replace 'nan' (in str) to None
    for col in df.select_dtypes(include='category').columns:
        nunique = df[col].nunique()
        df[col] = df[col].astype("str").replace('nan', None)
        if nunique > CATEGORY_UNIQUE_VALUES_THRESHOLD:
            print(col, 'has', nunique, 'unique values. Changed to str dtype.')
        else:
            df[col] = df[col].astype("category")

    # Replace all 'nan' values to None and infer the dtypes of each column
    df = df.replace('nan', None)
    df = df.infer_objects()
    
    # Build fields for Pydantic model
    fields = {
        col: (map_dtype(col, dtype, df), ...)
        for col, dtype in df.dtypes.items()
    }

    # Create dynamic Pydantic model
    return create_model(model_name, **fields)
