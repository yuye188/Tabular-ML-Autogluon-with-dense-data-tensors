#!/usr/bin/env python
# coding: utf-8

# # Generate dataset

# In[114]:


import numpy as np
import pandas as pd
import time

def generate_dataset(num_rows, noise_std=0.1):
    """
    Generate a dataset with:
    - Two discrete features
    - One categorical feature
    - One continuous feature
    - One output variable using all input features in a Gaussian-like function with noise

    Parameters:
    num_rows (int): Number of rows in the dataset.
    noise_std (float): Standard deviation of Gaussian noise added to the output.

    Returns:
    pd.DataFrame: Generated dataset.
    """
    # Features
    discrete_1 = np.random.randint(0, 10, size=num_rows)
    discrete_2 = np.random.randint(10, 20, size=num_rows)
    categories = ['A', 'B', 'C', 'D']
    categorical = np.random.choice(categories, size=num_rows)
    continuous = np.random.uniform(0, 1, size=num_rows)

    # Encode categorical feature numerically for output calculation
    cat_encoded = np.array([categories.index(c) for c in categorical])

    # Combine all features into a Gaussian-like function
    # Example: y = exp(-((sum(features) - mu)^2) / (2*sigma^2)) + noise
    mu, sigma = 15, 5  # Adjusted for combined scale
    combined = discrete_1 + discrete_2 + cat_encoded + continuous * 10
    gaussian = np.exp(-((combined - mu)**2) / (2 * sigma**2))
    noise = np.random.normal(0, noise_std, size=num_rows)
    output = gaussian + noise

    # Create DataFrame
    df = pd.DataFrame({
        'Discrete_1': discrete_1,
        'Discrete_2': discrete_2,
        'Category': categorical,
        'Continuous': continuous,
        'Output': output
    })

    return df

# Example usage:
dataset = generate_dataset(8000, noise_std=0.15)
dataset


# In[104]:


dataset.describe()


# # Comparison between cpu-gpu

# In[ ]:


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

num_gpu_available = len(get_available_gpus())
print("Num GPUs Available: ", num_gpu_available)
device_lib.list_local_devices()


# In[ ]:


dataset_all = generate_dataset(100000, noise_std=0.15)


# In[ ]:


from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
import time

results = pd.DataFrame(columns=['num_rows', 'cpu_time', 'gpu_time', 'cpu_rmse', 'gpu_rmse', 
                                'cpu_disk_usage', 'cpu_disk_usage_opt', 'cpu_opt_rmse',
                                'gpu_disk_usage', 'gpu_disk_usage_opt', 'gpu_opt_rmse'])

for num_rows in [100, 500, 1000, 5000, 10000, 50000, 100000]:
    dataset = dataset_all.iloc[:num_rows,:].copy()
    train, test = train_test_split(dataset, test_size=0.3, shuffle=False)

    presets = ['medium_quality']

    # Only cpu
    start_time = time.time()
    predictor = TabularPredictor(label='Output', problem_type='regression', eval_metric='rmse', path='./ag-generatedDataset',verbosity=1
                                     ).fit(train_data=train, excluded_model_types=['GBM'], presets=presets)
    end_time = time.time()

    cpu_time = end_time - start_time
    cpu_rmse = predictor.evaluate(test)['root_mean_squared_error']
    cpu_disk_usage = predictor.disk_usage()/1e6

    # CPU opt
    path_opt = predictor.clone_for_deployment('./ag-generatedDataset-opt', dirs_exist_ok=True)
    predictor_opt = TabularPredictor.load(path=path_opt)
    cpu_disk_usage_opt = predictor_opt.disk_usage()/1e6
    cpu_opt_rmse = predictor_opt.evaluate(test)['root_mean_squared_error']

    # Cpu+GPU
    presets = ['medium_quality']
    start_time = time.time()
    predictor = TabularPredictor(label='Output', problem_type='regression', eval_metric='rmse', path='./ag-generatedDataset',verbosity=1
                                     ).fit(train_data=train, excluded_model_types=['GBM'], presets=presets, ag_args_fit={"num_cpus": 0, "num_gpus": num_gpu_available})
    end_time = time.time()

    gpu_time = end_time - start_time
    gpu_rmse = predictor.evaluate(test)['root_mean_squared_error']
    gpu_disk_usage = predictor.disk_usage()/1e6

    # GPU opt
    path_opt = predictor.clone_for_deployment('./ag-generatedDataset-opt', dirs_exist_ok=True)
    predictor_opt = TabularPredictor.load(path=path_opt)
    gpu_disk_usage_opt = predictor_opt.disk_usage()/1e6
    gpu_opt_rmse = predictor_opt.evaluate(test)['root_mean_squared_error']

    results.loc[len(results)] = {'num_rows': num_rows, 'cpu_time': cpu_time, 'gpu_time': gpu_time, 
                                 'cpu_rmse': cpu_rmse, 'gpu_rmse': gpu_rmse,
                                 'cpu_disk_usage': cpu_disk_usage, 'gpu_disk_usage': gpu_disk_usage, 
                                 'cpu_disk_usage_opt': cpu_disk_usage_opt, 'cpu_opt_rmse': cpu_opt_rmse,
                                 'gpu_disk_usage_opt': gpu_disk_usage_opt, 'gpu_opt_rmse': gpu_opt_rmse,}

results

