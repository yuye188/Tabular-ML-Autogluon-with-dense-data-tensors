#!/usr/bin/env python
# coding: utf-8

# In[2]:

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # locate TabArenaIterator

from TabArenaIterator import TabArenaIterator
import pandas as pd
import numpy as np

final_table = pd.DataFrame(columns=['dataset_id', 'dataset_name', 
                                    'original_number_features', 'final_number_features',
                                    'original_number_rows', 'final_number_rows',
                                    'original_density', 'final_density', 
                                    'initial_r2', 'accuracy_initial',
                                    'r2', 'accuracy'])

tabArenaURL = 'https://raw.githubusercontent.com/TabArena/tabarena_dataset_curation/refs/heads/main/dataset_creation_scripts/metadata/tabarena_dataset_metadata.csv'
iterator = TabArenaIterator(tabArenaURL)
for row, df in iterator:
    print(row['dataset_id'])

    # Obtain the dense df and informations about the changes
    dense_df, infos = getDensitiesPlot(df, row['target_feature'], density_threshold=0.1, show_fig=False)

    infos['dataset_id'] = row['dataset_id']
    infos['dataset_name'] = row['dataset_name']

    # Append the row to the final table
    final_table.loc[len(final_table)] = infos



# In[1]:


def getDensitiesPlot(df, target_feature, density_threshold=0.1, show_fig = False):
    # Drop the target column and duplicated rows
    df_copy = df.copy().drop(columns=target_feature).drop_duplicates()
    densities = []

    # Calculate the original dataset informations
    original_number_features = len(df_copy.columns)
    original_number_rows = len(df_copy)
    print('Total features:', original_number_features, 'Total rows:', original_number_rows)

    # Calculate the orignial density
    total = 1
    for i in df_copy.nunique().values:
        total = np.multiply(total, i, dtype=object)
    original_density = len(df_copy)/total
    print('Original density:', original_density)
    print()
    densities.append(original_density)

    # Dropping all available columns one by one until the threshold is chased
    for feature, cardinality in df_copy.nunique().sort_values(ascending=False).items():

        # Drop the corresponding column
        df_copy = df_copy.drop(columns=feature).drop_duplicates()

        # Calculate agian the density
        total = 1
        for i in df_copy.nunique().values:
            total = np.multiply(total, i, dtype=object)

        density = len(df_copy)/total
        densities.append(density)

        # Check the requirement to stop dropping columns
        if density > density_threshold:
            break

    # Calculate the final dataset informations
    final_number_features = len(df_copy.columns)
    final_number_rows = len(df_copy)
    print('Features left:', final_number_features, 'Rows left:', final_number_rows)
    print('Final density:', density)
    print()

    # Plot the column dropping process
    if show_fig:
        fig = px.line(x=range(0, len(densities), 1), y=densities, labels={'x':'Number features dropped', 'y':'Density'})
        fig.show()

    # Return the final dataset and the change informations 
    return df_copy, {'original_number_features': original_number_features,
                     'original_number_rows' : original_number_rows,
                     'original_density': original_density, 
                     'final_number_features': final_number_features,
                     'final_number_rows': final_number_rows,
                     'final_density': density}


# In[4]:


final_table


# In[46]:


np.array(final_table['original_density'])


# In[62]:


values = np.array(final_table['original_density'])
log_values = np.log10(values[values > 0])

# Create dataframe
df = pd.DataFrame({"log10(value)": log_values})

# Plot histogram of log-values
fig = px.histogram(df, x="log10(value)", labels={'log10(value)':'log10(original_density)'}, nbins=50, text_auto=True, title="Original density of the datasets from TabArena.")
fig.update_layout(bargap=0.2)

fig.show()


# In[35]:


import plotly.express as px
fig = px.histogram(x=final_table['original_density'], 
                   labels={'x':'Original_density'}, text_auto=True, title='Original density of datasets from TabArena')
fig.update_layout(bargap=0.2)
fig.show()


# In[21]:


import plotly.express as px
fig = px.histogram(x=final_table['final_density'], title='Final density of datasets from TabArena after features dropping process.',
                   labels={'x':'final_density'}, text_auto=True)
fig.update_layout(bargap=0.2)
fig.show()


# In[ ]:


from TabArenaIterator import TabArenaIterator
from autogluon.tabular import TabularPredictor
import os
import pandas as pd
import numpy as np

models= ['XGB', 'NN_TORCH']

final_table = pd.DataFrame(columns=['dataset_id', 'dataset_name', 
                                    'original_number_features', 'final_number_features',
                                    'original_number_rows', 'final_number_rows',
                                    'original_density', 'final_density'])

for model in models:
    final_table['initial_' + model + '_r2'] = None
    final_table['initial_' + model + '_accuracy'] = None
    final_table['final_' + model + '_r2'] = None
    final_table['final_' + model + '_accuracy'] = None

tabArenaURL = 'https://raw.githubusercontent.com/TabArena/tabarena_dataset_curation/refs/heads/main/dataset_creation_scripts/metadata/tabarena_dataset_metadata.csv'
iterator = TabArenaIterator(tabArenaURL)
for row, df in iterator:
    print(row['dataset_id'])

    # Obtain the dense df and informations about the changes
    dense_df, infos = getDensitiesPlot(df, row['target_feature'], density_threshold=0.1, show_fig=False)

    # Train the initial models and get the metrics
    for model in models:
        metric, result = trainAutogluonModels(df, 'AutoGluonModels/' + 'initial_' + model + '_' + row['dataset_name'], 
                                                        row['problem_type'], row['target_feature'], {model:{}})
        infos['initial_' + model + '_' + metric] = result

    # If the dense df does not fulfill the requirement, skip to the next dataset
    if infos['final_number_features'] < 3 or infos['final_number_rows'] < 30:
        print('Feature or obaservations not fulfill requirements.')
        continue

    # Extract the target feature to the dense_df
    dense_df[row['target_feature']] = df[row['target_feature']].iloc[dense_df.index,]
    dense_df = dense_df.reset_index(drop=True)
    '''
    # Split the train (70%) and test (30%) dataset
    train_data = dense_df.iloc[:int(len(dense_df) * 0.7), ]
    test_data = dense_df.iloc[int(len(dense_df) * 0.7):, ]

    # Set the path where the model should be saved 
    path = 'AutoGluonModels/' + row['dataset_name']

    # Choose the metric for regression or classification
    if row['problem_type'] == 'regression':
        metric = 'r2'
    else:
        metric = 'accuracy' 

    # If the path (the model) already exists, load the model, if not, train the model
    if os.path.isdir(path):
        print('Model already exists. Loading...')
        predictor = TabularPredictor.load(path)
    else:
        print('Train a new model...')
        predictor = TabularPredictor(label=row['target_feature'], path=path, 
                                     eval_metric=metric, problem_type=row['problem_type'],
                                     verbosity=0).fit(train_data)

    # Set the result and other informations about the dataset
    result_metric = predictor.evaluate(test_data).get(metric)
    '''

    for model in models:
        metric, result = trainAutogluonModels(dense_df, 'AutoGluonModels/' + 'final_' + model + '_' + row['dataset_name'], 
                                                        row['problem_type'], row['target_feature'], {model:{}})
        infos['final_' + model + '_' + metric] = result

    infos['dataset_id'] = row['dataset_id']
    infos['dataset_name'] = row['dataset_name']


    # Append the row to the final table
    final_table.loc[len(final_table)] = infos


