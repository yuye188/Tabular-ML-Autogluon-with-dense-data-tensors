#!/usr/bin/env python
# coding: utf-8

# In[1]:


import openml
import pandas as pd
import numpy as np
import plotly.express as px


# In[93]:


df_openml = openml.datasets.list_datasets(output_format='dataframe')
df_openml


# In[15]:


dataset = openml.datasets.get_dataset(46904)
# Get the data itself as a dataframe (or otherwise)
X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe")
X


# In[247]:


dataset = openml.datasets.get_dataset(46927)
# Get the data itself as a dataframe (or otherwise)
X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe")
display(X)
total = 1
X = X.drop(columns=['attended', 'months_as_member', 'weight', 'days_before']).drop_duplicates()
for i in X.nunique().values:
    total = np.multiply(total, i, dtype=object)
print(total)
print(len(X))
len(X)/total


# In[139]:


obj = {'p1': 2, 'p2': 4}
obj['name'] = 'n1'
obj


# In[141]:


df1 = pd.DataFrame(columns= ['p1', 'p2', 'p3', 'name'])
df1.loc[len(df1)] = obj
df1


# In[35]:


df1 = pd.DataFrame(columns= ['p1', 'p2', 'p3', 'name'])
df1['jj'] = None
df1['kk'] = None
df1.loc[len(df1)] = obj
df1


# In[103]:


dataset = openml.datasets.get_dataset(46904)
# Get the data itself as a dataframe (or otherwise)
X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe")
X
dense_df = getDensitiesPlot(X, 'scaled-sound-pressure', 0.1, show_fig=True)
dense_df


# In[106]:


dense_df['scaled-sound-pressure'] = X['scaled-sound-pressure'].iloc[dense_df.index,]
dense_df = dense_df.reset_index(drop=True)
dense_df


# In[16]:


train_data = X.iloc[:int(len(X) * 0.7), ]
test_data = X.iloc[int(len(X) * 0.7):, ]
display(train_data)


# In[19]:


from  autogluon.tabular import TabularPredictor

label='scaled-sound-pressure'
predictor = TabularPredictor(label=label, problem_type='regression', path='./test', 
                             eval_metric='r2').fit(train_data, hyperparameters={'NN_TORCH':{}}, fit_weighted_ensemble=False, 
                                                         presets='best', time_limit=60, dynamic_stacking=False)


# In[20]:


y_pred = predictor.predict(test_data)
y_pred


# In[21]:


predictor = TabularPredictor.load('./test')
predictor.evaluate(test_data)


# In[22]:


predictor.leaderboard(test_data)


# In[231]:


predictor.leaderboard(test_data)


# In[255]:


predictor.evaluate(test_data)


# In[241]:


predictor.evaluate(test_data)


# In[233]:


predictor.model_best


# # Density plots

# # Iteration of all datasets from TabArena

# In[46]:


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


# In[ ]:


import os 

def trainAutogluonModels(dataset, path, problem_type, target_feature, model_hypers, target_numbers):
    # Split the train (70%) and test (30%) dataset
    train_data = dataset.iloc[:int(len(dataset) * 0.7), ]
    test_data = dataset.iloc[int(len(dataset) * 0.7):, ]

    # Choose the metric for regression or classification
    if problem_type == 'regression':
        metric = 'r2'
    else:
        metric = 'accuracy' 

    # If the path (the model) already exists, load the model, if not, train the model
    if os.path.isdir(path):
        print('Model already exists. Loading...')
        predictor = TabularPredictor.load(path)
    else:
        print('Train a new model...')
        predictor = TabularPredictor(label=target_feature, path=path, 
                                     eval_metric=metric, problem_type=problem_type,
                                     verbosity=0).fit(train_data, hyperparameters=model_hypers, fit_weighted_ensemble=False, 
                                                      time_limit=60, dynamic_stacking=False,
                                                      presets='medium' #if target_feature == 'LET_IS' else 'best',
                                                      )
    #print(path)
    # Get the metric
    result_metric = predictor.evaluate(test_data).get(metric)

    if metric == 'accuracy':
        chance = 1/target_numbers
        normalized_accuracy = (result_metric - chance)/ (1-chance)
    else:
        normalized_accuracy = None

    return metric, result_metric, normalized_accuracy


# In[49]:


from TabArenaIterator import TabArenaIterator
from autogluon.tabular import TabularPredictor
import os
import pandas as pd
import numpy as np

models= ['XGB', 'NN_TORCH']

final_table = pd.DataFrame(columns=['dataset_id', 'dataset_name', 
                                    'original_number_features', 'final_number_features',
                                    'original_number_rows', 'final_number_rows',
                                    'original_density', 'final_density', 'binary_feature'])

# Create the columns for models metrics
for model in models:
    final_table['initial_' + model + '_r2'] = None
    final_table['initial_' + model + '_accuracy'] = None
    final_table['initial_' + model + '_normalized_accuracy'] = None
    final_table['final_' + model + '_r2'] = None
    final_table['final_' + model + '_accuracy'] = None
    final_table['final_' + model + '_normalized_accuracy'] = None

tabArenaURL = 'https://raw.githubusercontent.com/TabArena/tabarena_dataset_curation/refs/heads/main/dataset_creation_scripts/metadata/tabarena_dataset_metadata.csv'
iterator = TabArenaIterator(tabArenaURL)
for row, df in iterator:
    print(row['dataset_id'])

    #if row['dataset_id'] != 46964:
    #    continue

    # Obtain the dense df and informations about the changes
    dense_df, infos = getDensitiesPlot(df, row['target_feature'], density_threshold=0.1, show_fig=False)

    # Train the initial models and get the metrics
    for model in models:
        metric, result, normalized_accuracy = trainAutogluonModels(df, 'AutoGluonModels/' + 'initial_' + model + '_' + row['dataset_name'], 
                                                        row['problem_type'], row['target_feature'], {model:{}}, row['num_classes'])
        infos['initial_' + model + '_' + metric] = result
        infos['initial_' + model + '_normalized_accuracy'] = normalized_accuracy

    # If the dense df does not fulfill the requirement, skip to the next dataset
    if infos['final_number_features'] >= 3 and infos['final_number_rows'] >= 30:
        print('Feature and observations fulfill requirements.')


        # Extract the target feature to the dense_df
        dense_df[row['target_feature']] = df[row['target_feature']].iloc[dense_df.index,]
        dense_df = dense_df.reset_index(drop=True)
        dense_df.to_csv('./dense_dfs/'+str(row['dataset_id'])+'.csv', index=False)

        # Train the models for dense_df and get the metrics
        for model in models:
            metric, result, normalized_accuracy = trainAutogluonModels(dense_df, 'AutoGluonModels/' + 'final_' + model + '_' + row['dataset_name'], 
                                                            row['problem_type'], row['target_feature'], {model:{}}, row['num_classes'])
            infos['final_' + model + '_' + metric] = result
            infos['final_' + model + '_normalized_accuracy'] = normalized_accuracy


    binary_feature = None
    for feature in dense_df.columns:
        if dense_df[feature].nunique() == 2:
            binary_feature = feature
            break

    # Append the dataset id and name
    infos['dataset_id'] = row['dataset_id']
    infos['dataset_name'] = row['dataset_name']
    infos['binary_feature'] = binary_feature


    # Append the row (dataset results) to the final table
    final_table.loc[len(final_table)] = infos



# In[50]:


final_table


# # Scatter plots

# In[51]:


def calculateInitialFinalImprovement(x):
    if pd.notna(x.initial_XGB_r2):
        return (x.final_XGB_r2 - x.initial_XGB_r2)/abs(x.initial_XGB_r2) * 100
    else:
        return (x.final_XGB_normalized_accuracy - x.initial_XGB_normalized_accuracy)/abs(x.initial_XGB_normalized_accuracy) * 100 
final_table['final_improvement_xgb'] = final_table.apply(lambda x : calculateInitialFinalImprovement(x), axis=1)
final_table['final_improvement_accuracy'] = final_table.apply(lambda x : (x.final_XGB_normalized_accuracy - x.final_NN_TORCH_normalized_accuracy)/x.final_XGB_normalized_accuracy * 100, axis=1)
final_table['final_improvement_r2'] = final_table.apply(lambda x : (x.final_XGB_r2 - x.final_NN_TORCH_r2)/x.final_XGB_r2 * 100, axis=1)
final_table


# In[131]:


final_table[(final_table['final_number_features'] >= 3) & (final_table['final_number_rows'] >= 30)]


# In[52]:


final_table.to_csv('./final_table_medium_preset.csv', index=False)


# In[71]:


import plotly.express as px
fig = px.scatter(final_table, x='final_density', y='final_improvement_xgb', size='final_number_rows')
fig.show()


# In[51]:


import plotly.express as px
fig = px.scatter(final_table, x='final_density', y='final_improvement_xgb', size='final_number_rows')
fig.show()


# In[72]:


import plotly.express as px
fig = px.scatter(final_table, x='final_density', y='final_improvement_accuracy')
fig.show()


# In[54]:


import plotly.express as px
fig = px.scatter(final_table, x='final_density', y='final_improvement_accuracy')
fig.show()


# In[53]:


fig = px.scatter(final_table, x='final_density', y='final_improvement_r2', size='final_number_rows')
fig.show()


# # Histogram plots

# In[169]:


fig = px.histogram(x=final_table['r2'].apply(lambda x: x * 100), nbins=10,
                   labels={'x':'r2'}, text_auto=True)
fig.update_layout(bargap=0.2)
fig.show()


# In[168]:


fig = px.histogram(x=final_table['accuracy'].apply(lambda x: x * 100), nbins=10,
                   labels={'x':'Accuracy'}, text_auto=True)
fig.update_layout(bargap=0.2)
fig.show()


# # Transform the binary feature (featurization)

# In[209]:


df = pd.read_csv('./dense_dfs/46911.csv')

display(df)

df_wide = df.pivot(
    index=df.drop(columns=['gender', 'churn']),
    columns='gender',
    values='churn'
)
display(df_wide)

# Rename columns
df_wide.columns = [f"churn {col}" for col in df_wide.columns]

# Flatten the index to get a clean table
df_wide = df_wide.reset_index()

df_wide


# In[210]:


binary_columns = list(filter(lambda x: 'churn' in x,  df_wide.columns.values))
df_wide[binary_columns].isnull().mean().values * 100


# In[218]:


df = pd.read_csv('./dense_dfs/46908.csv')

df = df.dropna(subset='ch_000')

display(df)

df_wide = df.pivot(
    index=df.drop(columns=['ch_000', 'AirPressureSystemFailure']),
    columns='ch_000',
    values='AirPressureSystemFailure'
)

display(df_wide)

# Rename columns
df_wide.columns = [f"AirPressureSystemFailure {col}" for col in df_wide.columns]

# Flatten the index to get a clean table
df_wide = df_wide.reset_index()

df_wide


# In[144]:


data = {
    'Batch Size': [2, 4, 2, 4, 8],
    'Hardware': ['A100', 'A100', 'A100', 'A100', 'A300'],
    'Precision': ['fp15', 'fp16', 'fp15', 'fp18', 'fp19'],
    'LLM': ['granite-13b', 'granite-13b', 'llama2-7b', 'llama2-7b', 'llama2-7b'],
    'Throughput': [1367, 2184, 2012, 2936, 3813]
}

df = pd.DataFrame(data)
display(df)

df_wide = df.pivot(
    index=['Batch Size', 'Hardware', 'Precision'],
    columns='LLM',
    values='Throughput'
)

display(df_wide)

# Rename columns
df_wide.columns = [f"Throughput {col}" for col in df_wide.columns]

# Flatten the index to get a clean table
df_wide = df_wide.reset_index()

df_wide


# In[219]:


dense_dfs_path = './dense_dfs/'

transform_binary_feature_table = pd.DataFrame(columns=['dataset_id', 'original_num_rows', 'final_num_rows', 'nan_percentage'])

tabArenaURL = 'https://raw.githubusercontent.com/TabArena/tabarena_dataset_curation/refs/heads/main/dataset_creation_scripts/metadata/tabarena_dataset_metadata.csv'
iterator = TabArenaIterator(tabArenaURL)
for row, df in iterator:
    print(row['dataset_id'])
    dense_df_path = dense_dfs_path+str(row['dataset_id'])+'.csv'

    # Check if dataset is one of the dense_dfs
    if os.path.isfile(dense_df_path):
        print('exists')
        # Read the dense_df
        dense_df = pd.read_csv(dense_df_path)

        # Find the binary feature
        binary_feature = final_table[final_table['dataset_id'] == row['dataset_id']]['binary_feature'].values[0]

        # If the dense_df has binary feature 
        if binary_feature != None:

             # In case the binary feature has nan drop those rows
            dense_df = dense_df.dropna(subset=binary_feature)

            # Featurization
            df_wide = dense_df.pivot(
                index=dense_df.drop(columns=[binary_feature, row['target_feature']]),
                columns=binary_feature,
                values=row['target_feature']
            )

            # Rename columns as 'target_feature binary_feature_val_A' and 'target_feature binary_feature_val_B'
            df_wide.columns = [f"{row['target_feature']} {col}" for col in df_wide.columns]

            # Flatten the index to get a clean table
            df_wide = df_wide.reset_index()

            # Save the transformed df
            df_wide.to_csv('dense_dfs_with_transformed_binary_feature/'+str(row['dataset_id'])+'.csv')

            # Find the transformed binary columns
            binary_columns = list(filter(lambda x: row['target_feature'] in x,  df_wide.columns.values))

            # sum the percentage of nans of two colummns
            nan_percentage = np.sum(df_wide[binary_columns].isnull().mean().values * 100) / 2

            # Append the information about the df
            transform_binary_feature_table.loc[len(transform_binary_feature_table)] = {'dataset_id': row['dataset_id'],
                                                                                        'original_num_rows' : len(dense_df),
                                                                                        'final_num_rows': len(df_wide),
                                                                                        'nan_percentage': nan_percentage}
    print()


# In[220]:


transform_binary_feature_table


# In[221]:


joined_table = pd.merge(final_table, transform_binary_feature_table, how='left', on='dataset_id')
joined_table


# In[222]:


selected_df = joined_table[['dataset_id', 'original_density', 'final_density', 'nan_percentage']]
selected_df


# # Comparison best-medium preset

# In[2]:


import pandas as pd
df_beste_preset = pd.read_csv('./final_table_best_preset.csv')[['dataset_id', 'initial_XGB_r2', 'final_XGB_r2', 'initial_XGB_normalized_accuracy', 'final_XGB_normalized_accuracy', 
                                                            'initial_NN_TORCH_r2', 'final_NN_TORCH_r2', 'initial_NN_TORCH_normalized_accuracy', 'final_NN_TORCH_normalized_accuracy']]
df_medium_preset = pd.read_csv('./final_table_medium_preset.csv')[['dataset_id', 'initial_XGB_r2', 'final_XGB_r2', 'initial_XGB_normalized_accuracy', 'final_XGB_normalized_accuracy', 
                                                            'initial_NN_TORCH_r2', 'final_NN_TORCH_r2', 'initial_NN_TORCH_normalized_accuracy', 'final_NN_TORCH_normalized_accuracy']]
df_medium_preset


# In[19]:


df_beste_preset.describe().iloc[[1]]


# In[18]:


df_medium_preset.describe().iloc[[1]]


# In[3]:


import plotly.graph_objects as go

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=df_beste_preset['dataset_id'], y=df_beste_preset['final_XGB_r2'],
                    mode='markers',
                    name='best_final_XGB_r2'))
fig.add_trace(go.Scatter(x=df_medium_preset['dataset_id'], y=df_medium_preset['final_XGB_r2'],
                    mode='markers',
                    name='medium_final_XGB_r2'))
fig.show()


# In[4]:


import plotly.graph_objects as go

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=df_beste_preset['dataset_id'], y=df_beste_preset['final_XGB_normalized_accuracy'],
                    mode='markers',
                    name='best_final_XGB_normalized_accuracy'))
fig.add_trace(go.Scatter(x=df_medium_preset['dataset_id'], y=df_medium_preset['final_XGB_normalized_accuracy'],
                    mode='markers',
                    name='medium_final_XGB_normalized_accuracy'))
fig.show()


# In[5]:


import plotly.graph_objects as go

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=df_beste_preset['dataset_id'], y=df_beste_preset['final_NN_TORCH_r2'],
                    mode='markers',
                    name='best_final_NN_TORCH_r2'))
fig.add_trace(go.Scatter(x=df_medium_preset['dataset_id'], y=df_medium_preset['final_NN_TORCH_r2'],
                    mode='markers',
                    name='medium_final_NN_TORCH_r2'))
fig.show()


# In[6]:


import plotly.graph_objects as go

fig = go.Figure()
# Add traces
fig.add_trace(go.Scatter(x=df_beste_preset['dataset_id'], y=df_beste_preset['final_NN_TORCH_normalized_accuracy'],
                    mode='markers',
                    name='best_final_NN_TORCH_normalized_accuracy'))
fig.add_trace(go.Scatter(x=df_medium_preset['dataset_id'], y=df_medium_preset['final_NN_TORCH_normalized_accuracy'],
                    mode='markers',
                    name='medium_final_NN_TORCH_normalized_accuracy'))
fig.show()

