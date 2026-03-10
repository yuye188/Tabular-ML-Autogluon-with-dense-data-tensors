import pandas as pd
import openml

class TabArenaIterator:

    def __init__(self, source):
        self.df_source = pd.read_csv(source)
        self.idx = 0


    def __iter__(self):
        return self

    def __next__(self):
        
        if self.idx >= len(self.df_source):            
            raise StopIteration
        
        dataset_row = self.df_source.iloc[self.idx]
        dataset, y, categorical_indicator, attribute_names = openml.datasets.get_dataset(int(dataset_row.dataset_id)).get_data(dataset_format="dataframe")
        self.idx = self.idx + 1

        return (dataset_row, dataset)