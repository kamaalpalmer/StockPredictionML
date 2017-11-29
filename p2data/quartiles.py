import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_name = "dow_jones_index.data.csv" 
#f = open(file_name)
#data = pd.loadtxt(fname=f, delimiter = ',')

# these come from reading the names file; could be first line in some datasets
cols = ['quarter', 'open', 'high', 'low', 'close', 'volume',
        'percent_change_price', 'percent_change_volume_over_last_wk', 'previous_weeks_volume', 'next_weeks_open', 'next_weeks_close', 'days_to_next_dividend','percent_return_next_dividend','percent_change_next_weeks_price']

# this adds the column headers to the data frame while reading in the data
data = pd.read_csv(file_name, names=cols)

# use the column for pregnancies, make three bins, use numbers, not labels
newcol = pd.qcut(data['percent_change_next_weeks_price'], 3, labels=False)
print(newcol)

data['q_percent_change_next_weeks_price'] = newcol
print(data)

