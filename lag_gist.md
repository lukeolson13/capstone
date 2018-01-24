## Create Lag Columns in Pandas DataFrame via Hierarchal Column Filtering

What is a 'lag' column?
  - A lag column (in this context), is a column of values that references another column a values, just at a different time period.

The problem to be addressed:
  - Normally, creating lag columns in pandas is as simple as df.shift(x), which allows you to shift your index by x. However, this only works if your dataframe is already filtered down on the values you wish to shift it by, or if your dataframe doesn't need any filtering. These functions attempt to address that column by allowing for filtering dataframes down by one or more columns before shifting them and creating new lag values.

What these functions do:
  - Take a Pandas dataframe and create lag column on any column based on a single date column
  - Allow for filtering down on multiple hierarchal levels (ie state -> region -> store -> item)
  - Create lag values for the last column filtered on
  - Allow the user to enter in how many time periods they would like to look back. These are not exact dates, but simply one time step back according to your data

What these functions don't do:
  - Attempt to create lag values on the last column, and if there's not enough data, move up to a more general filter in order to create a lag value (I was hoping to get to this had I had more time)
  - Allow the user to set a date in the past for the lag column to reference

A word of caution:
  - I'm not a python developer, so I can assure you that this code is far from perfect
  - This can be relatively time consuming on large datasets (>100,000 rows)
  - I didn't make this into a class because it's not a one size fits all solution. This code is meant to be more of a reference in order to help someone else with a similar problem

Here's an example (and why I wrote this in the first place):
  - In trying to predict shrink (essentially stolen goods) for a large number of convenience stores (where a timeseries wasn't applicable due to a limited time range of data), I wanted to create lag columns for shrink at the stores on previous dates to use in the model (this is an example of using your previous target variables in combination with your features to predict your target variable at the current time stamp). Since data was sparse for some stores (ie only one or two timestamps), I wanted to try creating lag columns on different hierarchal levels:
    - By store
    - By item category
    - By specific item
  If I wanted to create lag columns by item category, I needed to first filter my dataframe by each store, and then look at each of the item categories within that store. Likewise, I could filter down further by specific item if I would like. This is the point of the recursive function. This is how I would execute this (only need to call one function):
    - lag(df=my_df, num_periods=3, col_filters=['store', 'item_category', 'item'], date_col='visit_date', lag_vars=['shrink'], col_name_suf='')
      - Setting num_periods=3 would result in 3 new columns for every lag variable chosen (one for each time period back)
      - col_filters is order specific, with the most general filtering level first, and the most specific (that you will actually be referencing in your new lag columns) last
      - The date_col is what is used to sort your filtered dataframe in order to find the previous occurrences
      - Lag_var is/are the column(s) you wish to create lag columns for. In this case, I want a column for previous shrink values
      -col_name_suf allows you to name your new lag columns differently. I used this to distinguish between store, item_category, and item level filtering (would have to run this 3 times to achieve that)

```python
import numpy as np

def init_nans(df, num_periods, lag_vars, col_name_suf):
    '''
    Initiate column(s) of nan values for your lag columns in the given dataframe
    Inputs:
        df - dataframe
        num_periods - number of periods (previous dates) to go back and attempt to fill lag values for
        lag_vars - columns of values to create lag columns with
        col_name_suf - suffix to append to newly created columns (help to distinguish between last filtered column choosen)
    '''
    for period in range(1, num_periods + 1):
        for lag_var in lag_vars:
            df['{}_lag{}{}'.format(lag_var, period, col_name_suf)] = np.nan

def set_lag_vals(df, comb_mask, date_col, lag_vars, num_periods, col_name_suf):
    '''
    Sets lag values according to the last column in the heirarchy or columns
    Inputs:
        df - dataframe
        comb_mask - combined mask for current value combinations between columns. Used to filter dataframe
        date_col - date column to use in grouping and lag periods
        lag_vars - columns of values to create lag columns with
        num_periods - number of periods (previous dates) to go back and attempt to fill lag values for
        col_name_suf - suffix to append to newly created columns (help to distinguish between last filtered column choosen)
    '''
    foo = df[ comb_mask ].sort_values(date_col, ascending=False)     
    length = len(foo[date_col].unique()) # determine number of visits (because multiple item categories can be updated in a single visit)
    for period in range(1, num_periods + 1):
        # skip if there's not enough data to create lag columns
        if length < period + 1:
            continue
        i = 0
        # create duplicate df, but with all indices shifted by the current 'period' number
        foo_shifted = foo.shift(-period)
        foo_grouped = foo.groupby(date_col).mean()
        for index, row in foo.iterrows():
            date = foo_shifted[ foo_shifted.index == index ][date_col].values[0]
            for lag_var in lag_vars:
                lag_val = foo_grouped[ foo_grouped.index == date ][lag_var].values[0]
                # set value
                df.set_value(index, '{}_lag{}{}'.format(lag_var, period, col_name_suf), lag_val)
            i += 1
            if i + period == length:
                break # back to period loop

def lag_rec(df, num_periods, col_filters, date_col, lag_vars, col_name_suf, mask=True):
    '''
    Recursively loop through various heirarchaly ordered columns, grouping by date
    INPUTS:
        df - pandas dataframe
        num_periods - number of periods (previous dates) to go back and attempt to fill lag values for
        col_filters - columns to heiracrchally filter down on, with the last column being the one ultimately used
        date_col - date column to use in grouping and lag periods
        lag_vars - columns of values to create lag columns with
        col_name_suf - suffix to append to newly created columns (help to distinguish between last filtered column choosen)
        mask - DO NOT CHANGE. Required to be True to maintain dataframe mask between recursive iterations
    '''
    # begin with mask of all trues
    true_mask = np.ones(len(df), dtype=bool)
    loop_mask = mask & true_mask
    col_filter = col_filters[0]
    for val in df[ loop_mask ][col_filter].unique():
        val_mask = df[col_filter] == val
        comb_mask = loop_mask & val_mask

        if len(col_filters) > 1:
            #recursively update the remaining items' positions
            lag_rec(df, num_periods, col_filters[1:], date_col, lag_vars, col_name_suf, mask=comb_mask)
        else:
            set_lag_vals(df, comb_mask, date_col, lag_vars, num_periods, col_name_suf)

def lag(df, num_periods, col_filters, date_col, lag_vars, col_name_suf):
    '''
    Wrapper function to execute init_nans and lag_rec.
    Inputs:
        df - dataframe
        num_periods - number of periods (previous dates) to go back and attempt to fill lag values for
        col_filters - columns to heiracrchally filter down on, with the last column being the one ultimately used
        date_col - date column to use in grouping and lag periods
        lag_vars - columns of values to create lag columns with
        col_name_suf - suffix to append to newly created columns (help to distinguish between last filtered column choosen)
    Returns:
        Updated dataframe containing new lag columns
    '''
    init_nans(df, num_periods, lag_vars, col_name_suf)
    lag_rec(df, num_periods, col_filters, date_filter, lag_vars, col_name_suf)
    return df
```
