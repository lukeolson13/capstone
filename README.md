# README #

#### Table of Contents
1. [Background](#background)
2. [Project Goal](#project-goal)
3. [Methods](#methods)
4. [The Data](#the-data)
5. [Customer Segmentation](#customer-segmentation)
6. [Model Selection](#model-selection)
7. [Per Item Prediction](#per-item-prediction)
8. [Store Forecasting](#store-forecasting)
9. [Store Flag](#store-flag)
10. [Future Directions](#future-directions)

## Background ##

SRP distributes and owns merchandise in thousands of convenience stores across the nation. Since they don't actually sell the merchandise to the convenience store, and it is technically sold directly to the consumer, they keep a close eye on merchandise inventories in all of the stores that carry their items. One number they track religiously is the amount of shrink for their inventory. This number is the difference between the expected inventory of an item, and the actual inventory. Sometimes there are more items than expected, but often it is the other way around. Taking accounting/human error out of the equation, this 'loss' is typically associated with item theft. Since SRP takes the loss on these items (rather than the convenience store), they are interested in being able to predict the value of this shrink, as well as forecast this value for each of their customers.

## Project Goal ##

The goals of this project are as follows:
1. Determine what factors influence shrink, and how these factors may be different across locations
2. Predict shrink value on a visit by visit basis, better preparing salesmen for site visits
3. Forecast shrink for each store

## Methods ##
1. Clean and engineer large dataset in order to model shrink value
2. Pull in public data sources to better define store demographics
3. Segment customers on a store by store basis
4. Prediction of per item shrink value for each cluster
5. Forecasting of shrink value per store for each cluster
6. Flagging of various stores based on user-input shrink value thresholds to determine where to focus attention going forward

<p align="center">
<img src="/images/data_pipeline.png" width="70%">
</p>

### The Data ###

The SRP data consists of ~4.6 million rows with 50 features each. Each row represents one visit by a salesmen to a single customer convenience store with regards to a single item. Features included customer demographic information, but were mostly specific to each item and visit, such as item price, quantity, visit dates(s), inventory, etc.

Public data that was considered includes:
   - Crime
   - Income
   - Population density
   - Unemployment rate
   - Food desert ratio
   
Being public data, this was far from perfect. For example, the most specific location for one of the datasets was on a county level, whereas it would've been preferred to have everything on a zip-code level. 
Another issue was with the crime data. This intuitively seems like an obvious indicator of convenience store theft. While there was decent correlation, the data was very sparse, and required throwing away ~2/3 of the SRP data available. Ultimately it was decided that cost outweighed benefit, and the crime data wasn't used in the analysis.

### Customer Segmentation ###

In order to better model both predictions of item shrink value, as well as forecast shrink value for each store, customer segmentation was performed on a per store basis via the following steps:
   1. Dimensionality reduction was performed using lasso regression to determine the most important features related to a customer store's average shrink value
   2. KMeans clustering was performed with varying number of clusters
   3. Silhouette scores were calculated, leading to the ideal amount of clusters (4)

<p align="center">
<img src="/images/Silhouette.png" width="70%">
</p>

<p align="center">
<img src="/images/cluster.png" width="70%">
</p>

The above image uses Principal Component Analysis to show weighted combinations of all features in 2-D space which explain the most variance. This is not necessarily the true distribution of clusters, but is merely a way to visualize the rough distribution of clusters.

<p align="center">
<img src="/images/nyc.png" width="70%">
</p>

### Model Selection ###

Twelve untuned regression models were tested using K-Fold Cross Validation (three are not pictured), and their negative mean absolute errors were compared:

<p align="center">
<img src="/images/model_selection.png" width="100%">
</p>

From here, Random Forests, Gradient Boosting, and Multilayer Perceptron were further tested (via GridSearchCV) in order to determine the optimal model for this dataset. Ultimately, Multilayer Perceptron was chosen (a vanilla neural net).

### Per Item Prediction ###

For the item prediction model, the goal was to be able to predict what an item's shrink would be prior to a salesman entering the store. To do this, all item level features were combined with store level features and public data. These features were then used to fit the multilayer perceptron model, one per cluster.
These predictions were then compared against the actual values, and a Root-Mean-Square Error (RMSE) was calculated. This was compared against the naive RMSE, which assumed that the amount of shrink for a particular item at a specific location would be the same as it was on the previous visit. The two are compared below:

<p align="center">
<img src="/images/pred_model_result.png" width="50%">
</p>

<p align="center">
<img src="/images/clust_color_map.png" width="20%">
</p>

As you can see, the new model lowered the RMSE of each cluster, resulting in an overall **27.0% reduction in RMSE**.

### Store Forecasting ###

For the store level forecasting, many of the features used in the prediction model couldn't be used. This is because many of the values are determined on the most recent visit to the store. Obviously, these values aren't known about a store 3 months in advance. What this left was store level information (including the public data), and previous visit shrink value data (the lag columns; see [this Gist](https://gist.github.com/lukeolson13/8047b3ecd54f6d7a02bdc18b8e0212c0) on how this was done). 

Running a similar test to the prediction model (just with the limited features), the forecast model was about on par (actually slightly worse) with predicting the next shrink value as the naive model: 

<p align="center">
<img src="/images/TrainingForecast.png" width="50%">
</p>

<p align="center">
<img src="/images/clust_color_map.png" width="20%">
</p>

This wasn't super surprising, given the limited amount of data.

Next, future visit predictions were made, and an RMSE was again calculated off of the actual value and compared to the naive approach (last visit shrink value extrapolated out). Now, it appears the forecast model was able to pick up better on trends within each store's shrink via the lag columns, and combine this with store demographics to come up with a more accurate forecast than the naive model:

<p align="center">
<img src="/images/1TimePeriod(s)Forward.png" width="50%">
</p>

<p align="center">
<img src="/images/2TimePeriod(s)Forward.png" width="50%">
</p>

<p align="center">
<img src="/images/3TimePeriod(s)Forward.png" width="50%">
</p>

<p align="center">
<img src="/images/4TimePeriod(s)Forward.png" width="50%">
</p>

Here, a single "time period" refers to the next expected visit by a salesman to a store. This value averages 28 days/store, with a standard deviation of 16 days/store. In this case, the test set allowed for forecasting of 16 weeks (all 12 "time periods" across these 16 weeks can be seen in the "images" folder).

On average, the forecast models were able to outperform the naive models by **41.5%** (average drop in RMSE of 0.222 $/day/store). This equates to roughly **$80,000 per month** across all stores.

### Store Flag ###

Given the relative success of the forecasting model, a method for flagging certain customers was developed. This essentially predicts shrink value for all customers a set number of periods into the future, and then allows a user determine what time period to look at (ie all of May 2018). Then, a total shrink value for that time period is created for each customer. The user can then give the method thresholds on dollar amounts or multiples of a minimum value, and customer stores that breach this threshold compared with other stores in the same area are flagged as problematic.

### Future Directions ###

Given the short timespan of this project (two weeks), there's definitely more work to do. Some future work might include:
   - Considering the bias the algorithm may create by segmenting customers by specific demographics, such as race
   - Further tuning of the lag column algorithm
   - Impute the public data to fill some of the nans or look for more different sources that don't result in missing values
   - Further model tuning: Multilayer Perceptron wasn't chosen because neural nets are the talk of the town, but because it performed the  best. The hyperparameters used were determined by the GridSearchCV, so there is probably room for improvement by further tuning the grid search or looking at this from a model architecture standpoint (ie there should be X hidden layers of size Y because...)
