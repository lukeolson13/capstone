# README #

#### Table of Contents
1. [Background](#background)
2. [Project Goal](#project-goal)
3. [Methods](#methods)
4. [The Data](#the-data)
5. [Customer Segmentation](#customer-segmentation)
6. [Model Selection](#model-selection)
7. [Per Item Prediction](#per-item-prediction)
8. [Store Forcasting](#store-forcasting)
9. [Store Flag](#store-flag)
10. [Future Directions](#future-directions)

## Background ##

SRP distributes and owns merchandise in thousands of convenience stores accross the nation. Since they don't actually sell the merchandise to the convience store, and it is technically sold directly to the consumer, they keep a close eye on merchandise invetories in all of the stores that carry their items. One number they track religiously is the amount of shrink for their inventory. This number is the difference between the expected inventory of an item, and the actual inventory. Sometimes there are more items than expected, but often it is the other way around. Taking human error out of the equation, this 'loss' is typically associated with item theft. Since SRP takes the loss on these items (rather than the convience store), they are interesetd in being able to predict the value of this shrink, as well as forcast this value for various regions of the US.

## Project Goal ##

The goals of this project are as follows:
1. Determine what factors influence shrink, and how these factors may be different across locations
2. Predict shrink value on a visit by visit basis, essentially better preparing salesmen for site visits
3. Forcast shrink for each store
4. Flag stores that appear to have much higher shrink compared with others in their area

## Methods ##
1. Clean and engineer large dataset in order to model shrink value
2. Pull in public data sources to better define store demographics
3. Customer segmentation on a store by store basis
4. Prediction of per item shrink value for each cluster
5. Forcasting of shrink value per store for each cluster
6. Flagging of various stores based on user-input shrink value thresholds to determine where to focus attention going forward

### The Data ###

The SRP data consists of ~4.6 million rows with 50 features each. Each row represents one visit by a salesmen to a single customer convience store with regards to a single item. Features included customer demographic information, but were mostly specific to each item and visit, such as item price, qty, visit dates(s), inventory, etc.

Public data that was considered includes:
   - Crime
   - Income
   - Population density
   - Unemployment rate
   - Food desert ratio
   
Being public data, this was far from perfect. For example, the most specific location for one of the datasets was on a county level, whereas it would've been prefered to have everything on a zip-code level. 
Another issue was with the crime data. This intuitively seems like an obvious indicator of convience store theft. While there was decent correlation, the data was very sparse, and required throwing away ~1/3 of the SRP data available. Ultimately it was decided that cost outweighed benefit, and the crime data wasn't used in the analysis.

### Customer Segmentation ###

In order to better model both predictions of item shrink value, as well as forcast shrink value for each store, customer segmentation was performed on a per store basis via the following steps:
   1. Dimensionality reduction was performed using lasso regression to determine the most important features related to a customer store's average shrink value
   2. KMeans clustering was performed with varying number of clusters
   3. Silhouette scores were calculated, leading to the ideal amount of clusters (4)

<img src="/images/Silhouette.png" width="70%">

<img src="/images/Clusters.png" width="70%">

The above image uses Principal Component Analysis to show weighted combinations of all features in 2-D space which explain the most variance. This is not neccessarily the true distribution of clusters, but is merely a way to visualize the rough distribution of clusters.

### Model Selection ###

Twelve non-tuned regression models were tested using K-Fold Cross Validation (three are not pictured):

<img src="/images/model_selection.png" width="100%">

From here, Random Forests, Gradient Boosting, and Multilayer Perceptron were further tested (via GridSearchCV) in order to determine the optimal model for this dataset. Ultimately, Multilayer Perceptron was choosen.

### Per Item Prediction ###

For the item prediction model, the goal was to be able to predict what an item's shrink value would be prior to a salesman entering the store. To do this all item level features were combined with store level features and public data. These features were then used to fit the Multilayer Perceptron model, one for each cluster.
These predictions were then compared against the actual values, and a Root-Mean-Square-Error (RMSE) calculated. This was compared against the naive RMSE, which was basically assuming the amount of shrink value for a particular item at a specific location would be the same as it was on the previous value. The two are compared below:

<img src="/images/pred_model_rmse.png" width="60%">

As you can see, the new model significantly lowered the averaged RMSE (averaged across each of the cluster models).

### Store Forcasting ###

For the store level forcasting, many of the features used in the prediction model couldn't be used. This is because many of the values are determined on the most recent visit to the store. Obviously, these values aren't known about a store 3 months in advance. What this left was store level information (including the public data), and previous visit shrink value data (the lag columns; see [this Gist](https://gist.github.com/lukeolson13/8047b3ecd54f6d7a02bdc18b8e0212c0) on how this was done). 

Running a similar test to the prediction model (just with the limited features), the forcast model was about on par with prediciting the next shrink value as the naive model: 

<img src="/images/forc_model_test.png" width="60%">

This wasn't super surprising, given the limited amount of data.

Next, future visit predictions were made, and an RMSE was again calculated off of the actual value (the test set was roughly the last month of data available) and compared to the naive approach (assuming the last visit shrink value, extrapolated into the future). Now, the forcast model was able to pick up better on trends within each store's shrink, and combine this with store demographics to come up with better predictions than the forcast model:

<img src="/images/forc_model_rmse1.png" width="60%">

<img src="/images/forc_model_rmse2.png" width="60%">

<img src="/images/forc_model_rmse3.png" width="60%">

*Note: there are blank values as the time visit periods go forward due to certain clusters of stores not having information.

### Store Flag ###

<img src="/images/flag.png" width="5%">

Given the relative success of the forcasting model, a method for flagging certain customers was developed. This essentially predicts shrink value for customers X periods into the future, and then allows a user determine what time period to look at. Then, a total shrink value for that time period is created for each customer. The user can then give the method thresholds on dollar amounts or multiples of a minimum value, and customer stores that breach this threshold compared with other stores in the same zip-code are flagged as problematic.

### Future Directions ###

Given the short timespan of this project (two weeks), there's definitely more work to do. Some future work might include:
   - Further tuning of the lag column algorithm
   - Impute the public data to fill some of the nans or look for more different sources that don't result in missing values
   - Further model tuning: I didn't choose Multilayer Perceptron because neural nets seem sexy, but because it performed the  best. While I understand the mechanics behind this model, I simply choose the hyperparameters spit out by the GridSearchCV, so I figure there is room for improvement looking at this from a model architecture standpoint
