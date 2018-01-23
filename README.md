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
9. [Future Directions](#future-directions)

## Background ##

SRP distributes and owns merchandise in thousands of convenience stores accross the nation. Since they don't actually sell the merchandise to the convience store, and it is technically sold directly to the consumer, they keep a close eye on merchandise invetories in all of the stores that carry their items. One number they track religiously is the amount of shrink for their inventory. This number is the difference between the expected inventory of an item, and the actual inventory. Sometimes there are more items than expected, but often it is the other way around. Taking human error out of the equation, this 'loss' is typically associated with item theft. Since SRP takes the loss on these items (rather than the convience store), they are interesetd in being able to predict the value of this shrink, as well as forcast this value for various regions of the US.

## Project Goal ##

The goals of this project are as follows:
1. Determine what factors influence shrink, and how these factors may be different across locations
2. Predict shrink value on a visit by visit basis, essentially better preparing salesmen for site visits
3. Forcast shrink for each store
4. Flag stores that appear to have much higher shrink compared with others in their area

## Methods ##

1. Clean and engineer large dataset (>50 features X ~4.6 million rows) in order to model shrink value
2. Pull in public data sources to better define stores:
   - Crime
   - Income
   - Population density
   - Unemployment rate
   - Food desert ratio
3. Customer segmentation on a store by store basis via KMeans clustering
4. Prediction of per item shrink value using a multilayer perceptron model for each cluster
5. Forcasting of shrink value per store using a multilayer percepton model for each cluster
6. Flagging of various stores based on user-input shrink value thresholds to determine where to focus attention going forward

### The Data ###

asd;fdsf

### Customer Segmentation ###

<img src="/images/Clusters.png" width="70%">

### Model Selection ###

<img src="/images/model_selection.png" width="70%">

### Per Item Prediction ###

dsfds

### Store Forcasting ###

Ddfdfs

### Future Directions ###

sdfsd
