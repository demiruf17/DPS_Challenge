# DPS_Challenge

## data_model.py ##

The DataModel class prepares the dataset in the desired format and encodes the categories and types for training. 

## visualization.ipynb ##

It visualizes historically the number of accidents per category and type. 

![alt text](https://github.com/demiruf17/DPS_Challenge/blob/main/assets/accidents_per_category.png)

![alt text](https://github.com/demiruf17/DPS_Challenge/blob/main/assets/accidents_per_type.png)

## train.py ##

This script trains and tests several regression models. It lists the root mean squared error (RMSE) of the models. The model that has the lowest error is selected to be used in the web service. 


## app.py ##

This is the main file for the web service. It takes input from a user and returns the value in JSON format. The request must contain year and month fields. The result is the number of accidents that happened in a given year and month for all categories and types.


## How It Works ##

The body of the request should be in JSON format, like below:

```
{
"year":2013,
"month":8
}
```

The user should send a request to the endpoint given below:

https://ufuk-dps-ai-challenge.herokuapp.com/result

Here is an example response:
```
{
    "prediction": 6319.215290905911
}
```
