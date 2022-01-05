# UBIWEAR

<p align="center">
  <img src="./assets/logo.png" width="250" title="UBIWEAR">
</p>

A Python library for pre-processing ubiquitous aggregated self-tracking data.

## What is this library about

This library is influenced by the work of ours in which we utilized in-the-wild data
coming from the "MyHeart Counts" study [1].

Through our time-consuming experimentation with these real-world data, we extracted 
a set of **prescriptive guidelines** of pre-processing steps related to aggregated data 
gathered from wearable devices.

We hope UBIWEAR serves as a starting point to the research community towards the unexplored 
domain of **physical activity prediction** and promote a standardized definition for pre-processing
wearables and self-tracking devices data.

## When to use this library

To the best of our knowledge since this library was written, there were no
suggested techniques to apply for handling time-series data coming from self-tracking devices.

In UBIWEAR we offer some pre-processing methods related to univariate time-series problems
with some slight modifications exclusively for wearables data.

It handles **univariate** time-series aggregated data and process the data in a structure  for predictive modeling.

## Usage of UBIWEAR

### Install the library
Create virtual environment

```
$ python3 -m venv venv
$ source venv/bin/activate
```

Upgrade pip

```
$ python -m pip install --upgrade pip
```

Install UBIWEAR

```
$ pip install ubiwear
```

### Load your data

The input to UBIWEAR is always a **pandas' DataFrame** with the index as type of `DatetimeIndex` and a column named `value` 
of type `float` or `int` with the recorded observations representing your time-series data.

For comprehension reasons we included an example of such data in the `assets/` directory in `.csv` format.

```python
import pandas as pd

df = pd.read_csv('assets/df-wearable-time-series-example.csv', index_col='startTime', parse_dates=True)
```

The `df` must have the following format like in the example:
```
                     value
startTime                 
2015-08-07 05:37:31   59.0
2015-08-07 05:43:31  139.0
2015-08-07 07:06:16  245.0
2015-08-07 07:11:18  148.0
2015-08-07 07:15:49   43.0
                    ...
2015-08-25 04:52:35   18.0
2015-08-25 05:03:11   15.0
2015-08-25 05:04:51   44.0
2015-08-25 05:06:13   80.0
2015-08-25 05:41:19  112.0
```

### Clean and process the data

Import the `Processor` class. Its' purpose is to pre-process time-series aggregated wearable data. 

The available methods of the class should be used in a chaining style. 

It also offers a "magic" method `process` that processes the data in a pre-defined suggested pipeline, 
that works especially for physical activity data.

```python
from ubiwear.processor import Processor

ubiwear_processor = Processor(df=df)

# Call the magic method
df = ubiwear_processor.process(granularity='1H', q=0.05, impute_start=8, impute_end=24)
```

The `df` has the following format:

```
                          value  dayofweek_sin  ...  hour_sin      hour_cos
startTime                                       ...                        
2015-08-07 05:00:00  198.000000      -0.433884  ...  0.965926  2.588190e-01
2015-08-07 06:00:00    0.000000      -0.433884  ...  1.000000  6.123234e-17
2015-08-07 07:00:00  467.000000      -0.433884  ...  0.965926 -2.588190e-01
2015-08-07 08:00:00  544.333333      -0.433884  ...  0.866025 -5.000000e-01
2015-08-07 09:00:00  621.666667      -0.433884  ...  0.707107 -7.071068e-01
                         ...            ...  ...       ...           ...
2015-08-25 01:00:00    0.000000       0.781831  ...  0.258819  9.659258e-01
2015-08-25 02:00:00   82.000000       0.781831  ...  0.500000  8.660254e-01
2015-08-25 03:00:00    0.000000       0.781831  ...  0.707107  7.071068e-01
2015-08-25 04:00:00    0.000000       0.781831  ...  0.866025  5.000000e-01
2015-08-25 05:00:00   95.000000       0.781831  ...  0.965926  2.588190e-01
```

What has happened ?

* removed duplicate observations related to time-series examples.
* removed NaN/NaT records
* removed outlier values using the quantiles method
* resampled the data in a unified granularity i.e. hourly granularity
* imputed specifically for wearables' data missing values on active hours (08:00 - 24:00)
* enhanced feature space with date features and converted them into their cyclical transformation

All of the above methods can be called individually and select those that fit your problem.

You can also implement your own methods in `Processor` class and call it in your desired pre-processing
pipeline in a chaining manner.

For example:
```python
from ubiwear.processor import Processor

ubiwear_processor = Processor(df=df)

ubiwear_processor \
    .remove_nan() \
    .remove_duplicate_values_at_same_timestamp() \
    .add_date_features() \
    # ... \    
    # your_own_method()

# Get the processed data
df = ubiwear_processor.df
```

### Re-frame the problem from time-series to a supervised dataset
Use the `Window` class which provides two main functionalities that transforms a time-series problem 
to a supervised set ready to be used by machine learning algorithms.

* **Sliding window** to transform a time-series problem to a supervised
* Our novel aggregated **tumbling window**

```python
from ubiwear.window import Window

# Transform from time-series to supervised dataset for ML
window = Window(n_in=2 * 24)
dataset = window.sliding_window(data=df)

# OR aggregated tumbling window
# dataset = window.tumbling_window(data=df, freq='1D')
```

The `dataset` has the following format:

```
                     var1(t-48)  var2(t-48)  ...  var11(t)  var1(t)
startTime                                    ...                   
2015-08-09 05:00:00       198.0   -0.433884  ...  0.258819      0.0
2015-08-10 05:00:00         0.0   -0.974928  ...  0.258819      0.0
                                                    ...
2015-08-11 05:00:00         0.0   -0.781831  ...  0.258819      0.0
2015-08-22 05:00:00         0.0    0.433884  ...  0.258819      0.0
2015-08-23 05:00:00         0.0   -0.433884  ...  0.258819   4562.0
2015-08-24 05:00:00         0.0   -0.974928  ...  0.258819   1861.5
2015-08-25 05:00:00       450.0   -0.781831  ...  0.258819    177.0
```

### Convert dataset for ML

The `Dataset` is a class that provides sub-datasets for training ML models. It takes as input the dataset
created from the UBIWEAR's `Window` class.

```python
from ubiwear.dataset import Dataset

ubiwear_dataset = Dataset(dataset=dataset)

# Get train/test sub-datasets
x_train, x_test, y_train, y_test = ubiwear_dataset.get_train_test(train_ratio=0.75)

# OR train/validation/test sub-datasets
x_train, x_val, x_test, y_train, y_val, y_test = ubiwear_dataset.get_train_val_test(train_ratio=0.75, val_ratio=0.2)
```

### Apply your favorite ML or DL model

You know have clean, pre-processed and ready your well-known `X's` and `y's` for your ML problem!

You can call your favorite model, and record the performance on your favorite regression metrics.



## Literature
[1] Hershman, Steven G., et al. "Physical activity, sleep and cardiovascular health data for 50,000 individuals from the MyHeart Counts Study." Scientific data 6.1 (2019): 1-10.
