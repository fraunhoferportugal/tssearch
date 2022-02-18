[![license](https://img.shields.io/badge/License-BSD%203-brightgreen)](https://github.com/fraunhoferportugal/tssearch/blob/master/LICENSE.txt)
[![Documentation Status](https://readthedocs.org/projects/tssearch/badge/?version=latest)](https://tssearch.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tssearch)
![PyPI](https://img.shields.io/pypi/v/tssearch)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fraunhoferportugal/tssearch/blob/master/notebooks/Query_search_unidimensional.ipynb)

# Time Series Subsequence Search Library

## Intuitive time series subsequence search
This repository hosts the **TSSEARCH - Time Series Subsequence Search** Python package. TSSEARCH assists researchers in exploratory analysis for query search and time series segmentation without requiring significant programming effort.

## Functionalities

* **Search**: We provide methods for time series query search and segmentation
* **Weights**: The relative contribution of each point of the query to the overall distance can be expressed using a user-defined weight vector. 
* **Visualization**: We provide visualizations to present the results of the
segmentation and query search
* **Unit tested**: we provide unit tests for each feature
* **Easily extended**: adding new distances is easy, and we encourage you to contribute with your custom distances or search methods

## Get started

### Installation
TSSEARCH is available at PyPI:

```python
pip install tssearch
```

### Example
The code below segments a 10 s electrocardiography record:

```python
import tssearch

# Load the query, (optional) weight vector and sequence
data = tssearch.load_ecg_example()

# Selects the Dynamic Time Warping (DTW) as the distance for the segmentation
cfg = tssearch.get_distance_dict(["Dynamic Time Warping"])

# Performs the segmentation
out = tssearch.time_series_segmentation(cfg, data['query'], data['sequence'], data['weight'])
```

### Documentation
The documentation is available [here](https://tssearch.readthedocs.io/en/latest/).

## Available distances

| Lockstep                             |
|--------------------------------------|
| Lp Distances                         |
| Pearson Correlation Distance         |
| Short Time Series Distance (STS)     |

| Elastic                              |
|--------------------------------------|
| Dynamic Time Warping (DTW)           |
|Longest Common Subsequence (LCSS)     |
|Time Warp Edit Distance (TWED)        |

| Time                                 |
|--------------------------------------|
| Time Alignment Measurement (TAM)     |


## Acknowledgements
This work was supported by the project Geolocation non-Assisted by GPS for Mobile Networks in Indoor and Outdoor Environment (GARMIO), co-funded by Portugal 2020, framed through the COMPETE 2020 (Operational Programme Competitiveness and Internationalization) and European Regional Development Fund (ERDF) from European Union (EU) under Grant POCI-01-0247-FEDER-033479.

