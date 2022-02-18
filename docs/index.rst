Welcome to TSSEARCH documentation!
==================================

Time Series Subsequence Search Python package (TSSEARCH for short) is a Python package that assists researchers in exploratory analysis for query search and time series segmentation without requiring significant programming effort. It contains curated routines for query and subsequence search. TSSEARCH installation is straightforward and goes along with startup code examples. Our goal is to provide the tools to get faster insights for your time series.

Highlights
==========

- **Search**: we provide methods for time series query search and segmentation
- **Weights**: the relative contribution of each point of the query to the overall distance can be expressed using a user-defined weight vector
- **Visualization**: we provide visualizations to present the results of the segmentation and query search
- **Unit tested**: we provide unit tests for each distance
- **Easily extended**: adding new distances is easy, and we encourage you to contribute with your custom distances or search methods

Contents
========

In development

.. toctree::
   :maxdepth: 2

   Lockstep Distances <descriptions/lockstep_distances>
   Elastic Distances <descriptions/elastic_distances>
   Segmentation and Search <descriptions/segmentation_search>
   Module Reference <descriptions/modules>
   Authors <authors>
   Changelog <changelog>
   License <license>

Installation
============

This packages is available on PyPI:

.. code:: bash

    $ pip install tssearch

Get started
===========

The code below segments a 10 s electrocardiography record:

.. code:: python

    import tssearch

    # Load the query, (optional) weight vector and sequence
    data = tssearch.load_ecg_example()

    # Selects the Dynamic Time Warping (DTW) as the distance for the segmentation
    cfg = tssearch.get_distance_dict(["Dynamic Time Warping"])

    # Performs the segmentation
    out = tssearch.time_series_segmentation(cfg, data['query'], data['sequence'], data['weight'])

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
