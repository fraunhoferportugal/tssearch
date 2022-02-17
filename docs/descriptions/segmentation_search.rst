=======================
Segmentation and Search
=======================

************
Segmentation
************


The :class:`~tssearch.search.segmentation.time_series_segmentation` locates the time instants between consecutive query repetitions on a longer and repetitive sequence.
You will need to define the distance used for segmentation and provide a query and a sequence as inputs to :class:`~tssearch.search.segmentation.time_series_segmentation`, as follows:

.. code:: python

    import tssearch
    import numpy as np

    query, weights, sequence = tssearch.examples.get_ecg_example_data()
    cfg = tssearch.get_distance_dict(["Dynamic Time Warping"])

    out = tssearch.time_series_segmentation(cfg, query, weights, sequence)

In the code above a ten-second segment from an electrocardiography record is used to define the query and the sequence and the DTW is defined as the distance for the segmentation. Then, the segmentation is calculated and the output is assigned to a variable. The method receives as inputs the configuration file, the query, and the sequence. Additionally, an optional vector input that assigns weights for each time instance of the query is also given as input.

.. image:: https://i.postimg.cc/4yfGJJVB/Fig-4-1.png
  :alt: Example ECG segmentation output

In this example, the specified weights vector assigned less contribution to the second local maxima of the ECG (T wave).

If you are interested in further characterizing each subsequence, this could be accomplished using the distances values calculated for each segment and/or using `TSFEL
<https://github.com/fraunhoferportugal/tsfel/>`_ to extract temporal, statistical, and spectral features as data representations for classification algorithms.

******
Search
******

The :class:`~tssearch.search.query_search.time_series_search` method locates the k-best occurrences of a given query on a longer sequence based on a distance measurement. By default, k is set to retrieve the maximum number of matches. The user can also explicitly define the value of k to retrieve the k-best occurrences.

An illustrative example is provided below:

.. code:: python

    import tssearch
    import numpy as np

    query = np.loadtxt("query.txt")
    sequence = np.loadtxt("sequence.txt")

    cfg = tssearch.get_distance_dict(["Dynamic Time Warping"])
    cfg['elastic']['Dynamic Time Warping']['parameters']['alpha'] = 0.5

    out = tssearch.time_series_search(cfg, query, sequence)

In the above code, the DTW with an additional parameter :math:`{\alpha}` that weights the contribution between the cost in the amplitude and its first derivative is defined. Then, the query search is calculated, and the output is assigned to a variable. The method receives as inputs the configuration file, the query, and the sequence. Since the number of matches is not defined, the method retrieves the maximum number of matches.

To illustrate this example, a wearable sensor-based human activity dataset with multidimensional data was used and the following visualization was obtained:

.. image:: https://i.postimg.cc/rmrp3Fcb/Fig-6-1.png
  :alt: Example of query search in stride segmentation