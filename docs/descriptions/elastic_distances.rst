=================
Elastic Distances
=================

Distance measures that perform a non-linear mapping to align the time series and allow flexible comparison of one-to-many or one-to-none points (e.g., Dynamic Time Warping, Longest Common Subsequence). These measures produce elastic adjustment to compensate for potential localized misalignment.

**************************
Dynamic Time Warping (DTW)
**************************

DTW Description

*********************************
Longest Common Subsequence (LCSS)
*********************************

The Longest Common Subsequence (LCSS) measures the similarity between two time series whose lengths might be different. Since it is formulated based on edit distances,  gaps or unmatched regions are permitted and they are penalized with a value proportional to their length. It can be useful to identify similarities between time series whose lengths differ greatly or have noise [1]_.

In the example below, we compute the LCSS alignment between two time series, one of them with added noise.

.. code:: python

    import tssearch
    import numpy as np

    s1 = np.sin(np.arange(0, 4*np.pi, 0.1))
    noise = np.random.normal(0, 0.1, s1.shape)
    s2 = 1 + np.sin(np.arange(0, 4*np.pi, 0.1) + 2) + noise

    s1 = s1.reshape(-1, 1)
    s2 = s2.reshape(-1, 1)
    ac = tssearch.lcss_accumulated_matrix(x=s1, y=s2, eps=1.5)
    lcss_path = tssearch.lcss_path(x=s1, y=s2, ac, eps=1.5)

    plt.plot(s1, "b-", label='First time series')
    plt.plot(s2, "g-", label='Second time series')
    x=lcss_path[0]
    y=lcss_path[1]
    for i, j in zip(x,y):
        plt.plot([i, j], [s1[i], s2[j]], color='orange')
    plt.legend()
    plt.title("Time series matching with LCSS")


.. image:: https://i.postimg.cc/28XbZ9k8/lcss.png
   :alt: An example of LCSS.


******************************
Time Warp Edit Distance (TWED)
******************************

Time warp edit distance (TWED) uses sequences’ samples indexes/timestamps difference to linearly penalize the matching of samples for which indexes/timestamps values are too far and to favor the matching samples for which indexes/timestamps values are closed. Contrarily to other elastic measures, TWED entails a time shift tolerance controlled by the stiffness parameter of the measure. Moreover, it involves a second parameter defining a constant penalty for insert or delete operations. If stiffness > 0, TWED is a distance (i.e., verifies the triangle inequality) in both space and time [2]_.

TWED has been used in time series classification assessing classification performance while varying TWED input parameters [2]_, [3]_. In the example, we calculate TWED between two time series varying its parameters.

.. image:: https://i.postimg.cc/26WZhNcQ/data.png
  :alt: Two example series

.. image:: https://i.postimg.cc/rsQFthkG/twed.png
  :alt: Resulting TWED distances


.. [1] M. Vlachos, G. Kollios and D. Gunopulos, "Discovering similar multidimensional trajectories," Proceedings 18th International Conference on Data Engineering, 2002, pp. 673-684, doi: 10.1109/ICDE.2002.994784.

.. [2] P. Marteau, "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 31, no. 2, pp. 306-318, Feb. 2009, doi: 10.1109/TPAMI.2008.76.

.. [3] Joan Serrà, Josep Ll. Arcos, An empirical evaluation of similarity measures for time series classification, Knowledge-Based Systems, Volume 67, 2014, Pages 305-314, ISSN 0950-7051, https://doi.org/10.1016/j.knosys.2014.04.035.




