from typing import List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
import os

from timesead.utils.sys_utils import check_path
from timesead.utils.metadata import RESOURCE_DIRECTORY


def plot_error_bars(means : list, deviations : list, ax : Optional[plt.Axes] = None, offset : int = 0, **kwargs):

    if not ax:
        ax = plt.gca()

    ax.errorbar([_ + offset for _ in range(len(means))], means, deviations, **kwargs)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_ticks([])


def plot_histogram(data : Optional[List[int]] = None, resolution : int = 100,
                   yticks : Optional[Union[int, List]] = None, ax : Optional[mplt.axes.Axes] = None,
                   hist_range : Tuple[int, int] = (0, 1), xticks : Optional[List[int]] = None , **kwargs):

    if not ax:
        ax = plt.gca()

    ax.hist(data, resolution, hist_range, histtype='stepfilled', **kwargs)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if xticks:
        ax.get_xaxis().set_ticks(xticks)
    else:
        ax.get_xaxis().set_ticks([])

    if isinstance(yticks, int):
        plt.locator_params(axis='y', nbins=yticks)
    elif isinstance(yticks, list):
        ax.get_xaxis().set_ticks(yticks)
    else:

        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_ticks([])


def plot_sequence_against_anomaly(values : List[float], targets : List[float], ax : Optional[mplt.axes.Axes] = None,
                                  xticks : Optional[Union[int, List]] = None, scatter : bool = True,
                                  yticks : bool = True):
    """Plot 1D sequence against anomaly windows.

    :param values: Sequence of values to plot against anomalies.
    :type values: List[float]
    :param targets: List of labels taking values in {0,1} of the same length as values.
    :type targets: List[float]
    :param ax: Axes object to use for plotting. Uses current Axes if None.
    :type ax: Optional[mplt.axes.Axes]
    :param xticks: Generates <xticks> xticks if int. Uses xticks directly if List. No xticks if None.
    :type xticks: Optional[Union[int, List]]
    :param scatter: Draw values as line plot or scatter plot.
    :type scatter: bool
    :param yticks: Set yticks.
    :type yticks: bool
    """

    if not ax:
        ax = plt.gca()

    assert len(values) == len(targets)

    # Plot anomalies
    boundaries = 2 * np.array(targets + [0]) - np.array([0] + targets)

    lbs = np.argwhere(boundaries > 1).flatten()
    rbs = np.argwhere(boundaries < 0).flatten()

    for lb, rb in zip(lbs, rbs):
        ax.axvspan(lb - 0.5, rb + 0.5, facecolor='r', alpha=0.5)

    # Plot values
    if scatter:
        ax.scatter(list(range(len(values))), values)
    else:
        ax.plot(list(range(len(values))), values)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if yticks:

        yticks = [round(min(values), 2), round(max(values), 2)]
        yticks = [int(tick) if tick.is_integer() else tick for tick in yticks]

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)

    else:

        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_ticks([])

    if isinstance(xticks, int):
        plt.locator_params(axis='x', nbins=xticks)
    elif isinstance(xticks, list):
        ax.get_xaxis().set_ticks(xticks)
    else:

        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_ticks([])


def save_plot(path : str):
    """Save the current figure and close it.

    :param path: Path to save the figure to.
    :type path: str
    """

    check_path(os.path.dirname(path))

    plt.savefig(path, bbox_inches='tight')
    plt.close()


def set_style(stylefile : str = os.path.join(RESOURCE_DIRECTORY, 'style', 'timesead.mplstyle')):
    """Sets the rcParmaters of matplotlib from a style file.

    :param stylefile: Style file in mplstyle format.
    :type stylefile: str
    """

    plt.style.use(stylefile)

