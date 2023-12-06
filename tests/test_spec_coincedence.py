import numpy as np
import matplotlib.pyplot as plt
from agepy.spec import coincidence

def test_plot_coinc_map():
    # AI generated test. Think about more meaningful tests.
    # Test data
    coincmap = np.random.rand(10, 10)
    xedges = np.linspace(0, 1, 11)
    yedges = np.linspace(0, 1, 11)
    figsize = (8, 6)
    cmap = 'YlOrRd'
    title = "Coincidence Map"
    norm = None
    vmin = 1
    vmax = None
    num = 1
    xlabel = "early electron kinetic energy"
    ylabel = "late electron kinetic energy"

    # Call the function
    fig, (ax_coinc, ax_x, ax_y) = coincidence.plot_coinc_map(
        coincmap, xedges, yedges, figsize=figsize, cmap=cmap, title=title,
        norm=norm, vmin=vmin, vmax=vmax, num=num, xlabel=xlabel, ylabel=ylabel
    )

    # Assertions
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax_coinc, plt.Axes)
    assert isinstance(ax_x, plt.Axes)
    assert isinstance(ax_y, plt.Axes)

    # Cleanup
    plt.close(fig)