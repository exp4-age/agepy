import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from matplotlib.colors import LogNorm


def plot_coinc_map(num, coincmap, range_x, range_y, vrange=(1, None),
                   cmap='YlOrRd',
                   xlabel = 'early electron kinetic energy / eV',
                   ylabel = 'late electron kinetic energy / eV',
                   origin = 'lower',
                   title = None,
                   norm = None,
                   save = None
                   ):
    fig = plt.figure(num=num, figsize=(2,2), clear=True)
    # fig.clf()
    
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[3, 1],
                           height_ratios=[1, 3]
                           )                     # grid with columns=2, row=2
    gs.update(wspace=0.05, hspace=0.05)          # distance between subplots in gridspace
    ax_coinc = plt.subplot(gs[2])                # coinc matrix is subplot 2: lower left
    ax_x = plt.subplot(gs[0], sharex=ax_coinc)   # spectrum of E0 is subplot 0: upper left
    ax_y = plt.subplot(gs[3], sharey=ax_coinc)   # spectrum of E1 is subplot 3: lower right
    ax_cb = plt.subplot(gs[1])                   # colorbar is subplot 1: upper right
    
    bins_y = coincmap[:,0].size
    bins_x = coincmap[0,:].size
    vmin, vmax = vrange
    start_x, end_x = range_x
    start_y, end_y = range_y
    
    ###### sum spectrum of E0 (top)
    hist_x = np.sum(coincmap, axis=0)
    #ax_x.set_xlim([start_x, end_x])
    ax_x.plot(np.linspace(start_x, end_x, bins_x), hist_x, 'k',
              drawstyle='steps')
    ax_x.set_title(title) 
    
    
    ###### sum spectrum of E1 (right)
    hist_y = np.sum(coincmap, axis=1)
    #ax_y.set_ylim([start_y, end_y])
    ax_y.plot(hist_y, np.linspace(start_y, end_y, bins_y), 'k',
              drawstyle='steps')  # plots hist_y vertically not horizontally
     
    
    ###### coinc matrix
    if origin == 'lower':
        extent = (start_x ,end_x, start_y, end_y)
    else:
        extent= (start_x ,end_x, end_y, start_y)
    if norm == 'log':
        norm = LogNorm(vmin=vmin,vmax=vmax)
    else:
        norm = None
    acimg = ax_coinc.imshow(coincmap, cmap=cmap,
                            norm=norm, 
                            origin=origin,
                            aspect='auto',
                            extent=extent,
                            interpolation='None'
                            )
                   
    fig.colorbar(acimg, cax=ax_cb)  # generates a colorbar for the histogram in the upper right panel
    ax_cb.set_aspect(7)             # resizes the colorbar so it does not fill out the full panel
    
    ax_coinc.set_xlabel(xlabel)
    ax_coinc.set_ylabel(ylabel)
    ax_x.tick_params(axis='both', left=False, labelleft=False, bottom=False,
                     labelbottom=False)  # removes x labels
    ax_y.tick_params(axis='both', left=False, labelleft=False, bottom=False,
                     labelbottom=False)  # removes x labels
    
    if save != None:
        fig.savefig(f'{save}{num}.png')
    else:
        pass
    return fig, hist_x, hist_y, ax_coinc, ax_x, ax_y
