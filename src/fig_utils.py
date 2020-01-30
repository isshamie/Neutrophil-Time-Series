import numpy as np
import matplotlib.pyplot as plt
import os


#################################################
### Plot helpers
#################################################
def helper_save(f_save, remove_bg=True):
    """ Function to save as png and svg if f_save is not None."""
    if remove_bg:
        f = plt.gcf()
        f.patch.set_facecolor("w")
        axs = f.get_axes()
        for ax in axs:
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

    if f_save is not None:
        #Remove suffixes
        f_save = f_save.replace('.csv', '')
        f_save = f_save.replace('.tsv', '')
        f_save = f_save.replace('.txt', '')

        if '.png' in f_save:
            plt.savefig(f_save, bbox_inches="tight", transparent=True)
            plt.savefig(f_save.split('.png')[0] + '.svg')
        else:
            plt.savefig(f_save + '.png',bbox_inches='tight', transparent=True)
            plt.savefig(f_save + '.svg')
        return


def determine_rows_cols(num_samples):
    nrows = 2
    ncols = 2
    while nrows * ncols <= num_samples:
        ncols = ncols + 1
        if nrows * ncols <= num_samples:
            nrows = nrows + 1

    return nrows,ncols


#######################################
def wrap_plots(group,func, f_save, names=None, *args):

    xlim = [np.infty, -np.infty]
    ylim = [np.infty, -np.infty]

    num_samples = len(group)
    nrows,ncols = determine_rows_cols(num_samples)

    f = plt.figure()
    axs = []

    for ind, fname in enumerate(group):
        axs.append(plt.subplot(nrows, ncols, ind + 1))

        func(fname, fname + "_nucl", *args,f=f)

        # heat_plot(fname, save_f=heat_save, f=f, curr_ax=axs[ind],
        #           num_peaks=num_peaks, is_norm=is_norm)
        xlim[0] = min(axs[ind].get_xlim()[0], xlim[0])
        ylim[0] = min(axs[ind].get_ylim()[0], ylim[0])
        xlim[1] = max(axs[ind].get_xlim()[1], xlim[1])
        ylim[1] = max(axs[ind].get_ylim()[1], ylim[1])
        if names is None:
            curr_label = os.path.basename(fname)
        else:
            curr_label = names[ind]
        axs[ind].set_title(curr_label)

    [ax.set_xlim(xlim) for ax in axs]
    [ax.set_ylim(ylim) for ax in axs]
    helper_save(f_save)

    return


def wrap_plots(group,func, f_save, names=None, *args):

    xlim = [np.infty, -np.infty]
    ylim = [np.infty, -np.infty]

    num_samples = len(group)
    nrows,ncols = determine_rows_cols(num_samples)

    f = plt.figure()
    axs = []

    for ind, fname in enumerate(group):
        axs.append(plt.subplot(nrows, ncols, ind + 1))

        func(fname, fname + "_nucl", *args,f=f)

        # heat_plot(fname, save_f=heat_save, f=f, curr_ax=axs[ind],
        #           num_peaks=num_peaks, is_norm=is_norm)
        xlim[0] = min(axs[ind].get_xlim()[0], xlim[0])
        ylim[0] = min(axs[ind].get_ylim()[0], ylim[0])
        xlim[1] = max(axs[ind].get_xlim()[1], xlim[1])
        ylim[1] = max(axs[ind].get_ylim()[1], ylim[1])
        if names is None:
            curr_label = os.path.basename(fname)
        else:
            curr_label = names[ind]
        axs[ind].set_title(curr_label)

    [ax.set_xlim(xlim) for ax in axs]
    [ax.set_ylim(ylim) for ax in axs]
    helper_save(f_save)

    return
