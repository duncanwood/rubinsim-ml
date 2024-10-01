# from . import events
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import seaborn as sns

single_exp_m5 = {
    'u':23.8, 
    'g':24.5,
    'r':24.03,
    'i':23.41,
    'z':22.74,
    'y':22.96}
color_filter = {
    'u':'blue',
    'g':'green',
    'r':'orange',
    'i':'red',
    'z':'brown',
    'y':'black'}
    

def check_linear(x: np.array):
 d = np.diff(x)
 return np.allclose(d[:-1], d[1:])

def check_log(x: np.array):
  return check_linear(np.log(x))

def plot_hist_color(
        counts, bins, colors, clabel='Color', maxcolor=None,
        mincolor=None, scale=None):

  if scale is None:
    if check_log(bins):
      scale = 'log'
    else: 
      scale = 'linear'
  nanmask = np.isfinite(colors)
  
  colors = colors[nanmask]
  if maxcolor is None:
      maxcolor = np.max(colors)
  if mincolor is None:
      mincolor = np.min(colors)
  viridis = mpl.colormaps['viridis'].resampled(256)
  fig, ax = plt.subplots(1,1)
  # counts, bins = np.histogram(data, bins=bins)

  bars = ax.bar(bins[:-1][nanmask], counts[nanmask], 
  width=(bins[1:]-bins[:-1])[nanmask], align='edge', 
         color=viridis((colors-mincolor)/(maxcolor-mincolor)))
  nanbars = ax.bar(bins[:-1][~nanmask], counts[~nanmask], 
                   width=(bins[1:]-bins[:-1])[~nanmask], 
                   align='edge', color='white',hatch='/',
                   edgecolor='black')
  plt.xscale(scale)

  sm = ScalarMappable(cmap=viridis, norm=plt.Normalize(mincolor,maxcolor))
  sm.set_array([])
  cbar = plt.colorbar(sm,ax=ax)
  cbar.set_label(clabel, rotation=270,labelpad=25)
  return fig, ax, cbar


def dl_hist(df: pd.DataFrame, outdir: str, name: str):
  """

  Args:
      df (pd.DataFrame): _description_
      outdir (str): _description_
      filename (str)
  """  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  df.hist('dl', bins=np.linspace(0,20,100), ax=ax);
  ax.set_xlabel('Lens distance (kpc)')

  if outdir is None or name is None:
    plt.show()
  else:
    fig.savefig(outdir + '/' + name + '_dl_hist.pdf')
  plt.clf()

def umin_hist(df: pd.DataFrame, outdir: str, name: str):
  """

  Args:
      df (pd.DataFrame): _description_
      outdir (str): _description_
      filename (str)
  """  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  df.hist('umin', bins=200, ax=ax)
  ax.set_xlabel('Minimum impact parameter')

  if outdir is None or name is None:
    plt.show()
  else:
    fig.savefig(outdir + '/' + name + '_umin_hist.pdf')
  plt.clf()

def crossing_time_hist(df: pd.DataFrame, outdir: str, name: str):
  """

  Args:
      df (pd.DataFrame): _description_
      outdir (str): _description_
      filename (str)
  """  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  mintime, maxtime = np.log10(np.percentile(df['crossing_time'], (0,100)))
  df.hist('crossing_time', bins=np.logspace(mintime, maxtime, 100), ax=ax);
  ax.set_xlabel('Crossing time (hours)')
  ax.set_xscale('log')
  ax.set_yscale('log')

  if outdir is None or name is None:
    plt.show()
  else:
    fig.savefig(outdir + '/' + name + '_crossing_time_hist.pdf')
  plt.clf()


def crossing_time_umin_scatter(df: pd.DataFrame, outdir: str, name: str):
  """

  Args:
      df (pd.DataFrame): _description_
      outdir (str): _description_
      filename (str)
  """  

  plot = sns.displot(
    data=df, x="crossing_time", y="umin",
      cmap="mako", log_scale=(True, False), 
      legend=True, cbar=True)
  fig = plot.fig
  ax = plot.ax
  ax.set_xlabel('Crossing time (hours)')
  ax.set_ylabel('Minimum impact parameter')

  if outdir is None or name is None:
    plt.show()
  else:
    fig.savefig(outdir + '/' + name + '_crossing_time_umin_scatter.pdf')
  plt.clf()

def mag_in_te_bins(df: pd.DataFrame, outdir=None, name=None):
  bins=np.geomspace(*np.percentile(df['crossing_time'],(0,100)),4*3+1)
  percentiles = [50+44.6,50-44.6]
  fig = plt.figure(figsize=(15,10))
  for filter in [col for col in df.columns if 'mag' in col]:
      brightness_crossing_time = np.array(bin_func(df, 'crossing_time', 
                                                   bins, mean_of_col, [filter]))
      brightness_crossing_time_percentiles = np.array(bin_func(df, 'crossing_time', bins, perc_of_col, filter, percentiles)).T
      brightness_crossing_time_errs = np.abs(brightness_crossing_time_percentiles - np.vstack([brightness_crossing_time, brightness_crossing_time]))
      plt.errorbar(bins[:-1], brightness_crossing_time, brightness_crossing_time_errs, label=filter, marker='o', linestyle='None', capsize=5,capthick=2,color=color_filter[filter[0]])
      plt.xscale('log')
      plt.title('Mean source brightness in crossing time bins')
      plt.xlabel('Crossing time (hours)')
      plt.hlines(single_exp_m5[filter[0]],bins[0],bins[-1],color=color_filter[filter[0]],linestyle='dashed', label=f'{filter} m5 limit')
  plt.legend()
  plt.gca().invert_yaxis()
  if outdir is None or name is None:
    plt.show()
  else:
    fig.savefig(outdir + '/' + name + '_mag_in_te_bins.pdf')
  plt.clf()

def make_event_plots(df: pd.DataFrame, outdir=None, name=None):
  """_summary_

  Args:
      df (pd.DataFrame): _description_
      outdir (str): _description_
      name (str): _description_
  """  
  p = (df, outdir, name)
  dl_hist(*p)
  umin_hist(*p)
  crossing_time_hist(*p)
  crossing_time_umin_scatter(*p)
  mag_in_te_bins(*p)
  plt.close('all')

def rows_in_bin(df, col, minval, maxval):
    return df[(df[col] >= minval) & (df[col] < maxval)]

def bin_efficiency(df: pd.DataFrame, col, bins):
    detections_in_bins = []
    for i in range(len(bins)-1):
        l_edge, r_edge = bins[i],bins[i+1]
        events_ind = rows_in_bin(df, col, l_edge, r_edge).index
        n_detected = np.sum(df['detected'][events_ind])
        n_not_detected = df.shape[0] - n_detected
        detections_in_bins.append([n_detected, n_not_detected])
    return np.asarray(detections_in_bins)

def bin_func(full_event_df: pd.DataFrame, col, bins, func, *args):
    bin_results = []
    for i in range(len(bins)-1):
        l_edge, r_edge = bins[i],bins[i+1]
        events = rows_in_bin(full_event_df, col, l_edge, r_edge)
        bin_results.append(func(events, *args))
    return bin_results
def mean_of_col(events, meancol):
    return events[meancol].to_numpy().mean()
def std_of_col(events, meancol):
    return events[meancol].to_numpy().std()
def perc_of_col(events, meancol, percentiles):
    col_vals = events[meancol].to_numpy()
    if col_vals.shape[0] > 0:
        return np.percentile(col_vals, percentiles)
    else:
        return np.full_like(percentiles, np.nan)
def detection_efficiency(df: pd.DataFrame):
  if df.shape[0] == 0:
    return 0.
  else:
    return np.sum(df['detected'])/df.shape[0]

def compute_efficiency(det: np.array):
    return det[:,0]/(det[:,0] + det[:,1])

def compute_efficiency_df(df: pd.DataFrame, col:str, bins: np.array):
  return np.array(bin_func(df, col, bins, detection_efficiency))

def efficiency_in_bins(df: pd.DataFrame, bin_col='crossing_time', nbins=100, binscale='log', title=None):

  if title is None:
    title = f'Detection efficiency in {bin_col} bins'

  if binscale == 'log':
    bins = np.geomspace(*np.percentile(df[bin_col],(0,100)), nbins)
  else:
    bins = np.linspace(*np.percentile(df[bin_col],(0,100)), nbins)
    
  efficiencies = compute_efficiency_df(df, bin_col, bins)
  crossing_time_counts = np.array(bin_func(df, bin_col, bins, len))
  fig, ax, cbar = plot_hist_color(crossing_time_counts, bins, 
                                  efficiencies, clabel='Efficiency in bin')
  plt.xlabel(bin_col)
  plt.ylabel('Count of events')

  plt.title(title)
  return fig, ax, cbar

def efficiency_in_te_bins(df: pd.DataFrame, outdir=None, name=None, nbins=100, binscale='log', title=None):

  bin_col='crossing_time'
  fig, ax, cbar =  efficiency_in_bins(df, bin_col, nbins, binscale, title)

  plt.xscale(binscale)
  plt.xlabel('Crossing time (hours)')
  plt.yscale('log')

  if outdir is None or name is None:
    plt.show()
  else:
    fig.savefig(outdir + '/' + name + '_crossing_time_umin_scatter.pdf')
  plt.clf()



def detection_scatter_mag(df: pd.DataFrame, outdir=None, name=None, band='ymag'):
  ax = sns.histplot(
    data=df, x="umin", y=band, hue='detected', stat='percent',
      cmap="rocket", 
      legend=True, cbar=True)
  fig = plt.gcf()
  ax.set_xlabel('Minimum impact parameter')
  ax.set_ylabel(band)
  ax.invert_yaxis()

  if outdir is None or name is None:
    plt.show()
  else:
    fig.savefig(outdir + '/' + name + f'_umin_{band}_scatter.pdf')
  plt.clf()
  

def make_detection_plots(df: pd.DataFrame, outdir: str, name: str):
  """_summary_

  Args:
      df (pd.DataFrame): _description_
      outdir (str): _description_
      name (str): _description_
  """  
  p = (df, outdir, name)
  efficiency_in_te_bins(*p)
  detection_scatter_mag(*p)
  plt.close('all')

def make_all_plots(df: pd.DataFrame, outdir: str, name: str):

  p = (df, outdir, name)
  make_event_plots(*p)
  make_detection_plots(*p)
  plt.close('all')


