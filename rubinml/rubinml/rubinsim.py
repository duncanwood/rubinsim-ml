import os
import datetime
from pathlib import Path
import csv
import pickle

import numpy as np
import pandas as pd
from numba import njit

from tqdm.notebook import tqdm

import rubin_sim.maf as maf
import rubin_sim.utils as rsUtils
from rubin_sim.data import get_baseline

import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.db as db

from . import events

def run_microlensing_metric(events: pd.DataFrame, baseline_file: str, outdir: str,
                            t_start=1., t_end=3652.):
  # Map sampled events onto an opsim via a UserPointsSlicer and run the MAF
  # MicrolensingMetric (detect mode) over the requested crossing-time range.
  filters = [ col for col in events.columns if 'mag' in col ]
  opsim = os.path.basename(baseline_file).replace('.db','')
  print(f'running on {opsim}')

  metric = maf.MicrolensingMetric()
  summaryMetrics = maf.batches.lightcurve_summary()
  bundles = {}
  resultDbs = {}

  crossing_times = [[0, 30000]]
  seed = 42
  rng = np.random.default_rng(seed)
  for crossing in crossing_times:
    key = f'{crossing[0]} to {crossing[1]}'
    bundle_rates = {}
    slicer = slicers.UserPointsSlicer(events['ra'].array,
                                      events['dec'].array, 
                                      lat_lon_deg=True, badval=0)
    # MAF expects the Einstein crossing time tE in days; events store it in hours.
    slicer.slice_points["crossing_time"]=events['crossing_time'].array/24
    slicer.slice_points["impact_parameter"]=events['umin'].array
    slicer.slice_points["peak_time"]= rng.uniform(low=t_start, high=t_end, size=len(events))
    for filter in filters:
      filtername = filter[0]
      slicer.slice_points["apparent_m_no_blend_{}".format(filtername)] = events[filter].array
      slicer.slice_points["apparent_m_{}".format(filtername)] = events[filter].array
    bundle_rates['rates'] = maf.MetricBundle(metric, slicer, None, run_name=opsim,  
                                            info_label=f'Microlensing rate (1/hour)')
    bundles[key] = maf.MetricBundle(metric, slicer, None, run_name=opsim, 
                                            summary_metrics=summaryMetrics, 
                                            info_label=f'tE {crossing[0]}_{crossing[1]} days')

    g = maf.MetricBundleGroup(bundles, baseline_file, outdir)
    g.run_all()

def run_microlensing_metric_mult(events_list: list, sources: pd.DataFrame, 
                                 baseline_file: str, outdir: str,
                                 constraint=None, metric_options={},
                                 t_start=1., t_end=3652.):
  filters = [ col for col in sources.columns if 'mag' in col ]
  opsim = os.path.basename(baseline_file).replace('.db','')
  print(f'running on {opsim}')

  metric = maf.MicrolensingMetric(**metric_options)
  summaryMetrics = maf.batches.lightcurve_summary()
  bundles = {}
  resultDbs = {}

  seed = 42
  rng = np.random.default_rng(seed)
  for events_df, events_info in events_list:

    full_events_df = events.make_full_event_df(events_df, sources)
    key = f'{events_info["pbhmass"]:08.4e}sm_{constraint}'
    bundle_rates = {}
    slicer = slicers.UserPointsSlicer(full_events_df['ra'].array,
                                      full_events_df['dec'].array, 
                                      lat_lon_deg=True, badval=0)
    slicer.slice_points["crossing_time"]=full_events_df['crossing_time'].array/24
    slicer.slice_points["impact_parameter"]=full_events_df['umin'].array
    slicer.slice_points["peak_time"]= rng.uniform(low=t_start, high=t_end, 
                                                  size=len(full_events_df))
    for f in filters:
      filtername = f[0]
      slicer.slice_points["apparent_m_no_blend_{}".format(filtername)] = full_events_df[f].array
      slicer.slice_points["apparent_m_{}".format(filtername)] = full_events_df[f].array
    bundles[key] = maf.MetricBundle(metric, slicer, constraint=constraint, run_name=opsim, 
                                            summary_metrics=summaryMetrics, 
                                            info_label=f'PBH mass {events_info["pbhmass"]:08.4e}sm_{constraint}')

  g = maf.MetricBundleGroup(bundles, baseline_file, outdir)
  g.run_all()

  for events_df, events_info in events_list:
    result_fname = f'{outdir}/{opsim}_{events_info["pbhmass"]:08.4e}sm'
    if constraint is not None:
      result_fname += f'_{constraint}'
    result_fname += '.pickle'
    bundle = bundles[f'{events_info["pbhmass"]:08.4e}sm_{constraint}']
    events_info['opsim'] = opsim
    events_info['constraint'] = constraint
    with open(result_fname, 'wb') as f:
      pickle.dump((events_info, bundle.metric_values), f)

def run_microlensing_metric_mult_Fisher_Npts_Nights(events_list: list, sources: pd.DataFrame, 
                                 baseline_file: str, outdir: str,
                                 constraint=None, metric_options={},
                                 t_start=1., t_end=3652.):
  filters = [ col for col in sources.columns if 'mag' in col ]
  opsim = os.path.basename(baseline_file).replace('.db','')
  print(f'running on {opsim}')

  metric_Fisher = maf.MicrolensingMetric(metric_calc='Fisher', **metric_options)
  metric_Npts = maf.MicrolensingMetric(metric_calc='Npts', **metric_options)
  metric_Nights = maf.MicrolensingMetric(metric_calc='Nnights', **metric_options)
  summaryMetrics = maf.batches.lightcurve_summary()
  bundles = {}


  seed = 42
  rng = np.random.default_rng(seed)
  for events_df, events_info in events_list:

    full_events_df = events.make_full_event_df(events_df, sources)
    key = f'{events_info["pbhmass"]:08.4e}sm_{constraint}'

    slicer = slicers.UserPointsSlicer(full_events_df['ra'].array,
                                      full_events_df['dec'].array, 
                                      lat_lon_deg=True, badval=0)
    slicer.slice_points["crossing_time"]=full_events_df['crossing_time'].array/24
    slicer.slice_points["impact_parameter"]=full_events_df['umin'].array
    slicer.slice_points["peak_time"]= rng.uniform(low=t_start, high=t_end, 
                                                  size=len(full_events_df))
    for f in filters:
      filtername = f[0]
      slicer.slice_points["apparent_m_no_blend_{}".format(filtername)] = full_events_df[f].array
      slicer.slice_points["apparent_m_{}".format(filtername)] = full_events_df[f].array
    bundles[key+'_Fisher'] = maf.MetricBundle(metric_Fisher, slicer, constraint=constraint, run_name=opsim, 
                                            summary_metrics=summaryMetrics, 
                                            info_label=f'PBH mass {events_info["pbhmass"]:08.4e}sm_{constraint} Fisher')
    bundles[key+'_Npts'] = maf.MetricBundle(metric_Npts, slicer, constraint=constraint, run_name=opsim, 
                                            summary_metrics=summaryMetrics, 
                                            info_label=f'PBH mass {events_info["pbhmass"]:08.4e}sm_{constraint} Npts')
    bundles[key+'_Nights'] = maf.MetricBundle(metric_Nights, slicer, constraint=constraint, run_name=opsim, 
                                            info_label=f'PBH mass {events_info["pbhmass"]:08.4e}sm_{constraint} Nights')

  g = maf.MetricBundleGroup(bundles, baseline_file, outdir)
  g.run_all()

  for events_df, events_info in events_list:
    result_fname = f'{outdir}/{opsim}_{events_info["pbhmass"]:08.4e}sm'
    if constraint is not None:
      result_fname += f'_{constraint}'
    result_fname += '.pickle'
    bundle_Fisher = bundles[f'{events_info["pbhmass"]:08.4e}sm_{constraint}_Fisher']
    bundle_Npts = bundles[f'{events_info["pbhmass"]:08.4e}sm_{constraint}_Npts']
    bundle_Nights = bundles[f'{events_info["pbhmass"]:08.4e}sm_{constraint}_Nights']
    events_info['opsim'] = opsim
    events_info['constraint'] = constraint
    with open(result_fname, 'wb') as f:
      pickle.dump((events_info, bundle_Fisher.metric_values,
                   bundle_Npts.metric_values, bundle_Nights.metric_values), f)


def make_metric_plots(bundles, outDir:str):

    plotDict = {'reduce_func': np.sum, 'nside': 64, 'color_min': 0}
    plotFunc = maf.plots.HealpixSkyMap()
    ph = maf.plots.PlotHandler(out_dir=outDir, thumbnail=False)
    for k in bundles:
        ph.set_metric_bundles([bundles[k]])
        fig = ph.plot(plot_func=plotFunc, plot_dicts=plotDict)
        fig.close()

    max_colors = [30,300]
    for c in max_colors:
        plotDict = {'reduce_func': np.sum, 'nside': 64, 
                    'color_min': 0, 'color_max': c}
    plotFunc = maf.plots.HealpixSkyMap()
    ph = maf.plots.PlotHandler(out_dir=outDir, thumbnail=False, outfile_suffix=f'maxcolor-{c}')
    for k in bundles:
        ph.set_metric_bundles([bundles[k]])
        fig = ph.plot(plot_func=plotFunc, plot_dicts=plotDict)
        fig.close()

def make_ndet_df(result_files: list):
  ndetections = []
  masses = []
  opsims= []
  for file in result_files:
    with open(file, 'rb') as f:
      events_info, detections = pickle.load(f)
      masses.append(events_info['pbhmass'])
      opsims.append(events_info['opsim'])
      ndetections.append(np.sum(detections)/len(detections))
  return pd.DataFrame({'ndet':np.array(ndetections),
                       'pbhmass' : masses,
                       'opsim' : opsims})