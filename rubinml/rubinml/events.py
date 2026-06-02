import time
import warnings
import datetime
import glob
from pathlib import Path
import csv
import pickle

import numpy as np
import pandas as pd
from numba import njit

from tqdm.notebook import tqdm

from LensCalcPy.galaxy import MilkyWayModel
import LensCalcPy.pbh

# Monte-Carlo (Metropolis-Hastings) sampler for microlensing events and their
# Rubin/LSST detectability. The target density is the analytic differential
# event rate of an NFW dark-matter (e.g. PBH) lens population folded through a
# Milky Way model (LensCalcPy); see the dissertation for the rate derivation.
# This is the laptop-scale replacement for the cluster-scale PopSyCLE run:
# sampling cost is linear in the number of proposed events, not quadratic in
# the source catalog. Units: lens/source distances dl, ds in kpc; mu0 a
# distance modulus; crossing_time in hours; u_t the threshold impact parameter
# (in Einstein radii) bounding what counts as a detectable event.

@njit
def wrap_degrees(x):
    x = (x + 180) % 360
    return (x - 180)

@njit
def kpc_from_mu0(mu0):
    return np.power(10, mu0/5.+1-3)
@njit
def mu0_from_kpc(kpc):
    return 5*(np.log10(kpc) + 3. - 1. )

def differential_rate_integrand_mw_maker(l, b, ds, u_t, mass, mw_model, 
                                         finite=True, v_disp=None, t_e=True, 
                                         tmax=np.inf, tmin=0):
    def differential_rate_integrand_mw(umin, dl, t, finite=True):
        return LensCalcPy.pbh.differential_rate_integrand(
                l, b, dl, ds, umin, t, u_t, mass, mw_model, 
                finite=finite, v_disp=v_disp, t_e = t_e, 
                tmax=tmax, tmin=tmin)
    return differential_rate_integrand_mw

def source_lensing_rate(l, b, ds, mw_model, u_t=5, 
                        mass=1,tcad = 24, tobs = 24*365*10, \
                        epsabs = 1.49e-08, epsrel = 1.49e-08, 
                        efficiency=None) :
    return LensCalcPy.pbh.rate_total(ds, mass, u_t, 
            differential_rate_integrand_mw_maker(l, b, ds, u_t, mass, mw_model), 
            tcad = tcad, tobs = tobs, epsabs =epsabs, epsrel = epsrel, 
            efficiency=efficiency) 

def calculate_lensing_rates(sources: pd.DataFrame, mw: LensCalcPy.galaxy.Galaxy, 
                            pbhmass, outdir: str, n_sources=None, u_t=5, write_csv=False):
   
    rates_info = {'sources_checksum' : pd.util.hash_pandas_object(sources).sum(),
                  'sources_len'  : sources.shape[0],
                  'pbhmass'      : pbhmass,
                  'n_sources'    : n_sources,
                  'u_t'          : u_t}

    if write_csv:
        rates_file = f'{outdir}/rates_ml_rates_{int(np.ceil(time.time()))}_ut-{u_t}_{pbhmass:.02e}sm.csv'
    else:
        rates_file = f'{outdir}/rates_ml_rates_{int(np.ceil(time.time()))}_ut-{u_t}_m-{pbhmass:.02e}sm.pickle'
    basepath = Path(outdir)
    basepath.mkdir(parents=True, exist_ok=True)
    rates = {}

    if n_sources is None:
        n_sources = sources.shape[0]
    else:
        n_sources = min(sources.shape[0], n_sources)
    if write_csv:
        with open(rates_file, 'x') as f:
            writer = csv.writer(f)
            writer.writerow(("l","b","mu0","ML rate (1/hour)"))
    source_arr = sources[['gall', 'galb', 'mu0']].to_numpy()
    indices = np.random.randint(sources.shape[0], size=n_sources)
    for i in tqdm(indices,smoothing=0):
        l,b,mu0 = source_arr[i]
        ds = kpc_from_mu0(mu0)
        with warnings.catch_warnings(action="ignore"):
            rates[i] = source_lensing_rate(l, b, ds, mw, u_t=u_t, mass=pbhmass)
        if write_csv:
            with open(rates_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow((l,b,mu0,rates[i]))
    if not write_csv:
        with open(rates_file, 'wb') as f:
            pickle.dump((rates, rates_info), f)
    return rates

def load_rates_from_file(old_rates_file):

    rates = {}
    with open(old_rates_file, 'r') as f:
        print(f'opened {old_rates_file}')
        reader = csv.reader(f)
        next(reader) #skip header
        for line in reader:
            rates[tuple(float(_) for _ in line[:-1])]=float(line[-1])
    
    return rates

def rates_to_rubin_counts(rates, n_tristar=11_433_322_690):
    return round(sum(rates.values())*n_tristar/len(rates.values())*24*365*10)

def log_rates_to_rubin_counts(rates: pd.DataFrame, n_tristar=11_433_322_690):
    return round(sum(np.rates.values())*n_tristar/len(rates.values())*24*365*10)

def rubin_counts_from_rates_file(file: str): 
    return rates_to_rubin_counts(load_rates_from_file(file))

def get_partial_rates(filename):
    lastkey = ()
    rates = {}
    with open(filename, 'r') as f:
        _=next(f)
        for line in f:
            l,b,mu0,rate = [float(_) for _ in 
                            line.replace('\n',',').split(',') if len(_)>0]
            rates[(l,b,mu0)] = rate
            lastkey = (l,b,mu0) 
    return rates, lastkey


def sample_density_single_source(params,
                 mw_model: LensCalcPy.galaxy.Galaxy,
                 lbounds=(-180,180),
                 bbounds=(-90,90),
                 mass=1,
                 u_t=5,
                 **lcp_params
):
    """
    Compute density of microlensing event space in differential volume.

    Parameters
    ----------
    params : np.array
       source_index - index of source in catalog
       l  - galactic longitude (degrees)
       b  - galactic latitide (degrees) 
       mu0 - distance modulus
       dl - lens distance from Earth (kpc)
       umin - minimum impact parameter
       crossing time - timescale of microlensing event

    lbounds : tuple(float, float)
        bounds on galactic longitude in degrees
    bbounds : tuple(float, float)
        bounds on galactic latitude in degrees

    Returns
    -------
    float
        Event rate in (hours)**-(2) * (kpc)**(-2)
    """
    source_index, l, b, mu0, dl, umin, crossing_time = params
    l = wrap_degrees(l)
    b = wrap_degrees(b)
    ds = kpc_from_mu0(mu0)
    if l < lbounds[0] \
    or l > lbounds[1] \
    or b > bbounds[1] \
    or b < bbounds[0] \
    or dl < 0 or dl > ds \
    or umin <= 0 \
    or crossing_time <= 0:
        return 0
    prob = LensCalcPy.pbh.differential_rate_integrand(l, b, dl, ds, umin, crossing_time, u_t, mass, mw_model,**lcp_params)
    if prob < 0 or np.isnan(prob):
        return 0
    return prob

def sample_density_single_source_log(params,
                 mw_model,
                 lbounds=(-180,180),
                 bbounds=(-90,90),
                 mass=1,
                 u_t=2,
                 **lcp_params):
    density = sample_density_single_source(params, mw_model, lbounds, bbounds, mass, u_t, **lcp_params)
    if density==0.:
        return -np.inf
    else:
        return np.log(density)


def make_events(
    sources,
    outfile,
    n_survey_events: int,
    u_t = 5.,
    t_min = 24.,
    t_max = 24.*365*10,
    pbhmass = 1.,
    ntoss = 20000,
    write_progress = False):
    """_summary_

    Args:
        sources (_type_): _description_
        outfile (_type_): _description_
        n_survey_events (int): Number of events to generate.
        u_t (float, optional): _description_. Defaults to 5.
        t_min (float, optional): Minimum crossing time cutoff (hours). Defaults to 24.
        t_max (float, optional): Maximum crossing time cutoff (hours). Defaults to 24*365*10.
        pbhmass (float, optional): PBH mass in Solar masses. Defaults to 1.
        ntoss (int, optional): Number of initial samples to discard. Defaults to 20000.
        write_progress (bool, optional): Write every sample immediately 
                                         to a csv upon generation. Defaults to False.
    """

    nsteps=round(n_survey_events)
    headers=['source_index', 'dl', 'umin', 'crossing_time', 'lograte']
    metadatastr = \
        f'''# original file, {outfile}
# n_survey_events, {n_survey_events}
# u_t, {u_t}
# t_min, {t_min}
# t_max, {t_max}
# pbhmass, {pbhmass}
# ntoss, {ntoss}
# simulation date, {datetime.datetime.now()}
'''
    events_info = {'original file': outfile, 
                   'n_survey_events' : n_survey_events,
                   'u_t' : u_t,
                   't_min' : t_min,
                   't_max' : t_max,
                   'pbhmass' : pbhmass,
                   'ntoss' : ntoss,
                   'simulation date' : {datetime.datetime.now()}
    
    }

    events_file = outfile
    basedir = '/'.join(events_file.split("/")[:-1])
    basepath = Path(basedir)
    basepath.mkdir(parents=True, exist_ok=True)
    with open(events_file, 'w') as f:
        f.write(metadatastr)
        f.write(','.join(headers))
    

    mw = MilkyWayModel()

    p0 = []
    for i in range(1):
        source_index = np.random.randint(0,len(sources))
        l,b,mu0 = sources.iloc[source_index][['gall','galb','mu0']]
        ds = kpc_from_mu0(mu0)
        p0.append([source_index, l, b, mu0, ds*np.random.random(), 1, 100])

    mu0 = sources.iloc[p0[0][0]]['mu0']
    ds = np.power(10, mu0/5.-3)
    new_lograte = np.log(ds)+sample_density_single_source_log(p0[0], 
                 mw, # LensCalcPy.galaxy object
                 u_t=u_t,
                 mass=pbhmass)

    samples = [p0[0]+[new_lograte]]
    i = 0 
    n_proposed = 0

    # much faster (like 1000x) to pull these into numpy arrays than use iloc
    source_arr = sources[['gall', 'galb', 'mu0']].to_numpy()

    # Metropolis-Hastings over event space. Each proposal redraws a source and
    # then draws lens distance dl ~ U(0, ds), impact parameter umin ~ U(0, u_t),
    # and crossing time log-uniform in [t_min, t_max] -- between the daily
    # cadence and the full survey length, outside which events are not
    # recoverable. The proposal log-rate adds log(ds) (line-of-sight volume) and
    # log(crossing_time) (Jacobian of the log-uniform t draw). Acceptance is the
    # standard log-space rule; the first ntoss samples are burn-in.
    for i in tqdm(range(nsteps+ntoss), smoothing=0):
    
        source_index, l, b, mu0, dl, umin, crossing_time, lograte = samples[-1]    
        if i == ntoss: # clear the events after generating enough to throw out
            samples = []
        while True:
            new_source_index = np.random.randint(sources.shape[0])
            new_l,new_b,new_mu0 = source_arr[source_index]
            ds = kpc_from_mu0(new_mu0)
            new_dl = np.random.random()*ds # put lens uniformly random between earth and source
            new_umin = np.random.random()*u_t # limit to photometric repeatability limit for dim sources
            new_crossing_time = t_min * np.power(t_max/t_min, np.random.random()) # limit between daily cadence and full survey length
            new_event=[new_source_index, new_l, new_b, new_mu0, 
                       new_dl, new_umin, new_crossing_time]
            new_lograte =  sample_density_single_source_log(new_event, # galactic longitude (degrees)
                    mw, # LensCalcPy.galaxy object
                    u_t=u_t,
                    mass=pbhmass,
                    t_e=True) \
                    + np.log(ds) \
                    + np.log(new_crossing_time)
            n_proposed += 1
            if new_lograte > lograte or np.exp(new_lograte - lograte) > np.random.random():
                samples.append(new_event+[new_lograte])
                if write_progress and (i >= ntoss): # only write values you're not going to toss
                    with open(events_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow(samples[-1])
                break
    event_df = pd.DataFrame(samples, columns=['source_index', 'gall', 'galb', 
                                              'mu0', 'dl', 'umin', 
                                              'crossing_time', 'lograte'])

    with open(outfile, 'wb') as f:
        pickle.dump((event_df, events_info), f)

    return event_df

def make_full_event_df(event_df: pd.DataFrame, source_df: pd.DataFrame):
    cols=['ra','dec',  'umag', 'gmag', 
          'rmag', 'imag', 'zmag', 'ymag']
    sub_sources = source_df[cols]
    return event_df.merge(sub_sources, how='inner', 
                          left_on='source_index', right_index=True)