from .microlensing_metric import *
import LensCalcPy.eventsampling as es
import os
# from LensCalcPy.pbh import Pbh
from LensCalcPy.galaxy import MilkyWayModel
import numpy as np
import pandas as pd

from rubin_sim.data import get_data_dir
from rubin_sim.utils import equatorial_from_galactic, \
                            hpid2_ra_dec, ra_dec2_hpid
import rubin_sim.maf.slicers as slicers
from numba import njit
from tqdm import tqdm
from dl import authClient as ac, queryClient as qc
from dl.helpers.utils import convert
from getpass import getpass


filters = list('ugrizy')

@njit
def mu0_from_dist(d): #d in kpc
    return 5*np.log10(100*d)

def get_nearby_sources(ra, dec, mu0, angular_radius=.01):
    query = "SELECT * FROM lsst_sim.simdr2 AS s " \
            f"WHERE 't' = Q3C_RADIAL_QUERY(s.ra, s.dec,{ra}, {dec},{angular_radius}) " \
            f"ORDER BY POWER(q3c_dist({ra}, {dec},s.ra,s.dec),2) + POWER(s.mu0 - {mu0}, 2) " \
            "ASC LIMIT 1"
    result = qc.query(sql=query) # your query result as a CSV-formatted string
    return convert(result) # result as a Pandas data frame

def adjust_mags(mu0, sources):
    d_mu = mu0 - sources['mu0']
    for filter in filters:
        sources[filter + "mag"] += d_mu
    return sources

def generate_dm_microlensing_slicer(
    min_crossing_time=1,
    max_crossing_time=10,
    t_start=1,
    t_end=3652,
    n_events=10000,
    seed=42,
    nside=128,
    initial_states=None,
    filtername="r",
    sim_sources=True
):
    """Generate a UserPointSlicer with a population of microlensing events.
    To be used with MicrolensingMetric

    Parameters
    ----------
    min_crossing_time : `float`
        The minimum crossing time for the events generated (days)
    max_crossing_time : `float`
        The max crossing time for the events generated (days)
    t_start : `float`
        The night to start generating peaks (days)
    t_end : `float`
        The night to end generating peaks (days)
    n_events : `int`
        Number of microlensing events to generate
    seed : `float`
        Random number seed
    nside : `int`
        HEALpix nside, used to pick which stellar density map to load
    filtername : `str`
        The filter to use for the stellar density map

    Returns
    -------
    slicer : `maf.UserPointsSlicer`
        A slicer populated by microlensing events
        (each slice_point is a different event)
    """
    # Seed the random number generator and generate random parameters
    rng = np.random.default_rng(seed)
    peak_times = rng.uniform(low=t_start, high=t_end, size=n_events)

    m_lens = 1
    # this_pbh = Pbh(m_lens, 1, l=0, b=0, ds=8)
    mw_model = MilkyWayModel()
    
    event_samples = es.generate_events(mw_model, nsamples=n_events, initial_states=initial_states)
    ra, dec = equatorial_from_galactic(event_samples[:,0], event_samples[:,1])

    mags = {}
    if not sim_sources:
        
        map_dir = os.path.join(get_data_dir(), "maps", "TriMaps")
        data = np.load(os.path.join(map_dir, "TRIstarDensity_%s_nside_%i.npz" % (filtername, nside)))
        star_density_cdf = data["starDensity"]
        norm_cdf = (star_density_cdf.T/star_density_cdf[:,-1]).T
    
        # magnitude bins
        bins = data["bins"]
    
        uniform_draw = rng.uniform(size=n_events)
        hp_ids = ra_dec2_hpid(nside, ra, dec)
        mags[filtername] = [np.interp(uniform_draw[i], norm_cdf[hp_id], bins[1:]) for i, hp_id \
                in enumerate(hp_ids) if not np.isnan(norm_cdf[hp_id]).any()]
    else:
        ac.login(input("Enter user name: (+ENTER) "),getpass("Enter password: (+ENTER) "))
        mu0 = mu0_from_dist(event_samples[:,3])
        sources = []
        for i in tqdm(range(event_samples.shape[0])):
            sources.append(get_nearby_sources(ra[i],dec[i],mu0[i]))
        sources = pd.concat(sources)
        sources = adjust_mags(mu0, sources)
        for filter in filters:
            mags[filter] = sources[filter + "mag"].to_numpy()
            
    
    # Set up the slicer to evaluate the catalog we just made
    slicer = slicers.UserPointsSlicer(ra, dec, lat_lon_deg=True, badval=0)
    # Add any additional information about each object to the slicer
    slicer.slice_points["peak_time"] = peak_times

    slicer.slice_points["d_lens"] = event_samples[:,2]
    slicer.slice_points["d_source"] = event_samples[:,3]
    slicer.slice_points["impact_parameter"] = event_samples[:,4]
    slicer.slice_points["crossing_time"] = event_samples[:,5]/24 # hours to days

    
    for filter in mags:
        slicer.slice_points["apparent_m_no_blend_{}".format(filter)] = mags[filter]
        slicer.slice_points["apparent_m_{}".format(filter)] = mags[filter] # kludge for no blending
    
    # print(event_samples.shape)
    # print(slicer.slice_points["peak_time"].shape)
    # print(slicer.slice_points["crossing_time"].shape)
    # print(slicer.slice_points["impact_parameter"].shape)
    

    return slicer
