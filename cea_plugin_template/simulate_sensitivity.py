"""
Creates a simple summary of the demand totals from the cea demand script.

NOTE: This is an example of how to structure a cea plugin script. It is intentionally simplistic to avoid distraction.
"""
from __future__ import division
from __future__ import print_function

import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

import cea.config
import cea.inputlocator
import cea.plugin

__author__ = "Daren Thomas"
__copyright__ = "Copyright 2020, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Daren Thomas"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Daren Thomas"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"


class DemandSummaryPlugin(cea.plugin.CeaPlugin):
    """
    Define the plugin class - unless you want to customize the behavior, you only really need to declare the class. The
    rest of the information will be picked up from ``default.config``, ``schemas.yml`` and ``scripts.yml`` by default.
    """
    pass

def apply_sample_parameters(sample_index, samples_path, scenario_path, simulation_path):
    """
    Copy the scenario from the `scenario_path` to the `simulation_path`. Patch the parameters from
    the problem statement. Return an `InputLocator` implementation that can be used to simulate the demand
    of the resulting scenario.

    The `simulation_path` is modified by the demand calculation. For the purposes of the sensitivity analysis, these
    changes can be viewed as temporary and deleted / overwritten after each simulation.

    :param sample_index: zero-based index into the samples list, which is read from the file `$samples_path/samples.npy`
    :type sample_index: int

    :param samples_path: path to the pre-calculated samples and problem statement (created by
                        `sensitivity_demand_samples.py`)
    :type samples_path: str

    :param scenario_path: path to the scenario template
    :type scenario_path: str

    :param simulation_path: a (temporary) path for simulating a scenario that has been patched with a sample
                            NOTE: When simulating in parallel, special care must be taken that each process has
                            a unique `simulation_path` value. For the Euler cluster, this is solved by ensuring the
                            simulation is done with `multiprocessing = False` and setting the `simulation_path` to
                            the special folder `$TMPDIR` that is set to a local scratch folder for each job by the
                            job scheduler of the Euler cluster. Other setups will need to adopt an equivalent strategy.
    :type simulation_path: str

    :return: InputLocator that can be used to simulate the demand in the `simulation_path`
    """
    if os.path.exists(simulation_path):
        shutil.rmtree(simulation_path)
    shutil.copytree(scenario_path, simulation_path)
    locator = InputLocator(scenario=simulation_path)

    problem = read_problem(samples_path)
    samples = read_samples(samples_path)
    try:
        sample = samples[sample_index]
    except IndexError:
        return None

    prop = Gdf.from_file(locator.get_zone_geometry()).set_index('Name')
    prop_overrides = pd.DataFrame(index=prop.index)
    for i, key in enumerate(problem['names']):
        print("Setting prop_overrides['%s'] to %s" % (key, sample[i]))
        prop_overrides[key] = sample[i]

    sample_locator = InputLocator(scenario=simulation_path)
    prop_overrides.to_csv(sample_locator.get_building_overrides())

    return sample_locator





def read_problem(samples_path):
    with open(os.path.join(samples_path, 'problem.pickle'), 'r') as f:
        problem = pickle.load(f)
    return problem

def read_samples(samples_path):
    samples = np.load(os.path.join(samples_path, 'samples.npy'))
    return samples

def simulate():

for i in range(sample_index, sample_index+batch_size):
    locator = 


def summarize(total_demand_df, fudge_factor):
    """
    Return only the following fields from the Total_demand.csv file:
    - Name (Unique building ID)
    - GFA_m2 (Gross floor area)
    - QC_sys_MWhyr (Total system cooling demand)
    - QH_sys_MWhyr (Total building heating demand)
    """
    result_df = total_demand_df[["Name", "GFA_m2", "QC_sys_MWhyr", "QH_sys_MWhyr"]].copy()
    result_df["QC_sys_MWhyr"] *= fudge_factor
    result_df["QH_sys_MWhyr"] *= fudge_factor
    return result_df


def main(config):
    """
    This is the main entry point to your script. Any parameters used by your script must be present in the ``config``
    parameter. The CLI will call this ``main`` function passing in a ``config`` object after adjusting the configuration
    to reflect parameters passed on the command line / user interface

    :param cea.config.Configuration config: The configuration for this script, restricted to the scripts parameters.
    :return: None
    """
    locator = cea.inputlocator.InputLocator(config.scenario, config.plugins)
    summary_df = summarize(locator.get_total_demand.read(), config.demand_summary.fudge_factor)
    locator.demand_summary.write(summary_df)


if __name__ == '__main__':
    main(cea.config.Configuration())
