
from __future__ import division
from __future__ import print_function

import os
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame as Gdf
from cea.demand import demand

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


class SimulatePlugin(cea.plugin.CeaPlugin):
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

    prop_overrides = pd.DataFrame(index=locator.get_zone_building_names())
    for i, key in enumerate(problem['names']):
        print("Setting prop_overrides['%s'] to %s" % (key, sample[i]))
        prop_overrides[key] = sample[i]

    prop_overrides.to_csv(locator.get_building_overrides())

    return locator


def read_problem(samples_path):
    with open(os.path.join(samples_path, 'problem.pickle'), 'rb') as f:
        problem = pickle.load(f)
    return problem


def read_samples(samples_path):
    samples = np.load(os.path.join(samples_path, 'samples.npy'))
    return samples


def simulate_demand_sample(locator, config, output_parameters):
    """
    Run a demand simulation for a single sample. This function expects a locator that is already initialized to the
    simulation folder, that has already been prepared with `apply_sample_parameters`.

    :param locator: The InputLocator to use for the simulation
    :type locator: InputLocator

    :param output_parameters: The list of output parameters to save to disk
    :type output_parameters: list of str

    :return: Returns the columns of the results of the demand simulation as defined in `output_parameters`
    :rtype: pandas.DataFrame
    """

    # Modify config file to run the demand simulation for only specific quantities
    config.demand.resolution_output = "monthly"
    config.demand.massflows_output = []
    config.demand.temperatures_output = []
    config.demand.format_output = "csv"
    config.demand.override_variables = True

    # Run demand simulation
    totals, time_series = demand_main.demand_calculation(locator, config)
    return totals[output_parameters], time_series


def simulate_demand_batch(sample_index, batch_size, samples_folder, scenario, simulation_folder, config,
                          output_parameters):
    """
    Run the simulations for a whole batch of samples and write the results out to the samples folder.

    Each simulation result is saved to the samples folder as `result.$i.csv` with `$i` representing the index into
    the samples array.

    :param sample_index: The index into the first sample of the batch as defined in the `samples.npy` NumPy array in
                         the samples folder
    :type sample_index: int

    :param batch_size: The number of simulations to perform, starting at `sample_index`
    :type batch_size: int

    :param samples_folder: The path to the samples folder, containing the `samples.npy` and `problem.pickle` files
    :type samples_folder: str

    :param scenario: The path to the scenario template
    :type scenario: str

    :param simulation_folder: The path to the folder to use for simulation
    :type simulation_folder: str

    :param config: The CEA configuration object
    :type config: cea.config.Configuration

    :param output_parameters: The list of output parameters to save to disk
    :type output_parameters: list of str

    :return: None
    """
    for i in range(sample_index, sample_index + batch_size):
        locator = apply_sample_parameters(i, samples_folder, scenario, simulation_folder)
        print("Running demand simulation for sample %i" % i)
        totals, time_series = simulate_demand_sample(locator, config, output_parameters)

        # Save results in samples folder
        totals.to_csv(os.path.join(samples_folder, 'result.%i.csv' % i))
        for j, item in enumerate(time_series):
            item.to_csv(os.path.join(samples_folder, 'result.%i.%i.csv' % (i, j)))


def main(config):
    """
    Parse the arguments passed to the script and run the demand simulations for each sample in the batch.

    The current batch is the list of samples starting at `--sample-index`, up until
    `--sample-index + --number-of-simulations`.

    Run this script with the argument `--help` to get an overview of the parameters.
    """
    assert os.path.exists(config.scenario), 'Scenario not found: %s' % config.scenario
    locator= cea.inputlocator.InputLocator(config.scenario, config.plugins)


    # Save output parameters
    np.save(os.path.join(config.sensitivity_demand.samples_folder, 'output_parameters.npy'),
            np.array(config.sensitivity_demand.output_parameters))

    samples = read_samples(config.sensitivity_demand.samples_folder)
    if config.sensitivity_demand.number_of_simulations is None:
        # Simulate all remaining samples
        config.sensitivity_demand.number_of_simulations = len(samples) - config.sensitivity_demand.sample_index
    else:
        # Ensure batch size does not exceed number of remaining samples
        config.sensitivity_demand.number_of_simulations = min(
            config.sensitivity_demand.number_of_simulations,
            len(samples) - config.sensitivity_demand.sample_index)

    simulate_demand_batch(sample_index=config.sensitivity_demand.sample_index,
                          batch_size=config.sensitivity_demand.number_of_simulations,
                          samples_folder=config.sensitivity_demand.samples_folder,
                          scenario=config.scenario,
                          simulation_folder=config.sensitivity_demand.simulation_folder,
                          config=config,
                          output_parameters=config.sensitivity_demand.output_parameters)


if __name__ == '__main__':
    config = cea.config.Configuration()
    config.apply_command_line_args(sys.argv[1:], ['general', 'sensitivity-demand'])
    main(config)
