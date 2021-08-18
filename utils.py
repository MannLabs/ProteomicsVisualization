import logging
import pandas as pd
import numpy as np
from numba import njit
from tqdm import tqdm
from pyrawfilereader import RawFileReader

# specify a configuration whether to display the Plotly logo in the toolbar and how to save the plot
config = {
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
        'filename': 'custom_image',
        'height': 400,
        'width': 700,
        'scale': 1 # Multiply title/legend/axis/canvas sizes by this factor
      }
}

# Necessary functions to read Thermo Raw file
# This code was taken from the AlphaPept Python package (https://github.com/MannLabs/alphapept/blob/master/nbs/02_io.ipynb)

def load_thermo_raw(
    raw_file,
    most_abundant=1000
):
    """
    Load a Thermo raw file and extract all spectra
    """

    rawfile = RawFileReader(raw_file)

    spec_indices = np.array(
        range(rawfile.FirstSpectrumNumber, rawfile.LastSpectrumNumber + 1)
    )

    scan_list = []
    rt_list = []
    mass_list = []
    int_list = []
    ms_list = []
    prec_mzs_list = []
    mono_mzs_list = []
    charge_list = []

    for i in tqdm((spec_indices)):
        try:
            ms_order = rawfile.GetMSOrderForScanNum(i)
            rt = rawfile.RTFromScanNum(i)
            

            if ms_order == 2:
                prec_mz = rawfile.GetPrecursorMassForScanNum(i, 0)

                mono_mz, charge = rawfile.GetMS2MonoMzAndChargeFromScanNum(i)
            else:
                prec_mz, mono_mz, charge = 0,0,0

            masses, intensity = rawfile.GetCentroidMassListFromScanNum(i)
            if ms_order == 2:
                masses, intensity = get_most_abundant(masses, intensity, most_abundant)

            scan_list.append(i)
            rt_list.append(rt)
            mass_list.append(np.array(masses))
            int_list.append(np.array(intensity, dtype=np.int64))
            ms_list.append(ms_order)
            prec_mzs_list.append(prec_mz)
            mono_mzs_list.append(mono_mz)
            charge_list.append(charge)
        except KeyboardInterrupt as e:
            raise e
        except SystemExit as e:
            raise e
        except Exception as e:
            logging.info(f"Bad scan={i} in raw file '{raw_file}'")

    scan_list_ms1 = [scan_list[i] for i, _ in enumerate(ms_list) if _ == 1]
    rt_list_ms1 = [rt_list[i] for i, _ in enumerate(ms_list) if _ == 1]
    mass_list_ms1 = [mass_list[i] for i, _ in enumerate(ms_list) if _ == 1]
    int_list_ms1 = [int_list[i] for i, _ in enumerate(ms_list) if _ == 1]
    ms_list_ms1 = [ms_list[i] for i, _ in enumerate(ms_list) if _ == 1]

    scan_list_ms2 = [scan_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    rt_list_ms2 = [rt_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    mass_list_ms2 = [mass_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    int_list_ms2 = [int_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    ms_list_ms2 = [ms_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    mono_mzs2 = [mono_mzs_list[i] for i, _ in enumerate(ms_list) if _ == 2]
    charge2 = [charge_list[i] for i, _ in enumerate(ms_list) if _ == 2]

    prec_mass_list2 = [
        calculate_mass(mono_mzs_list[i], charge_list[i])
        for i, _ in enumerate(ms_list)
        if _ == 2
    ]

    check_sanity(mass_list)

    data = {}
    
    data["scan_list_ms1"] = np.array(scan_list_ms1)
    data["rt_list_ms1"] = np.array(rt_list_ms1)
    data["mass_list_ms1"] = np.array(mass_list_ms1, dtype=object)
    data["int_list_ms1"] = np.array(int_list_ms1, dtype=object)
    data["ms_list_ms1"] = np.array(ms_list_ms1)

    data["scan_list_ms2"] = np.array(scan_list_ms2)
    data["rt_list_ms2"] = np.array(rt_list_ms2)
    data["mass_list_ms2"] = mass_list_ms2
    data["int_list_ms2"] = int_list_ms2
    data["ms_list_ms2"] = np.array(ms_list_ms2)
    data["prec_mass_list2"] = np.array(prec_mass_list2)
    data["mono_mzs2"] = np.array(mono_mzs2)
    data["charge_ms2"] = np.array(charge2)
    
    rawfile.Close()
    return data

def get_most_abundant(
    mass, 
    intensity, 
    n_max
):
    """
    Returns the n_max most abundant peaks of a spectrum.
    Setting `n_max` to -1 returns all peaks.
    """
    if n_max == -1:
        return mass, intensity
    if len(mass) < n_max:
        return mass, intensity
    else:
        sortindex = np.argsort(intensity)[::-1][:n_max]
        sortindex.sort()

    return mass[sortindex], intensity[sortindex]

@njit
def calculate_mass(
    mono_mz, 
    charge
):
    """
    Calculate the precursor mass from mono mz and charge
    """
    M_PROTON = 1.00727646687
    prec_mass = mono_mz * abs(charge) - charge * M_PROTON

    return prec_mass

def check_sanity(
    mass_list
):
    """
    Sanity check for mass list to make sure the masses are sorted
    """
    if not all(
        mass_list[0][i] <= mass_list[0][i + 1] for i in range(len(mass_list[0]) - 1)
    ):
        raise ValueError("Masses are not sorted.")