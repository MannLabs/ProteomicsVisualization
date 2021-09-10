import logging
import pandas as pd
import numpy as np
from numba import njit
from tqdm import tqdm

# specify a configuration whether to display the Plotly logo in the toolbar and how to save the plot
config = {
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'svg', # one of png, svg, jpeg, webp
        'filename': 'custom_image',
        'height': 500,
        'width': 1200,
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
    from pyrawfilereader import RawFileReader
    
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
        
        
from numba import types
from numba.typed import Dict

# This code was taken from the AlphaPept Python package (https://github.com/MannLabs/alphapept/blob/master/nbs/03_fasta.ipynb)
#generates the mass dictionary from table
def get_mass_dict(modfile:str="Data/modifications.tsv", aasfile: str="Data/amino_acids.tsv", verbose:bool=True):
    """
    Function to create a mass dict based on tsv files.
    This is used to create the hardcoded dict in the constants notebook.
    The dict needs to be hardcoded because of importing restrictions when using numba.
    More specifically, a global needs to be typed at runtime.
    Args:
        modfile (str): Filename of modifications file.
        aasfile (str): Filename of AAs file.
        verbose (bool, optional): Flag to print dict.
    Returns:
        Returns a numba compatible dictionary with masses.
    Raises:
        FileNotFoundError: If files are not found.
    """
    import pandas as pd

    mods = pd.read_csv(modfile, delimiter="\t")
    aas = pd.read_csv(aasfile, delimiter="\t")

    mass_dict = Dict.empty(key_type=types.unicode_type, value_type=types.float64)

    for identifier, mass in aas[["Identifier", "Monoisotopic Mass (Da)"]].values:
        mass_dict[identifier] = float(mass)

    for identifier, aar, mass in mods[
        ["Identifier", "Amino Acid Residue", "Monoisotopic Mass Shift (Da)"]
    ].values:
        #print(identifier, aar, mass)

        if ("<" in identifier) or (">" in identifier):
            for aa_identifier, aa_mass in aas[["Identifier", "Monoisotopic Mass (Da)"]].values:
                if '^' in identifier:
                    new_identifier = identifier[:-2] + aa_identifier
                    mass_dict[new_identifier] = float(mass) + mass_dict[aa_identifier]
                elif aar == aa_identifier:
                    new_identifier = identifier[:-2] + aa_identifier
                    mass_dict[new_identifier] = float(mass) + mass_dict[aa_identifier]
                else:
                    pass
        else:
            mass_dict[identifier] = float(mass) + mass_dict[aar]

    # Manually add other masses
    mass_dict[
        "Electron"
    ] = (
        0.000548579909070
    )  # electron mass, half a millimass error if not taken into account
    mass_dict["Proton"] = 1.00727646687  # proton mass
    mass_dict["Hydrogen"] = 1.00782503223  # hydrogen mass
    mass_dict["C13"] = 13.003354835  # C13 mass
    mass_dict["Oxygen"] = 15.994914619  # oxygen mass
    mass_dict["OH"] = mass_dict["Oxygen"] + mass_dict["Hydrogen"]  # OH mass
    mass_dict["H2O"] = mass_dict["Oxygen"] + 2 * mass_dict["Hydrogen"]  # H2O mass

    mass_dict["NH3"] = 17.03052
    mass_dict["delta_M"] = 1.00286864
    mass_dict["delta_S"] = 0.0109135

    if verbose:

        for element in mass_dict:
            print('mass_dict["{}"] = {}'.format(element, mass_dict[element]))

    return mass_dict

import numba

@njit
def get_fragmass(parsed_pep:list, mass_dict:numba.typed.Dict)->tuple:
    """
    Calculate the masses of the fragment ions
    Args:
        parsed_pep (numba.typed.List of str): the list of amino acids and modified amono acids.
        mass_dict (numba.typed.Dict): key is the amino acid or the modified amino acid, and the value is the mass.
    Returns:
        Tuple[np.ndarray(np.float64), np.ndarray(np.int8)]: the fragment masses and the fragment types (represented as np.int8).
        For a fragment type, positive value means the b-ion, the value indicates the position (b1, b2, b3...); the negative value means
        the y-ion, the absolute value indicates the position (y1, y2, ...).
    """
    n_frags = (len(parsed_pep) - 1) * 2

    frag_masses = np.zeros(n_frags, dtype=np.float64)
    frag_type = np.zeros(n_frags, dtype=np.int8)

    # b-ions > 0
    n_frag = 0

    frag_m = mass_dict["Proton"]
    for idx, _ in enumerate(parsed_pep[:-1]):
        frag_m += mass_dict[_]
        frag_masses[n_frag] = frag_m
        frag_type[n_frag] = (idx+1)
        n_frag += 1

    # y-ions < 0
    frag_m = mass_dict["Proton"] + mass_dict["H2O"]
    for idx, _ in enumerate(parsed_pep[::-1][:-1]):
        frag_m += mass_dict[_]
        frag_masses[n_frag] = frag_m
        frag_type[n_frag] = -(idx+1)
        n_frag += 1

    return frag_masses, frag_type