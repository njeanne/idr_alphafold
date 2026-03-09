#! /usr/bin/env python3

"""
Created on 18 Jan. 2022
"""

import argparse
import logging
import os
import statistics
import sys

from Bio.PDB.PDBParser import PDBParser
from dna_features_viewer import GraphicFeature, GraphicRecord
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
import seaborn as sns

__author__ = "Nicolas JEANNE"
__copyright__ = "GNU General Public License"
__email__ = "jeanne.n@chu-toulouse.fr"
__version__ = "1.1.1"


def create_log(path, level):
    """Create the log as a text file and as a stream.

    :param path: the path of the log.
    :type path: str
    :param level: the level og the log.
    :type level: str
    :return: the logging:
    :rtype: logging
    """

    log_level_dict = {"DEBUG": logging.DEBUG,
                      "INFO": logging.INFO,
                      "WARNING": logging.WARNING,
                      "ERROR": logging.ERROR,
                      "CRITICAL": logging.CRITICAL}

    if level is None:
        log_level = log_level_dict["INFO"]
    else:
        log_level = log_level_dict[args.log_level]

    if os.path.exists(path):
        os.remove(path)

    logging.basicConfig(format="%(asctime)s %(levelname)s:\t%(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=log_level,
                        handlers=[logging.FileHandler(path), logging.StreamHandler()])
    return logging


def restricted_int(int_to_inspect):
    """Inspect if an int is between 0 and 100

    :param int_to_inspect: the int to inspect
    :type int_to_inspect: int
    :raises ArgumentTypeError: is not between 0 and 100
    :return: the int value if float_to_inspect is between 0 and 100
    :rtype: int
    """
    x = int(int_to_inspect)
    if x < 0 or x > 100:
        raise argparse.ArgumentTypeError(f"{x} not in range [0, 100]")
    return x


def extract_plddt(input_path):
    """
    Extract the pLDDT (Beta factor column) of the Alphafold PDB file by amino acid.

    :param input_path: the pdb file's path
    :type input_path: str
    :return: the residues and pLDDT dataframe.
    :rtype: Pandas.DataFrame
    """
    with open(input_path, "r") as input_pdb:
        plddt_data = {"position": [], "pLDDT": [], "amino-acid": []}
        parser_pdb = PDBParser()
        structure = parser_pdb.get_structure("old", input_pdb)
        for model in structure:
            for chain in model:
                for residue in chain.get_residues():
                    plddt_data["position"].append(residue.full_id[3][1])
                    plddt_data["pLDDT"].append(list(residue.get_atoms())[0].get_bfactor())
                    plddt_data["amino-acid"].append(residue.resname)
        return pd.DataFrame(plddt_data)


def get_residue_order_state(plddt_data, threshold, window):
    """
    On a window compute the mediaan of pLDDT values and determine if the region is ordered or disordered from the
    center of the window to each side of the window.

    :param plddt_data: the dataframe of the residues pLDDT values.
    :type plddt_data: Pandas.DataFrame
    :param threshold: the threshold for a disordered region, under or equal.
    :type threshold: float
    :param window: the window size to compute the median.
    :type window: int
    :return: the updated dataframe of the residues mean pLDDT on the window which center is the residue position and if
    the residue normalized on the window is ordered or not.
    :rtype: Pandas.DataFrame
    """
    plddt_medians = []
    plddt_list = list(plddt["pLDDT"])
    if window % 2 == 0:
        idx_start = int(window / 2) - 1
        left_window = int(window / 2) - 1
        right_window = int(window / 2)
    else:
        idx_start = int(window / 2)
        left_window = int(window / 2)
        right_window = int(window / 2)

    logging.info(f"window size for disordered regions search: {window}")
    logging.info(f"threshold for disordered regions <= {threshold}%")
    logging.debug(f"index start for disordered regions search: {idx_start}")
    logging.debug(f"window left size for disordered regions search: {left_window}")
    logging.debug(f"window right size for disordered regions search: {right_window}")

    for idx in range(idx_start):
        median_plddt = statistics.median(plddt_list[0:window])
        plddt_medians.append(median_plddt)
        logging.debug(f"pLDDT mean value computed on the first {window} residues (window size), residue {idx + 1}: "
                      f"{median_plddt}")
    for idx in range(idx_start, len(plddt_list) - right_window):
        median_plddt = statistics.median(plddt_list[(idx - left_window):(idx + right_window)])
        plddt_medians.append(median_plddt)
        logging.debug(f"pLDDT mean value computed on the window size ({window}), residue {idx + 1}: {median_plddt}")
    for idx in range(len(plddt_list) - right_window, len(plddt_list)):
        median_plddt = statistics.median(plddt_list[len(plddt_list)-window:len(plddt_list)])
        plddt_medians.append(median_plddt)
        logging.debug(f"pLDDT mean value computed on the last {window} residues (window size), residue {idx + 1}: "
                      f"{median_plddt}")
    plddt_data["pLDDT window median (%)"] = plddt_medians
    plddt_data["order state"] = ["ordered" if value >= threshold else "disordered" for value in plddt_medians]

    return plddt_data


def get_domains(plddt_data, domains):
    """
    Get the domains which each residue belongs to using its position.

    :param plddt_data: the dataframe of the pLDDT.
    :type plddt_data: Pandas.DataFrame
    :param domains: the protein's domains' dataframe.
    :type domains: Pandas.DataFrame
    :return: the updated dataframe with the domain which each residue belongs to.
    :rtype: Pandas.DataFrame
    """

    domains_list = []

    for pos in plddt_data["position"]:
        domain = None
        for _, row in domains.iterrows():
            if row["start"] <= pos <= row["end"]:
                domain = row["domain"]
                break
        domains_list.append(domain)

    plddt_data["domains"] = domains_list
    return plddt_data


def get_areas_order_state(plddt_data):
    """
    Get coordinates of the distinct areas depending on their order state.

    :param plddt_data: the dataframe of the pLDDT.
    :type plddt_data: Pandas.DataFrame
    :return: the dataframe of the areas depending on their order state.
    :rtype: Pandas.DataFrame
    """
    order_state_areas = {"state": [], "start": [], "end": [], "color": []}
    state_color = {"ordered": "#1500ff4e", "disordered": "#ff00004d"}
    residue_order_state = None
    previous_position = None
    row = None
    for _, row in plddt_data.iterrows():
        if not residue_order_state == row["order state"]:
            if residue_order_state is not None:
                order_state_areas["end"].append(previous_position)
            order_state_areas["state"].append(row["order state"])
            order_state_areas["start"].append(row["position"])
            order_state_areas["color"].append(state_color[row["order state"]])
        residue_order_state = row["order state"]
        previous_position = row["position"]
    # record the last area end position
    order_state_areas["end"].append(row["position"])

    return pd.DataFrame(order_state_areas)


def draw_chart_plddt(plddt_data, threshold, out_dir, prot_id, out_format, window, domains=None):
    """
    Draw the chart for the pLDDT values.

    :param plddt_data: the dataframe of the pLDDT.
    :type plddt_data: Pandas.DataFrame
    :param threshold: the order / disorder pLDDT threshold.
    :type threshold: int
    :param out_dir: the chart directory output path.
    :type out_dir: str
    :param prot_id: the name of the protein.
    :type prot_id: str
    :param out_format: the format of the output chart file.
    :type out_format: str
    :param window: the window size to compute the median.
    :type window: int
    :param domains: the protein's domains' dataframe.
    :type domains: Pandas.DataFrame
    :return: the chart directory output path.
    :rtype: str
    """
    out_path = os.path.join(out_dir, f"pLDDT_{prot_id}.{out_format}")

    # create the plddt plot and the domains' map
    fig, axs = plt.subplots(2, 1, layout="constrained", height_ratios=[10, 1])

    # pLDDT line plot
    plddt_chart = sns.lineplot(data=plddt_data, x="position", y="pLDDT window median (%)", color="black", ax=axs[0])

    # add the order state background
    areas_order_state = get_areas_order_state(plddt_data)
    for i, row in areas_order_state.iterrows():
        plddt_chart.axvspan(xmin=row["start"], xmax=row["end"], color=row["color"])
    legend_elements = [Patch(facecolor="#1500ff4e", label="Ordered"), Patch(facecolor="#ff00004d", label="Disordered")]
    # add the threshold horizontal line
    plddt_chart.axhline(y=threshold, color="red")
    # set the legends, axis and titles
    axs[0].legend(handles=legend_elements, loc="lower left")
    axs[0].set_xlabel("amino-acids positions", fontweight="bold")
    axs[0].set_ylim(1, 100)
    axs[0].set_ylabel(f"median of the pLDDT (%)", fontweight="bold")
    axs[0].set_title(f"{prot_id}: median of the pLDDT on a {window} residues window", fontweight="bold")

    # Domains' plot
    if domains is not None:
        features = []
        row = None
        for _, row in domains.iterrows():
            features.append(GraphicFeature(start=row["start"], end=row["end"], strand=+1, color=row["color"],
                                           label=row["domain"]))
        # set the last residue for the X axes 0 superior limit matches with the domains' representation
        axs[0].set_xlim(plddt_data.iloc[0]["position"], row["end"] + 1)

        record = GraphicRecord(sequence_length=row["end"] + 1, features=features, plots_indexing="genbank")
        ax_domains, _ = record.plot(ax=axs[1])

    plot = plddt_chart.get_figure()
    plot.savefig(out_path)

    return out_path


if __name__ == "__main__":
    descr = f"""
    {os.path.basename(__file__)} v. {__version__}

    Created by {__author__}.
    Contact: {__email__}
    {__copyright__}

    Distributed on an "AS IS" basis without warranties or conditions of any kind, either express or
    implied.

    Intrinsically Disordered Regions visualisation on Alphafold outputs, based on the pLDTT (predicted Local Distance 
    Difference Test) values in the .pdb files of the predicted structures.
    See Jumper et al. 2021, Suppl. Methods 1.9.6 and Mariani et al. 2013 Bioinformatics for details.
    
    An optional Comma Separated Values file can be provided with the AA coordinates of the different regions. The file
    must have a formatted header with the following values: \"domain,start,end,color\".
    - domain:   the domain name
    - start:    the domain AA start (1-index)
    - end:      the domain AA end (1-index)
    - color:    the color used to display the domain (hexadecimal format)
    """
    parser = argparse.ArgumentParser(description=descr, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out", required=True, type=str, help="path to the output directory.")
    parser.add_argument("-f", "--format", required=False, type=str, default="svg",
                        choices=["svg", "html"], help="the output format of the plot, default is SVG.")
    parser.add_argument("-d", "--domains", required=False, type=str,
                        help="CSV (comma separated) file defining the domains name, the start position, the stop "
                             "position and the color representation in hexadecimal format.")
    parser.add_argument("-w", "--window-size", required=False, type=int, default=11,
                        help="the window size to determine if the mean of the pLDDT values in the window are ordered "
                             "or disordered, default is 11.")
    parser.add_argument("-t", "--threshold", required=False, type=restricted_int, default=50,
                        help="the threshold percentage of disorder for plDDT. If the pLDDT is under or equal to this "
                             "threshold the pLDDT value is set as disordered. Default value is 50%%.")
    parser.add_argument("-l", "--log", required=False, type=str,
                        help=("the path for the log file. If this option is  skipped, the log file is created in the "
                              "output directory."))
    parser.add_argument("--log-level", required=False, type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="set the log level. If this option is skipped, the log level is INFO.")
    parser.add_argument("-v", "--version", action="version", version=__version__)
    parser.add_argument("input", type=str, help="path to the Alphafold prediction .pdb file.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # create the logger
    if args.log:
        log_path = args.log
    else:
        log_path = os.path.join(args.out, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
    create_log(log_path, args.log_level)

    logging.info(f"version: {__version__}")
    logging.info(f"CMD: {' '.join(sys.argv)}")

    # set the seaborn plots theme and size
    sns.set_theme()
    rcParams["figure.figsize"] = 15, 12

    alphafold_prediction_id = os.path.splitext(os.path.basename(args.input))[0]
    plddt = extract_plddt(args.input)
    plddt = get_residue_order_state(plddt, float(args.threshold), args.window_size)

    if args.domains:
        df_domains = pd.read_csv(args.domains, sep=",", header=0)
        plddt = get_domains(plddt, df_domains)
        path_chart = draw_chart_plddt(plddt, args.threshold, args.out, alphafold_prediction_id, args.format,
                                      args.window_size, df_domains)
    else:
        path_chart = draw_chart_plddt(plddt, args.threshold, args.out, alphafold_prediction_id, args.format,
                                      args.window_size)

    logging.info(f"pLDDT chart for {alphafold_prediction_id} created: {os.path.abspath(path_chart)}")
    path_data = os.path.join(args.out, f"pLDDT_{alphafold_prediction_id}.csv")
    plddt.to_csv(path_data, index=False)
    logging.info(f"pLDDT data for {alphafold_prediction_id} created: {os.path.abspath(path_data)}")
