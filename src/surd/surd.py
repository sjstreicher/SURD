import itertools
import warnings
from itertools import combinations
from typing import Dict, Tuple

import matplotlib.colors as mcolors
import numpy as np
import pymp
from tqdm import tqdm

from .utils import it_tools as it

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)


def surd(hist: np.ndarray) -> Tuple[Dict, Dict, Dict, float]:
    """
    Decompose the mutual information between a target variable and a set
    of agent variables into three terms: Redundancy (I_R), Synergy (I_S),
    and Unique (I_U) information.

    The surd function is designed to compute a decomposition of
    the mutual information between a target variable T (signal in the future)
    and agent variables A (signals in the present). This decomposition results
    in terms related to redundancy (overlapping information), synergy
    (information that arises only when considering multiple variables together),
    and unique information.

    Parameters:
    - hist (np.ndarray): A multi-dimensional array of the histogram, where the first dimension
      represents the target variable, and subsequent dimensions represent agent variables.

    Returns:
    - rd (dict): Redundancies and unique information for each variable combination.
    - sy (dict): Synergies for each variable combination.
    - mi (dict): Mutual information for each variable combination.
    - info_leak (float): Estimation of the information leak

    Example: To understand the mutual information between target variable T and
    a combination of agent variables A1, A2, and A3, you can use:
    rd, sy, mi, info_leak = surd(hist)
    """

    # Ensure no zero values in the probability distribution to avoid NaNs during log computations
    hist += 1e-14
    # Normalize the distribution
    hist /= hist.sum()

    # Total number of dimensions (target + agents)
    ntot = hist.ndim
    # Number of agent variables
    nvars = ntot - 1
    # Number of states for the target variable
    nstates = hist.shape[0]
    agent_indices = range(1, ntot)

    # Calculation of information leak
    target_entropy = it.entropy_nvars(hist, (0,))
    target_cond_entropy = it.cond_entropy(hist, (0,), range(1, ntot))
    info_leak = target_cond_entropy / target_entropy

    # Compute the marginal distribution of the target variable
    target_marginal_dist = hist.sum(axis=(*agent_indices,), keepdims=True)

    # Prepare for specific mutual information computation
    agent_combinations, specific_mi = [], {}

    # Iterate over all combinations of agent variables
    for num_agents in agent_indices:
        for agent_combination in list(combinations(agent_indices, num_agents)):
            agent_combinations.append(agent_combination)
            excluded_agent_indices = tuple(set(agent_indices) - set(agent_combination))

            # Compute joint and conditional distributions for current combinations
            p_a = hist.sum(axis=(0, *excluded_agent_indices), keepdims=True)
            p_as = hist.sum(axis=excluded_agent_indices, keepdims=True)

            p_a_s = p_as / target_marginal_dist
            p_s_a = p_as / p_a

            # Compute specific mutual information
            specific_mi[agent_combination] = (
                (p_a_s * (it.safe_log(p_s_a) - it.safe_log(target_marginal_dist)))
                .sum(axis=agent_combination)
                .ravel()
            )

    # Compute mutual information for each combination of agent variables
    mi = {
        k: (specific_mi[k] * target_marginal_dist.squeeze()).sum()
        for k in specific_mi.keys()
    }

    # Initialize redundancy and synergy terms
    rd = {cc: 0 for cc in agent_combinations}
    sy = {cc: 0 for cc in agent_combinations[nvars:]}

    # Process each value of the target variable
    for state_index in range(nstates):
        # Extract specific mutual information for the current target value
        state_specific_mi = np.array([ii[state_index] for ii in specific_mi.values()])

        # Sorting specific mutual information
        sorted_state_specific_mi = np.argsort(state_specific_mi)
        lab = [agent_combinations[i] for i in sorted_state_specific_mi]
        lens = np.array([len(l) for l in lab])

        # Update specific mutual information based on existing maximum values
        state_specific_mi = state_specific_mi[sorted_state_specific_mi]
        for l in range(1, lens.max()):
            inds_l2 = np.where(lens == l + 1)[0]
            Il1max = state_specific_mi[lens == l].max()
            inds_ = inds_l2[state_specific_mi[inds_l2] < Il1max]
            state_specific_mi[inds_] = 0

        # Recompute sorting of updated specific mutual information values
        sorted_state_specific_mi = np.argsort(state_specific_mi)
        lab = [lab[i_] for i_ in sorted_state_specific_mi]

        # Compute differences in sorted specific mutual information values
        sorted_mi_diffs = np.diff(
            state_specific_mi[sorted_state_specific_mi], prepend=0.0
        )
        red_vars = list(agent_indices)

        # Distribute mutual information to redundancy and synergy terms
        for i_, ll in enumerate(lab):
            info = sorted_mi_diffs[i_] * target_marginal_dist.squeeze()[state_index]
            if len(ll) == 1:
                rd[tuple(red_vars)] += info
                red_vars.remove(ll[0])
            else:
                sy[ll] += info

    return rd, sy, mi, info_leak


def surd_hd(data: np.ndarray, nbins: int, max_combs: int) -> Tuple[Dict, Dict, Dict]:
    """
    Extension of surd to high-dimensional systems. It computes the decomposition of
    information up to a given number of maximum combination between variables.

    Parameters:
    - data (np.ndarray): A multi-dimensional array with the temporal evolution of the variables.
    The first dimension represents the target variable, and subsequent dimensions represent
    agent variables.
    - nbins: Number of bins to discretize the histogram.
    - max_combs: maximum order of combinations for synergistic contributions

    Returns:
    - rd (dict): Redundancies and unique information for each variable combination.
    - sy (dict): Synergies for each variable combination.
    - mi (dict): Mutual information for each variable combination.

    Example: To understand the mutual information between target variable T and
    a combination of agent variables A1, A2, and A3, you can use:
    rd, sy, mi = surd_hd(data, nbins, max_combs)
    """

    # Total number of dimensions (target + agents)
    ntot = data.shape[0]
    # Number of agent variables
    nvars = ntot - 1
    # Limit the maximum number of combinations to max_combs
    max_inds = range(1, max_combs + 1)
    agent_indices = range(1, ntot)

    # Compute the marginal distribution of the target variable
    p_target = it.safe_histogram(data[0, :].T, nbins)
    p_target = p_target.reshape((nbins,) + (1,) * (ntot - 1))

    # Prepare for specific mutual information computation
    combs, specific_mi = [], {}
    red_combs = []

    # Iterate over all combinations of agent variables
    for num_agents in max_inds:
        for agent_combination in list(combinations(agent_indices, num_agents)):
            combs.append(agent_combination)

            shape = np.ones(ntot, dtype=int)

            # Compute joint distributions for current combinations
            p_a = it.safe_histogram(data[agent_combination, :].T, nbins)
            for index in agent_combination:
                shape[index] = nbins
            p_a = p_a.reshape(tuple(shape))
            p_as = it.safe_histogram(data[(0,) + agent_combination, :].T, nbins)
            shape[0] = nbins
            p_as = p_as.reshape(tuple(shape))

            # Compute conditional distributions for current combinations
            p_a_s = p_as / p_target
            p_s_a = p_as / p_a

            # Compute specific mutual information
            specific_mi[agent_combination] = (
                (p_a_s * (it.safe_log(p_s_a) - it.safe_log(p_target)))
                .sum(axis=agent_combination)
                .ravel()
            )

    # Compute mutual information for each combination of agent variables
    mi = {k: (specific_mi[k] * p_target.squeeze()).sum() for k in specific_mi.keys()}

    # Initialize redundancy and synergy terms
    for num_agents in agent_indices:
        for agent_combination in list(combinations(agent_indices, num_agents)):
            red_combs.append(agent_combination)
    rd = {cc: 0 for cc in red_combs}
    sy = {cc: 0 for cc in combs[nvars:]}

    # Process each value of the target variable
    for t in range(nbins):
        # Extract specific mutual information for the current target value
        I1 = np.array([ii[t] for ii in specific_mi.values()])

        # Sorting specific mutual information
        i1 = np.argsort(I1)
        lab = [combs[i_] for i_ in i1]
        lens = np.array([len(l) for l in lab])

        # Update specific mutual information based on existing maximum values
        I1 = I1[i1]
        for l in range(1, lens.max()):
            inds_l2 = np.where(lens == l + 1)[0]
            Il1max = I1[lens == l].max()
            inds_ = inds_l2[I1[inds_l2] < Il1max]
            I1[inds_] = 0

        # Recompute sorting of updated specific mutual information values
        i1 = np.argsort(I1)
        lab = [lab[i_] for i_ in i1]

        # Compute differences in sorted specific mutual information values
        Di = np.diff(I1[i1], prepend=0.0)
        red_vars = list(agent_indices)

        # Distribute mutual information to redundancy and synergy terms
        for i_, ll in enumerate(lab):
            info = Di[i_] * p_target.squeeze()[t]
            if len(ll) == 1:
                rd[tuple(red_vars)] += info
                red_vars.remove(ll[0])
            else:
                sy[ll] += info

    return rd, sy, mi


def plot(rd, sy, info_leak, axs, nvars, threshold=0):
    """
    This function computes and plots information flux for given data.
    :param rd: Data for redundant contribution
    :param sy: Data for synergistic contribution
    :param axs: Axes for plotting
    :param colors: Colors for redundant, unique and synergistic contributions
    :param nvars: Number of variables
    :param threshold: Threshold as a percentage of the maximum value to select contributions to plot
    """
    colors = {}
    colors["redundant"] = mcolors.to_rgb("#003049")
    colors["unique"] = mcolors.to_rgb("#d62828")
    colors["synergistic"] = mcolors.to_rgb("#f77f00")

    for key, value in colors.items():
        rgb = mcolors.to_rgb(value)
        colors[key] = tuple([c + (1 - c) * 0.4 for c in rgb])

    # Generate keys and labels
    # Redundant Contributions
    rd_keys = []
    rd_labels = []
    for r in range(nvars, 0, -1):
        for comb in combinations(range(1, nvars + 1), r):
            prefix = "U" if len(comb) == 1 else "R"
            rd_keys.append(prefix + "".join(map(str, comb)))
            rd_labels.append(f"$\\mathrm{{{prefix}}}{{{''.join(map(str, comb))}}}$")

    # Synergistic Contributions
    sy_keys = [
        "S" + "".join(map(str, comb))
        for r in range(2, nvars + 1)
        for comb in combinations(range(1, nvars + 1), r)
    ]

    sy_labels = [
        f"$\\mathrm{{S}}{{{''.join(map(str, comb))}}}$"
        for r in range(2, nvars + 1)
        for comb in combinations(range(1, nvars + 1), r)
    ]

    label_keys, labels = (
        rd_keys + sy_keys,
        rd_labels + sy_labels,
    )

    # Extracting and normalizing the values of information measures
    values = [
        (
            rd.get(tuple(map(int, key[1:])), 0)
            if "U" in key or "R" in key
            else sy.get(tuple(map(int, key[1:])), 0)
        )
        for key in label_keys
    ]
    values /= sum(values)
    # max_value = max(values)

    # Filtering based on threshold
    labels = [label for value, label in zip(values, labels) if value >= threshold]
    values = [value for value in values if value > threshold]

    # Plotting the bar graph of information measures
    for label, value in zip(labels, values):
        if "U" in label:
            color = colors["unique"]
        elif "S" in label:
            color = colors["synergistic"]
        else:
            color = colors["redundant"]
        axs[0].bar(label, value, color=color, edgecolor="black", linewidth=1.5)

    if nvars == 2:
        axs[0].set_box_aspect(1 / 2.5)
    else:
        axs[0].set_box_aspect(1 / 4)

    # Plotting the information leak bar
    axs[1].bar(" ", info_leak, color="gray", edgecolor="black")
    axs[1].set_ylim([0, 1])
    axs[0].set_yticks([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.0])

    # change all spines
    for axis in ["top", "bottom", "left", "right"]:
        axs[0].spines[axis].set_linewidth(2)
        axs[1].spines[axis].set_linewidth(2)

    # increase tick width
    axs[0].tick_params(width=3)
    axs[1].tick_params(width=3)

    return dict(zip(label_keys, values))


def plot_nlabels(rd, sy, info_leak, axs, nvars, nlabels=-1):
    """
    This function computes and plots information flux for given data.
    :param rd: Data for redundant contribution
    :param sy: Data for synergistic contribution
    :param axs: Axes for plotting
    :param colors: Colors for redundant, unique and synergistic contributions
    :param nvars: Number of variables
    :param threshold: Threshold as a percentage of the maximum value to select contributions to plot
    """
    colors = {}
    colors["redundant"] = mcolors.to_rgb("#003049")
    colors["unique"] = mcolors.to_rgb("#d62828")
    colors["synergistic"] = mcolors.to_rgb("#f77f00")

    for key, value in colors.items():
        rgb = mcolors.to_rgb(value)
        colors[key] = tuple([c + (1 - c) * 0.4 for c in rgb])

    # Generate keys and labels
    # Redundant Contributions
    rd_keys = []
    rd_labels = []
    for r in range(nvars, 0, -1):
        for comb in combinations(range(1, nvars + 1), r):
            prefix = "U" if len(comb) == 1 else "R"
            rd_keys.append(prefix + "".join(map(str, comb)))
            rd_labels.append(f"$\\mathrm{{{prefix}}}{{{''.join(map(str, comb))}}}$")

    # Synergistic Contributions
    sy_keys = [
        "S" + "".join(map(str, comb))
        for r in range(2, nvars + 1)
        for comb in combinations(range(1, nvars + 1), r)
    ]

    sy_labels = [
        f"$\\mathrm{{S}}{{{''.join(map(str, comb))}}}$"
        for r in range(2, nvars + 1)
        for comb in combinations(range(1, nvars + 1), r)
    ]

    label_keys, labels = (
        rd_keys + sy_keys,
        rd_labels + sy_labels,
    )

    # Extracting and normalizing the values of information measures
    values = [
        (
            rd.get(tuple(map(int, key[1:])), 0)
            if "U" in key or "R" in key
            else sy.get(tuple(map(int, key[1:])), 0)
        )
        for key in label_keys
    ]
    values /= sum(values)
    # max_value = max(values)

    # Filtering based on threshold
    top_n_indices = np.argsort(values)[-nlabels:]

    # Filter both the values and labels arrays
    filtered_values = values[top_n_indices]
    filtered_labels = np.array(labels)[top_n_indices]
    original_order_indices = np.argsort(top_n_indices)
    filtered_values_in_original_order = filtered_values[original_order_indices]
    filtered_labels_in_original_order = filtered_labels[original_order_indices]

    # Convert filtered arrays back to lists if necessary
    values = filtered_values_in_original_order
    labels = filtered_labels_in_original_order.tolist()

    # Plotting the bar graph of information measures
    for label, value in zip(labels, values):
        if "U" in label:
            color = colors["unique"]
        elif "S" in label:
            color = colors["synergistic"]
        else:
            color = colors["redundant"]
        axs[0].bar(label, value, color=color, edgecolor="black", linewidth=1.5)

    axs[0].set_box_aspect(1 / 4)

    # Plotting the information leak bar
    axs[1].bar(" ", info_leak, color="gray", edgecolor="black")
    axs[1].set_ylim([0, 1])
    axs[0].set_yticks([0.0, 1.0])
    axs[0].set_ylim([0.0, 1.0])

    # change all spines
    for axis in ["top", "bottom", "left", "right"]:
        axs[0].spines[axis].set_linewidth(2)
        axs[1].spines[axis].set_linewidth(2)

    # increase tick width
    axs[0].tick_params(width=3)
    axs[1].tick_params(width=3)

    return dict(zip(label_keys, values))


def nice_print(r_, s_, mi_, leak_):
    """Print the normalized redundancies, unique and synergy particles"""

    r_ = {key: value / max(mi_.values()) for key, value in r_.items()}
    s_ = {key: value / max(mi_.values()) for key, value in s_.items()}

    print("    Redundant (R):")
    for k_, v_ in r_.items():
        if len(k_) > 1:
            print(f"        {str(k_):12s}: {v_:5.4f}")

    print("    Unique (U):")
    for k_, v_ in r_.items():
        if len(k_) == 1:
            print(f"        {str(k_):12s}: {v_:5.4f}")

    print("    Synergistic (S):")
    for k_, v_ in s_.items():
        print(f"        {str(k_):12s}: {v_:5.4f}")

    print(f"    Information Leak: {leak_ * 100:5.2f}%")


def run(data: np.ndarray, nvars: int, nlag: int, nbins: int, axs):

    information_flux = {}

    for i in tqdm(range(nvars), desc="Processing variables"):
        print(f"SURD CAUSALITY FOR SIGNAL {i+1}")

        # Organize data (0 target variable, 1: agent variables)
        organised_data = np.vstack([data[i, nlag:], data[:, :-nlag]])

        # Run SURD
        hist, _ = np.histogramdd(organised_data.T, nbins)
        rd, sy, mi, info_leak = surd(hist)

        # Print results
        nice_print(rd, sy, mi, info_leak)

        # Plot SURD
        information_flux[i + 1] = plot(
            rd,
            sy,
            info_leak,
            axs[i, :],
            nvars,
            threshold=-0.01,
        )

        # Plot formatting
        axs[i, 0].set_title(
            f"${{\\Delta I}}_{{(\\cdot) \\rightarrow {i+1}}} / I \\left(Q_{i+1}^+ ; \\mathrm{{\\mathbf{{Q}}}} \\right)$",
            pad=12,
        )
        axs[i, 1].set_title(
            f"$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow {i+1}}}}}{{H \\left(Q_{i+1} \\right)}}$",
            pad=20,
        )
        axs[i, 0].set_xticklabels(
            axs[i, 0].get_xticklabels(),
            fontsize=20,
            rotation=60,
            ha="right",
            rotation_mode="anchor",
        )
        print("\n")

    # Show the results
    for i in range(0, nvars - 1):
        axs[i, 0].set_xticklabels("")


def run_parallel(data, nvars, nlag, nbins, axs):

    information_flux = {}
    rd_results = pymp.shared.dict({})  # Dictionary to store redundant contributions
    sy_results = pymp.shared.dict({})  # Dictionary to store synergistic contributions
    mi_results = pymp.shared.dict({})  # Dictionary to store mutual information results
    info_leak_results = pymp.shared.dict(
        {}
    )  # Dictionary to store information leak results

    with pymp.Parallel(nvars) as par:
        for i in par.range(nvars):

            # Organize data (0 target variable, 1: agent variables)
            organised_data = np.vstack([data[i, nlag:], data[:, :-nlag]])

            # Run SURD
            hist, _ = np.histogramdd(organised_data.T, nbins)
            rd, sy, mi, info_leak = surd(hist)

            # Print results
            print(f"SURD CAUSALITY FOR SIGNAL {i+1}")
            nice_print(rd, sy, mi, info_leak)
            print("\n")

            # Save the results
            (
                rd_results[i + 1],
                sy_results[i + 1],
                mi_results[i + 1],
                info_leak_results[i + 1],
            ) = (rd, sy, mi, info_leak)

    for i in range(nvars):
        # Plot SURD
        information_flux[i + 1] = plot(
            rd_results[i + 1],
            sy_results[i + 1],
            info_leak_results[i + 1],
            axs[i, :],
            nvars,
            threshold=-0.01,
        )

        # Plot formatting
        axs[i, 0].set_title(
            f"${{\\Delta I}}_{{(\\cdot) \\rightarrow {i+1}}} / I \\left(Q_{i+1}^+ ; \\mathrm{{\\mathbf{{Q}}}} \\right)$",
            pad=12,
        )
        axs[i, 1].set_title(
            f"$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow {i+1}}}}}{{H \\left(Q_{i+1} \\right)}}$",
            pad=20,
        )
        axs[i, 0].set_xticklabels(
            axs[i, 0].get_xticklabels(),
            fontsize=20,
            rotation=60,
            ha="right",
            rotation_mode="anchor",
        )

    # Show the results
    for i in range(0, nvars - 1):
        axs[i, 0].set_xticklabels("")


def plot_multiple_lags(I_R, I_S, info_leak, axs, n_vars_lag, n_lag, threshold=0):
    """
    This function computes and plots information flux for given data.
    :param I_R: Data for redundant contribution
    :param I_S: Data for synergistic contribution
    :param axs: Axis for plotting
    :param n_vars_lag: Number of variables including lags
    :param n_lag: Number of lags
    :param threshold: Threshold as a percentage of the maximum value to select contributions to plot
    """
    colors = {}
    colors["redundant"] = mcolors.to_rgb("#003049")
    colors["unique"] = mcolors.to_rgb("#d62828")
    colors["synergistic"] = mcolors.to_rgb("#f77f00")

    for key, value in colors.items():
        rgb = mcolors.to_rgb(value)
        colors[key] = tuple([c + (1 - c) * 0.4 for c in rgb])

    # Generate keys and labels
    n_vars = n_vars_lag // n_lag

    # Redundant Contributions
    I_R_keys = []
    I_R_labels = []
    # for r in range(1, n_vars_lag + 1):
    for r in range(n_vars_lag, 0, -1):
        for comb in itertools.combinations(range(1, n_vars_lag + 1), r):
            prefix = "U" if len(comb) == 1 else "R"
            I_R_keys.append(prefix + "".join(map(str, comb)))

            # New label generation with subscripts for lags
            new_comb_labels = []
            for c in comb:
                lag_number = (c - 1) // n_vars
                var_number = (c - 1) % n_vars + 1
                new_label = f"{var_number}_{{{lag_number+1}}}"
                new_comb_labels.append(new_label)

            I_R_labels.append(f"$\\mathrm{{{prefix}}}{{{''.join(new_comb_labels)}}}$")

    # Synergistic Contributions
    I_S_keys = []
    I_S_labels = []
    for r in range(
        2, n_vars_lag + 1
    ):  # Starting from 2 because synergistic contributions require at least two variables
        for comb in itertools.combinations(range(1, n_vars_lag + 1), r):
            # Generating the key
            I_S_keys.append("S" + "".join(map(str, comb)))

            # Generating the label with subscripts for lags
            new_comb_labels = []
            for c in comb:
                lag_number = (c - 1) // n_vars
                var_number = (c - 1) % n_vars + 1
                new_label = f"{var_number}_{{{lag_number+1}}}"
                new_comb_labels.append(new_label)

            I_S_labels.append(f"$\\mathrm{{S}}{{{''.join(new_comb_labels)}}}$")

    label_keys, labels = I_R_keys + I_S_keys, I_R_labels + I_S_labels

    # Extracting and normalizing the values of information measures
    values = [
        (
            I_R.get(tuple(map(int, key[1:])), 0)
            if "U" in key or "R" in key
            else I_S.get(tuple(map(int, key[1:])), 0)
        )
        for key in label_keys
    ]
    values /= sum(values)
    # max_value = max(values)

    # Filtering based on threshold
    labels = [label for value, label in zip(values, labels) if value >= threshold]
    values = [value for value in values if value > threshold]

    # Plotting the bar graph of information measures
    for label, value in zip(labels, values):
        if "U" in label:
            color = colors["unique"]
        elif "S" in label:
            color = colors["synergistic"]
        else:
            color = colors["redundant"]
        axs[0].bar(label, value, color=color, edgecolor="black", linewidth=1.5)

    # Plotting the bar graph of information measures
    axs[0].set_xticks(range(len(values)))
    axs[0].set_xticklabels(
        labels, fontsize=15, rotation=60, ha="right", rotation_mode="anchor"
    )
    axs[0].set_box_aspect(1 / 5)

    # Plotting the information leak bar
    axs[1].bar(" ", info_leak, color="gray", edgecolor="black")
    axs[1].set_ylim([0, 1])

    # change all spines
    for axis in ["top", "bottom", "left", "right"]:
        axs[0].spines[axis].set_linewidth(1.5)
        axs[1].spines[axis].set_linewidth(1.5)

    # increase tick width
    axs[0].tick_params(width=1.5)
    axs[1].tick_params(width=1.5)

    return dict(zip(label_keys, values))


def run_multiple_lags(X, nvars, nlag, nbins, max_combs, axs):
    "Run SURD causality for different lags (from lag 1 up to nlag)"
    information_flux = {}

    for i in tqdm(range(nvars), desc="Processing variables"):
        print(f"SURD CAUSALITY FOR SIGNAL {i+1}")

        # Organize data (0 target variable, 1: agent variables)
        Y = X[i, nlag + 1 :]
        # Create the lagged versions of X and append to the list
        for lag in range(nlag, 0, -1):
            Y = np.vstack([Y, X[:, lag : -nlag + lag - 1]])

        # Run SURD
        I_R, I_S, MI = surd_hd(Y, nbins, max_combs)

        # Calculate information leak
        hist = it.safe_histogram(Y[0, :].T, nbins)
        H = it.entropy_nvars(hist, (0,))
        info_leak = 1 - (sum(I_R.values()) + sum(I_S.values())) / H

        # Print results
        nice_print(I_R, I_S, MI, info_leak)
        print("\n")

        # Plot SURD
        information_flux[i + 1] = plot_multiple_lags(
            I_R, I_S, info_leak, axs[i, :], nvars * nlag, nlag, threshold=-0.01
        )

        # Plot formatting
        axs[i, 0].set_title(
            f"${{\\Delta I}}_{{(\\cdot) \\rightarrow {i+1}}} / I \\left(Q_{i+1}^+ ; \\mathrm{{\\mathbf{{Q}}}} \\right)$",
            pad=10,
        )
        axs[i, 1].set_title(
            f"$\\frac{{{{\\Delta I}}_{{\\mathrm{{leak}} \\rightarrow {i+1}}}}}{{H \\left(Q_{i+1} \\right)}}$",
            pad=18,
        )
        axs[i, 1].set_yticks([0, 1])
        axs[i, 0].set_xticklabels(
            axs[i, 0].get_xticklabels(),
            fontsize=14,
            rotation=60,
            ha="right",
            rotation_mode="anchor",
        )

        # change all spines
        for axis in ["top", "bottom", "left", "right"]:
            axs[i, 0].spines[axis].set_linewidth(2.5)
            axs[i, 1].spines[axis].set_linewidth(2.5)
        axs[i, 0].set_box_aspect(1 / 4.5)

    # Show the results
    for i in range(0, nvars - 1):
        axs[i, 0].set_xticklabels("")

    return I_R, I_S, MI, info_leak
