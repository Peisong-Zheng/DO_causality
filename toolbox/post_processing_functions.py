import numpy as np

def print_significant_links(N, var_names, p_matrix, val_matrix, alpha_level=0.05):
    """Generates and returns significant links based on p-values and values matrices.

    Parameters
    ----------
    N : int
        The number of variables (typically, this is the shape of the first dimension of p_matrix and val_matrix).
    var_names : list of str
        Names of the variables corresponding to the dimensions of p_matrix and val_matrix.
    alpha_level : float, optional (default: 0.05)
        Significance level.
    p_matrix : array-like
        P-values matrix of shape (N, N, tau_max + 1).
    val_matrix : array-like
        Values matrix of shape (N, N, tau_max + 1).
    
    Returns
    -------
    str
        A string containing formatted significant links information.
    """
    sig_links = p_matrix <= alpha_level
    results_text = "\n## Significant links at alpha = %s:\n" % alpha_level

    for j in range(N):
        links = {(p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])])
                 for p in zip(*np.where(sig_links[:, j, :]))}
        # Sort by value
        sorted_links = sorted(links, key=links.get, reverse=True)
        n_links = len(links)
        string = "    Variable %s has %d link(s):" % (var_names[j], n_links)
        for p in sorted_links:
            string += "\n        (%s % d): pval = %.5f" % (
                var_names[p[0]], p[1], p_matrix[p[0], j, abs(p[1])])
            string += " | val = % .3f" % (val_matrix[p[0], j, abs(p[1])])
        results_text += string + "\n"
        print(string)

    return results_text


 

import re

def parse_results_to_dict(results_text, var_names):
    causal_links = {}
    lines = results_text.split('\n')
    
    current_var = None
    for line in lines:
        print(repr(line))  # Use repr to show hidden characters like tabs or multiple spaces

        if "Variable" in line:
            match = re.search(r"Variable\s+(.*?)\s+has", line)  # More flexible with spaces
            if match:
                current_var_name = match.group(1)
                current_var = var_names.index(current_var_name)
                print("target var:", current_var_name)
                causal_links[current_var_name] = []

        elif "pval =" in line:
            cause_match = re.search(r"\((.*?)\)", line)
            print("Cause match found:", cause_match)
            pval_match = re.search(r"pval = ([\d\.]+)", line)
            # More robust regex, considering possible extra spaces and different formatting
            val_match = re.search(r"\|\s*val\s*=\s*([-\d\.]+)", line)

            print("Val match found:", val_match)

            lag_match = re.search(r"(\-\d+)", cause_match.group(1))
            
            if cause_match and pval_match and val_match and lag_match:
                cause_name = cause_match.group(1).split(' ')[0]+' '+ cause_match.group(1).split(' ')[1]  # Ensure correct format
                print(cause_match.group(1).split(' '))
                print("Cause name:", cause_name)
                cause_index = var_names.index(cause_name)
                print("Cause index:", cause_index)
                link_detail = {
                    "cause": cause_name,
                    "lag": int(lag_match.group(1)),
                    "pval": float(pval_match.group(1)),
                    "val": float(val_match.group(1))
                }
                causal_links[current_var_name].append(link_detail)
    
    return causal_links



def sort_causal_links_by_val(causal_links, ascending=True):
    sorted_causal_links = {}
    for key, links in causal_links.items():
        sorted_links = sorted(links, key=lambda x: abs(x['val']), reverse=not ascending)
        sorted_causal_links[key] = sorted_links
    return sorted_causal_links

def clean_causal_links(causal_links):
    cleaned_links = {}
    for target, links in causal_links.items():
        target_prefix = target.split(' ')[0]  
        # print("target prefix:", target_prefix)
        # Filter links where the cause prefix matches the target prefix
        filtered_links = [link for link in links if not link['cause'].startswith(target_prefix)]
        cleaned_links[target] = filtered_links
    return cleaned_links

# remove items with abs(val) < 0.1
def filter_causal_links_by_val(causal_links, threshold=0.1):
    filtered_links = {}
    for target, links in causal_links.items():
        filtered_links[target] = [link for link in links if abs(link['val']) >= threshold]
    return filtered_links



import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_inter_var_causal_links(causal_links):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges based on the causal links
    for target, causes in causal_links.items():
        for cause in causes:
            # Check if the edge already exists, and if so, update the weight attribute
            if G.has_edge(cause['cause'], target):
                G[cause['cause']][target]['weight'].append(cause['val'])
            else:
                G.add_edge(cause['cause'], target, weight=[cause['val']])
    
    # Define the circular layout for the nodes
    pos = nx.circular_layout(G)
    
    # Extract all weights to flatten the list for multiple edges
    weights = []
    for u, v, data in G.edges(data=True):
        weights.extend(data['weight'])
    
    # Prepare edge colors based on their 'val' attribute, using the RdBu_r colormap
    edge_cmap = plt.cm.RdBu_r
    
    # Normalize the color range
    norm = plt.Normalize(-1, 1)

    # Create figure and axis objects
    fig, ax = plt.subplots(dpi=600)
    
    # Draw nodes and edges with specified settings
    nx.draw_networkx_nodes(G, pos, node_color='grey', node_size=900, ax=ax)
    for u, v, data in G.edges(data=True):
        for idx, weight in enumerate(data['weight']):
            # Creating a slight arc for each edge based on its index to separate multiple edges
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                                   connectionstyle=f'arc3,rad={0.1*len(data["weight"])-0.2*idx}',
                                   edge_color=edge_cmap(norm(weight)), width=2, 
                                   arrowstyle='-|>', arrowsize=20, node_size=900)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', ax=ax)
    
    # Add a colorbar with the correct axis
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Causal Link Strength (val)')
    
    # Title and axis settings
    # plt.title('Inter-Variable Causal Links')
    ax.axis('off')  # Turn off the axis
    plt.tight_layout()
    plt.show()






import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import matplotlib.cm as cm
from matplotlib.patches import ConnectionPatch


def plot_vars_class_labels(datasets, causal_links, dpi=100):
    """
    Plots global maps of class labels for provided xarray datasets, with a single, independent color bar.
    
    Parameters:
    datasets : dict
        Dictionary of xarray datasets with keys indicating the dataset names ('sat', 'pre', 'sic').
    """
    # Calculate the maximum number of unique classes across all datasets
    max_classes = int(max(ds['class_label'].max() for ds in datasets.values()) + 1)

    # Define custom colors limited to the number of unique classes
    custom_colors = [
        (0.5, 0.5, 0.5),  # grey
        (0.9, 0.8, 0.1),  # yellow
        (0.0, 0.6, 0.5),  # teal
        (0.6, 0.4, 0.8),  # purple
        (0.7, 0.85, 0.9),  # pale cyan
        (0.65, 0.2, 0.2),  # burgundy, avoid if too close to red
        (0.8, 0.7, 0.15),  # mustard
    ][:max_classes]  # limit the colors to the max number of classes
    cmap = ListedColormap(custom_colors)
    norm = BoundaryNorm(boundaries=np.arange(-0.5, max_classes + 0.5, 1), ncolors=max_classes)

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=1, ncols=len(datasets), figsize=(18, 6), dpi=dpi, subplot_kw={'projection': ccrs.PlateCarree()})

    if len(datasets) == 1:  # If there's only one dataset, axs will not be an array
        axs = [axs]

    centroids = {}
    for ax, (key, ds) in zip(axs, datasets.items()):
        # Select the 'class_label' data for plotting
        class_label = ds['class_label']
        unique_labels = np.unique(class_label.values)

        lon_adjusted = (ds['lon'].values + 180) % 360 - 180
        lon, lat = np.meshgrid(lon_adjusted, ds['lat'].values)
        
        # # Plot the data using pcolormesh, suitable for lat/lon grids
        # pcm = ax.pcolormesh(ds['lon'], ds['lat'], class_label, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        
        # # Add coastlines for better geographical context
        ax.coastlines()
        
        # Plot the labels
        # im = class_label.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, shading='auto', add_colorbar=False, alpha=1)
        pcm = ax.pcolormesh(ds['lon'], ds['lat'], class_label, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        ax.set_global()
        # ax.set_title(title_text)
        ax.set_title(f'{key.upper()}')

        # Compute centroids and plot them
        
        # for i, label in enumerate(unique_labels):
        #     # print(i)
        #     mask = class_label.values == label
        #     lon_mean = np.median(lon[mask])
        #     lat_mean = np.median(lat[mask])
        #     centroids[f'{key} {i}'] = (lon_mean, lat_mean)
            
        #     ax.plot(lon_mean, lat_mean, 'ko', markersize=15, markeredgecolor='white')
        #     # Add label near each centroid
        #     ax.text(lon_mean, lat_mean, f'{label}', color='white', ha='center', va='center', fontsize=12)


        for label in unique_labels:
            mask = class_label == label
            lon_mean = np.median(lon[mask])
            lat_mean = np.median(lat[mask])
            centroids[f'{key} {label}'] = (lon_mean, lat_mean, ax)
            ax.plot(lon_mean, lat_mean, 'ko', markersize=15, markeredgecolor='white')
            # Add label near each centroid
            ax.text(lon_mean, lat_mean, f'{label}', color='white', ha='center', va='center', fontsize=12)

    val_norm = Normalize(vmin=-1, vmax=1)
    cmap_arrow = cm.RdBu_r
    for target, links in causal_links.items():
        target_ax = centroids[target][2]
        target_pos = centroids[target][:2]
        for link in links:
            cause_key = f"{link['cause']}"
            if cause_key in centroids:
                cause_pos = centroids[cause_key][:2]
                cause_ax = centroids[cause_key][2]
                # Creating a new arrow with higher zorder and clip_on set to False
                arrow = ConnectionPatch(xyA=cause_pos, xyB=target_pos, coordsA='data', coordsB='data',
                                        axesA=cause_ax, axesB=target_ax,
                                        arrowstyle="-|>", color=cmap_arrow(val_norm(link['val'])), lw=3, shrinkA=5, shrinkB=5,
                                        connectionstyle="arc3,rad=0.3", zorder=100, clip_on=False)  # High zorder and clipping off
                # Adding the arrow directly to the figure to bypass subplot clipping
                fig.add_artist(arrow)

    # Create an axis for the colorbar on the right side of the figure
    cbar_ax = fig.add_axes([0.92, 0.29, 0.01, 0.4])  # Adjust the position [left, bottom, width, height] as necessary
    cbar = plt.colorbar(pcm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Class Label')
    cbar.set_ticks(np.arange(0, max_classes))
    cbar.set_ticklabels(np.arange(0, max_classes))

    # Add a color bar for the arrows
    sm = cm.ScalarMappable(cmap=cmap_arrow, norm=val_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.33, 0.1, 0.36, 0.03])  # Position for bottom colorbar
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Causal Strength')

    
    # place holder to make the figure higher
    cbar_T = fig.add_axes([0.33, 1, 0.36, 0.03])  # Position for bottom colorbar
    # turn of ticklabels and set spine in visiable
    cbar_T.axis('off')
    cbar_T.spines['top'].set_visible(True)
    cbar_T.spines['right'].set_visible(True)
    cbar_T.spines['bottom'].set_visible(True)
    cbar_T.spines['left'].set_visible(True)



    # plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the rect parameter to make room for the colorbar

    plt.show()


def filter_single_class_causal_links(causal_links, class_name):
    # Create a new dictionary to store the filtered causal links
    filtered_links = {}

    # Loop through each class in the original causal links
    for key, links in causal_links.items():
        # If the key is the class name, add all its links
        if key == class_name:
            filtered_links[key] = links
        else:
            # Otherwise, filter links to include only those where the cause is the class_name
            new_links = [link for link in links if link['cause'] == class_name]
            if new_links:
                filtered_links[key] = new_links

    return filtered_links