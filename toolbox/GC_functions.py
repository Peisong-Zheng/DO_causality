import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests,kpss
import matplotlib.pyplot as plt

def is_stationary(series, test_method='ADF', print_results=True):
    """
    Check if a time series is stationary.

    Parameters:
    - series: NumPy ndarray, the time series data to be tested for stationarity.
    - test_method: string, the method used to test for stationarity. Supports 'ADF' for Augmented Dickey-Fuller test
                   and 'KPSS' for Kwiatkowski-Phillips-Schmidt-Shin test.

    Returns:
    - is_stationary: boolean, True if the series is stationary, False otherwise.
    """
    is_stationary = False

    if test_method == 'ADF':
        # Perform Augmented Dickey-Fuller test
        result = adfuller(series)
        if print_results:
            print(f'ADF Statistic: {result[0]}')
            print(f'p-value: {result[1]}')
            print('Critical Values:')
            for key, value in result[4].items():
                print(f'\t{key}: {value}')
        is_stationary = result[1] <= 0.05

    elif test_method == 'KPSS':
        # Perform KPSS test
        statistic, p_value, _, critical_values = kpss(series, 'c')
        if print_results:
            print(f'KPSS Statistic: {statistic}')
            print(f'p-value: {p_value}')
            print('Critical Values:')
            for key, value in critical_values.items():
                print(f'\t{key}: {value}')
        # In KPSS test, the null hypothesis is that the series is stationary
        is_stationary = p_value > 0.05

    # Print result
    if print_results:
        if is_stationary:
            print("The series is stationary.")
        else:
            print("The series is not stationary.")

    return is_stationary



def make_stationary(df, column_names, test_method='ADF',plot=True, print_results=False):
    """
    Process the specified columns of a DataFrame to ensure stationarity.
    If a series is not stationary, it is differenced up to a maximum of 5 times.

    Parameters:
    - df: pandas DataFrame, the DataFrame containing the time series data.
    - column_names: list of strings, the names of the columns to be processed for stationarity.
    - test_method: string, the method used to test for stationarity. Supports 'ADF' and 'KPSS'.

    Returns:
    - The DataFrame with the processed columns where necessary.
    """
    modified_df = df.copy()  # Copy the DataFrame to avoid modifying the original

    for column in column_names:
        for i in range(1, 6):  # Maximum of 5 differencing operations
            # Ensure the series does not have NaN values
            series = modified_df[column].dropna()
            if is_stationary(series.values, test_method, print_results):
                if print_results:
                    print(f"Series in column '{column}' is stationary after {i-1} differencing operation(s).")
                break
            else:
                if i < 5:  # Only difference if we haven't reached the max differencing order
                    modified_df[column] = modified_df[column].diff().dropna()  # Drop NaN after differencing
                    if plot:
                        plt.figure(figsize=(10, 6))
                        plt.plot(df[column], label='Original')
                        plt.plot(modified_df[column], label='Differenced', linestyle='--')
                        plt.legend()
                        plt.title(f"Original vs Differenced Data for {column} (Order {i})")
                        plt.show()
                else:
                    if print_results:
                        print(f"Series in column '{column}' is not stationary after 5 differencing operations.")
    # drop NaN values
    modified_df = modified_df.dropna()
    return modified_df




# from statsmodels.tsa.stattools import grangercausalitytests
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor, as_completed

# def gc_test_single_pair(args):
#     """
#     Perform Granger Causality test for a single pair of time series.
    
#     Parameters:
#     - args: tuple containing data_x_col, data_y_col, and max_lags
    
#     Returns:
#     - Tuple of two booleans: (result_xy, result_yx) indicating the direction of Granger causality for the pair.
#     """
#     data_x_col, data_y_col, max_lags = args
    
#     # Drop NaN values from both time series
#     valid_idx = ~np.isnan(data_x_col) & ~np.isnan(data_y_col)
#     data_x_col_clean = data_x_col[valid_idx]
#     data_y_col_clean = data_y_col[valid_idx]

#     # Combine the cleaned and cropped data for Granger Causality test
#     combined_data = np.column_stack((data_x_col_clean, data_y_col_clean))
    
#     # Granger Causality test
#     result_xy = grangercausalitytests(combined_data, max_lags, verbose=False)
#     result_yx = grangercausalitytests(combined_data[:, [1, 0]], max_lags, verbose=False)
    
#     # Simplify result interpretation
#     test_result_xy = any(result_xy[lag][0]['ssr_ftest'][1] < 0.05 for lag in range(1, max_lags + 1))
#     test_result_yx = any(result_yx[lag][0]['ssr_ftest'][1] < 0.05 for lag in range(1, max_lags + 1))
    
#     return test_result_xy, test_result_yx

# def gc_test_parallel(data_x, data_y, max_lags=4, num_workers=None):
#     """
#     Perform Granger Causality tests in parallel for each pair of columns in data_x and data_y.
    
#     Parameters:
#     - data_x, data_y: 2D NumPy ndarrays containing the time series data.
#     - max_lags: int, maximum number of lags to test for.
#     - num_workers: int or None, number of worker threads to use. If None, it will use the default.
    
#     Returns:
#     - List of tuples with the Granger Causality test results for each pair.
#     """
#     num_series = data_x.shape[1]
#     with ThreadPoolExecutor(max_workers=num_workers) as executor:
#         futures = [executor.submit(gc_test_single_pair, (data_x[:, i], data_y[:, i], max_lags)) for i in range(num_series)]
        
#         results = []
#         for future in as_completed(futures):
#             result = future.result()
#             results.append(result)
#     return results






def gc_test(data, column_x, column_y, max_lags=4, print_results=True):
    """
    Perform Granger Causality test in both directions and format the result.
    
    Parameters:
    - data: pandas DataFrame containing the time series data.
    - column_x, column_y: names of the columns to be tested.
    - max_lags: int, maximum number of lags to test for.
    
    Returns:
    - Formatted result as a string.
    """
    result_yx = grangercausalitytests(data[[column_x, column_y]], max_lags, verbose=False)
    result_xy = grangercausalitytests(data[[column_y, column_x]], max_lags, verbose=False)
    
    # Initialize variables to store the best p-value and corresponding F statistic and lag
    best_p_xy, best_f_xy, best_lag_xy = 1, None, None
    best_p_yx, best_f_yx, best_lag_yx = 1, None, None
    
    for lag in range(1, max_lags + 1):
        # XY direction
        f_test_xy = result_xy[lag][0]['ssr_ftest']
        if f_test_xy[1] < best_p_xy:
            best_p_xy, best_f_xy, best_lag_xy = f_test_xy[1], f_test_xy[0], lag
        # YX direction
        f_test_yx = result_yx[lag][0]['ssr_ftest']
        if f_test_yx[1] < best_p_yx:
            best_p_yx, best_f_yx, best_lag_yx = f_test_yx[1], f_test_yx[0], lag
    
    # Determine if tests are passed
    test_passed_xy = "True" if best_p_xy < 0.05 else "False"
    test_passed_yx = "True" if best_p_yx < 0.05 else "False"
    
    # Format the result
    result_str = "Granger Causality Test\n------------------------------------------------------------\n"
    result_str += "Direction                   F-statistics         p-value         lag         Granger cause\n"
    result_str += f"{column_x} => {column_y}    {best_f_xy:.3f}               {best_p_xy:.3f}                {best_lag_xy}                {test_passed_xy}\n"
    result_str += f"{column_y} => {column_x}    {best_f_yx:.3f}               {best_p_yx:.3f}                {best_lag_yx}                {test_passed_yx}\n"
    if print_results:
        print(result_str)
    
    return result_str

def gc4vars(df, max_lags=10, print_results=True):

    
    gc_results = []  # List to store the results
    columns = df.columns
    
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            column_x = columns[i]
            column_y = columns[j]
            if print_results:
                print(f"Testing causality between: {column_x} and {column_y}")
            
            # Call the gc_test function with specified max_lags
            result_str = gc_test(df, column_x, column_y, max_lags=max_lags, print_results=print_results)
            
            # Parse the result_str to extract causality results
            lines = result_str.split('\n')
            cause_yx = lines[-3].split()[-1]  # Extracts the "True" or "False" for X => Y
            cause_xy = lines[-2].split()[-1]  # Extracts the "True" or "False" for Y => X
            
            # Store the results
            gc_results.append({
                'variables': (column_x, column_y),
                'XY': cause_xy == 'True',
                'YX': cause_yx == 'True'
            })
    
    return gc_results


# import numpy as np
# from statsmodels.tsa.stattools import grangercausalitytests
# from concurrent.futures import ThreadPoolExecutor, as_completed

# def gc_test_single_pair(data_x_col, data_y, max_lags=4):
#     combined_data = np.column_stack((data_x_col, data_y))
#     result_xy = grangercausalitytests(combined_data, max_lags, verbose=False)
#     test_result_xy = any(result_xy[lag][0]['ssr_ftest'][1] < 0.05 for lag in range(1, max_lags + 1))
#     return test_result_xy

# def gc_test_parallel(data_x, data_y, max_lags=4):
#     n_series = data_x.shape[1]
#     results_xy = np.zeros(n_series, dtype=bool)
    
#     with ThreadPoolExecutor() as executor:
#         future_to_series = {executor.submit(gc_test_single_pair, data_x[:, i], data_y, max_lags): i for i in range(n_series)}
#         for future in as_completed(future_to_series):
#             series_index = future_to_series[future]
#             results_xy[series_index] = future.result()
    
#     return results_xy


# import numpy as np
# from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests

# def is_stationary_fast(ndarray, test_method='ADF'):
#     """
#     Test if a time series is stationary, optimized for speed.
    
#     Parameters:
#     - ndarray: NumPy ndarray, the time series data.
#     - test_method: string, the method used to test for stationarity ('ADF' or 'KPSS').
    
#     Returns:
#     - bool: True if the series is stationary, False otherwise.
#     """
#     if test_method == 'ADF':
#         test_result = adfuller(ndarray, autolag='AIC')
#         return test_result[1] <= 0.05  # p-value
#     elif test_method == 'KPSS':
#         test_result = kpss(ndarray, regression='c', nlags="auto")
#         return test_result[1] > 0.05  # p-value
#     else:
#         raise ValueError("Unsupported test method. Use 'ADF' or 'KPSS'.")

# # def make_stationary_fast(ndarray, test_method='ADF'):
# #     """
# #     Makes a time series stationary if it's not already, optimized for speed.
    
# #     Parameters:
# #     - ndarray: NumPy ndarray, the time series data.
# #     - test_method: string, the method used to test for stationarity.
    
# #     Returns:
# #     - ndarray: The (possibly differenced) stationary time series.
# #     """
# #     for _ in range(5):  # Attempt up to 5 differencing operations
# #         if is_stationary_fast(ndarray, test_method):
# #             break
# #         ndarray = np.diff(ndarray, n=1)  # Difference the series
# #     return ndarray
    

# def make_stationary_fast(series_a, series_b, test_method='ADF'):
#     """
#     Makes two time series stationary, ensuring they are differenced the same number of times.

#     Parameters:
#     - series_a, series_b: NumPy ndarrays, the time series data.
#     - test_method: string, the method used to test for stationarity.

#     Returns:
#     - Tuple of NumPy ndarrays: The (possibly differenced) stationary time series.
#     """
#     max_diffs = 0
#     for series in [series_a, series_b]:
#         for diffs in range(1, 6):  # Attempt up to 5 differencing operations
#             if is_stationary_fast(series, test_method):
#                 max_diffs = max(max_diffs, diffs-1)
#                 break
#             series = np.diff(series, n=1)  # Difference the series

#     # Apply the maximum number of differences needed to both series
#     if max_diffs > 0:
#         series_a = np.diff(series_a, n=max_diffs)
#         series_b = np.diff(series_b, n=max_diffs)

#     return series_a, series_b

# def gc_test_refactored(data_x, data_y, max_lags=4):
#     """
#     Perform Granger Causality test and return boolean results.
    
#     Parameters:
#     - data_x, data_y: NumPy ndarrays containing the time series data for X and Y.
#     - max_lags: int, maximum number of lags to test for.
    
#     Returns:
#     - Tuple of two booleans: (result_xy, result_yx) indicating the direction of Granger causality.
#     """
#     # Make data stationary
#     # data_x, data_y = make_stationary_fast(data_x,data_y,'ADF')

    
#     # Combine into a 2D array as required by grangercausalitytests
#     combined_data = np.column_stack((data_x, data_y))
    
#     # Granger Causality test
#     result_xy = grangercausalitytests(combined_data, max_lags, verbose=False)
#     result_yx = grangercausalitytests(combined_data[:, [1, 0]], max_lags, verbose=False)
    
#     # Simplify result interpretation
#     test_result_xy = any(result_xy[lag][0]['ssr_ftest'][1] < 0.05 for lag in range(1, max_lags + 1))
#     test_result_yx = any(result_yx[lag][0]['ssr_ftest'][1] < 0.05 for lag in range(1, max_lags + 1))
    
#     return test_result_xy, test_result_yx












from statsmodels.tsa.arima.model import ARIMA

def find_optimal_ar_order(df, show_figures=False):
    # Determine the maximum order for AR models based on the length of the data
    max_order = min(len(df) // 10, 30)  # Not to exceed 30 to avoid overfitting and computational burden
    
    results = {}  # To store the optimal order and AIC for each column
    
    for column in df.columns:
        best_aic = np.inf
        best_order = None
        for p in range(1, max_order + 1):
            try:
                model = ARIMA(df[column], order=(p, 0, 0))
                model_fit = model.fit()
                aic = model_fit.aic
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = p
            except Exception as e:
                print(f"Could not fit model for {column} with order={p}: {e}")
                break
        
        # Store the best order and AIC for the column
        results[column] = {'Optimal Order': best_order, 'AIC': best_aic}
        
        # If requested, show the comparison figure
        if show_figures and best_order is not None:
            model = ARIMA(df[column], order=(best_order, 0, 0))
            model_fit = model.fit()
            predictions = model_fit.predict()
            
            plt.figure(figsize=(10, 5))
            plt.plot(df[column], label=f'Original {column}', color='blue')
            plt.plot(predictions, label=f'Fitted AR({best_order}) Model', color='red', linestyle='--')
            plt.title(f'Comparison of Original Series and Fitted AR({best_order}) Model for {column}')
            plt.xlabel('Time')
            plt.ylabel(column)
            plt.legend()
            plt.show()
    
    return results






import networkx as nx
import matplotlib.pyplot as plt

def plot_causal_graph(gc_results, variables=None,dpi=600):
    # Create a directed graph
    G = nx.DiGraph()

    # Infer variables from gc_results if not provided
    if variables is None:
        variables = set()
        for result in gc_results:
            variables.update(result['variables'])
        variables = list(variables)

    # Add nodes (unique variables)
    for var in variables:
        G.add_node(var)

    # Storage for edges to draw them with specific styles later
    edges_uni = []
    edges_bi = []

    # Iterate over the stored results to categorize edges
    for result in gc_results:
        column_x, column_y = result['variables']
        if result['XY'] and result['YX']:  # If causality is bidirectional
            edges_bi.append((column_x, column_y))
            edges_bi.append((column_y, column_x))
        else:
            if result['XY']:  # If X causes Y
                edges_uni.append((column_x, column_y))
            if result['YX']:  # If Y causes X
                edges_uni.append((column_y, column_x))

    for column_x, column_y in edges_uni:
        # For unidirectional edges, add them to the graph with a default color (black) and strength
        G.add_edge(column_x, column_y, strength=1, color='black')

    for column_x, column_y in edges_bi:
        # For bidirectional edges, add them with a distinct color (red)
        G.add_edge(column_x, column_y, strength=1, color='red')

    # Choose layout
    pos = nx.circular_layout(G)

    plt.figure(figsize=(7, 7), dpi=dpi)  # Set the size of the figure

    connectionstyle = 'arc3,rad=0.1' 
    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=4200, edgecolors='black', node_color='grey', linewidths=2, alpha=1)

    # Draw the edges
    nx.draw_networkx_edges(G, pos, edgelist=edges_uni, edge_color='black', arrows=True, width=3, arrowsize=30,
                           min_source_margin=40, min_target_margin=40, connectionstyle=connectionstyle)
    nx.draw_networkx_edges(G, pos, edgelist=edges_bi, edge_color='violet', arrows=True, width=3, arrowsize=30,
                           min_source_margin=40, min_target_margin=40, connectionstyle=connectionstyle)

    # Labels for the nodes
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.axis('off')  # Turn off the axis
    plt.show()  # Display the graph


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def causal_matshow(causal_results):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=600)
    cmap = colors.ListedColormap(['wheat', 'mediumseagreen'])
    bounds = [-0.5, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    cax = ax.matshow(causal_results, interpolation='nearest', cmap=cmap, norm=norm)
    cbar = fig.colorbar(cax, ticks=[0, 1], shrink=0.8)

    lags = np.arange(causal_results.shape[1]) * 10  # Multiply by 10 for lags
    causal_strengths = np.linspace(0, 1, causal_results.shape[0])  # Linear space for causal strengths

    ax.set_xticks(np.arange(len(lags)))
    ax.set_xticklabels(lags)
    ax.set_yticks(np.arange(len(causal_strengths)))
    ax.set_yticklabels(np.round(causal_strengths, 1))
    ax.set_xlabel('Lags (Years)')
    ax.set_ylabel('Causal Strength')
    ax.xaxis.set_ticks_position('bottom')

    plt.show()