import pandas as pd
import matplotlib.pyplot as plt


def visualise_metric(df: pd.DataFrame, metric: str, log_scale: bool = False, x_range: tuple[float, float] = None):
    """
    Visualize the distribution of a metric by account type.
    
    Args:
        df: DataFrame containing the data
        metric: Name of the metric column to visualize
        log_scale: If True, uses logarithmic scale for y-axis (default: False)
        x_range: Optional tuple of (min, max) for x-axis range. If None, uses data range.
    """
    if metric not in df.columns:
        raise ValueError(f"Metric {metric} not found in the DataFrame")

    # Separate corporate and private transactions, removing any NaN values
    corporate_df = df[df['account_type'] == 'corporate'][metric].dropna()
    private_df = df[df['account_type'] == 'private'][metric].dropna()

    # Set x-axis range
    if x_range is not None:
        x_min, x_max = x_range
    else:
        # Calculate min and max values from data
        min_val = min(corporate_df.min(), private_df.min())
        max_val = max(corporate_df.max(), private_df.max())
        
        # Add a small buffer to the range
        val_range = max_val - min_val
        x_min = min_val - 0.05 * val_range
        x_max = max_val + 0.05 * val_range

    # Plot the metrics with normalized density and outlined bars
    plt.figure(figsize=(10, 6))
    plt.hist(corporate_df, density=True, histtype='step', linewidth=2, 
             label='Corporate', bins=20, range=(x_min, x_max), color='blue')
    plt.hist(private_df, density=True, histtype='step', linewidth=2, 
             label='Private', bins=20, range=(x_min, x_max), color='orange')
    
    if log_scale:
        plt.yscale('log')
        plt.ylabel('Density (log scale)')
    else:
        plt.ylabel('Density')
        

    plt.legend(loc='upper right')
    plt.title(f'{metric} Distribution by Account Type')
    plt.xlabel(metric)
    plt.xlim(x_min, x_max)
    plt.grid()
    plt.show()