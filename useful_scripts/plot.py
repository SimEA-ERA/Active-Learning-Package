import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_csv_columns(file_path, col_x, col_y):
    """
    Reads a CSV file and plots two specified columns as an x-y plot.

    Parameters:
        file_path (str): Path to the CSV file.
        col_x (int): Index of the column to use for x-axis (0-based).
        col_y (int): Index of the column to use for y-axis (0-based).
    """
    try:
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(file_path)
        if 'pred' in file_path:
            xi = 'Prediction'
        else:
            xi = 'training'
        if 'MAE' in data.columns[col_y]:
            units = 'meV'
            mult = 0.0433641153*1e3
        else:
            units = r'meV$^2$'
            mult = (0.0433641153*1e3)**2
        # Get the column names
        y_label = ' '.join([ xi, data.columns[col_y].split('_')[0], '('+ units+')'] )
        x_label = data.columns[col_x].replace('_', ' ')
        
        # Extract the x and y values
        x_values = data.iloc[:, col_x]
        
        y_values = data.iloc[:, col_y]*mult
        
        # Plot the data
        size = 3.3
        plt.figure(figsize=(size, size))
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size)
        plt.tick_params(direction='in', which='major',length=2*size)
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
        plt.title(f"{y_label} vs {x_label}")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(f'{y_label}-vs-{x_label}', bbox_inches='tight')
        #plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Plot two columns from a CSV file as an x-y plot.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file")
    parser.add_argument("col_x", type=int, help="Index of the column to use for x-axis (0-based)")
    parser.add_argument("col_y", type=int, help="Index of the column to use for y-axis (0-based)")
    
    args = parser.parse_args()
    
    # Call the plot function with arguments
    plot_csv_columns(args.file_path, args.col_x, args.col_y)

if __name__ == "__main__":
    main()

