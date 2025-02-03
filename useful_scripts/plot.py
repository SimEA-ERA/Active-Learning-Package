import pandas as pd
import matplotlib.pyplot as plt
import argparse

import re
from collections import defaultdict

def identify_name(c):
    t = c.split('_')[1]
    return t

def contains_number(string):
    return any(char.isdigit() for char in string)

def find_MAE_per_system(columns):
    return ' '.join( [ c for c in columns if 'MAE' in c and  'energy' in c and contains_number(c) and 'norm' not in c] )
def identify_natoms(formula: str):
    element_pattern = re.findall(r'([A-Z][a-z]*)(\d*)|\(([^)]+)\)(\d*)', formula)
    atom_count = defaultdict(int)
    
    for match in element_pattern:
        if match[0]:  # Element case
            element = match[0]
            count = int(match[1]) if match[1] else 1
            atom_count[element] += count
        elif match[2]:  # Parenthesis case
            inner_formula = match[2]
            multiplier = int(match[3]) if match[3] else 1
            inner_atoms = identify_natoms(inner_formula)
            for elem, num in inner_atoms.items():
                atom_count[elem] += num * multiplier
    
    return sum( list(atom_count.values()) )



def get_columns(data, col):
    tc = col.split(' ')

    values, names, nums, cols = [], [] , [], []
    for col in tc:
        c = col if col  in data.columns else data.columns [ int(col) ]
        values.append( data.loc[:,c])
        name = identify_name(c)
        names.append(name)
        nums.append( identify_natoms(name) )
        cols.append(c)
    return values, names, nums, cols

def plot_csv_columns(file_path, col_x, col_y, xlabel, ylabel,title):
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
        # Get the column names
        
        # Extract the x and y values
        x_values, dumb, dumb, dumb = get_columns( data,  col_x)
        x_values = x_values[0] 
        
        if col_y == 'find_which_contain_numbers':
            col_y =  find_MAE_per_system(data.columns)
        
        y_values, labels, nums, cols = get_columns( data, col_y) 
        print(  cols )
        # Plot the dat
        size = 3.3
        plt.figure(figsize=(size, size))
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size)
        plt.tick_params(direction='in', which='major',length=2*size)
        
        for j, (y_v,lab, n, col) in enumerate (zip(y_values,labels,nums, cols) ):
            if 'MAE' in col:
                units = 'meV'
                ylabel+=' MAE ' + units
                mult = 0.0433641153*1e3
            else:
                units = r'meV$^2$'
                ylabel+=' MSE ' + units
                mult = (0.0433641153*1e3)**2
            
            if n ==0:
                lab = ' '
                n=1
                y = y_v*mult
            else:
                y = y_v*mult/n
                ylabel+='/atom'

            plt.plot(x_values, y, marker='o', linestyle='-', label = lab)
        
        if title !='':
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale('log')
        plt.grid(True)
        plt.legend(frameon=False, fontsize=2*size)
        plt.savefig(f'PREDICTION.png', bbox_inches='tight')
        #plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Plot two columns from a CSV file as an x-y plot.")
    parser.add_argument('-f',"--file_path", type=str, help="Path to the CSV file")
    parser.add_argument('-x',"--col_x", default='0', type=str, help="Index or name of the column to use for x-axis ")
    parser.add_argument('-y',"--col_y",default='find_which_contain_numbers', type=str, help="Index(s) or name(s) of the column(s) to use for y-axis")
    parser.add_argument('-xlabel', "--xlabel", type=str, help="xlabel",default='AL iteration')
    parser.add_argument('-ylabel', "--ylabel", type=str, help="ylabel", default= 'Prediction')
    parser.add_argument('-t', "--title", type=str, help="ylabel", default= '')
    
    args = parser.parse_args()
    
    # Call the plot function with arguments
    plot_csv_columns(args.file_path, args.col_x, args.col_y, args.xlabel, args.ylabel,args.title)

if __name__ == "__main__":
    main()

