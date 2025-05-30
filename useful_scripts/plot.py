import pandas as pd
import matplotlib.pyplot as plt
import argparse
import matplotlib.ticker as ticker
import re
import numpy as np
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
            continue
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
        print(col)
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
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)
    # Get the column names
    
    # Extract the x and y values
    x_values, dumb, dumb, dumb = get_columns( data,  col_x)
    x_values = x_values[0] 
    if col_x =='0':
        x_values = np.array(x_values)
    if col_y == 'find_which_contain_numbers':
        col_y =  find_MAE_per_system(data.columns)
    print(col_y)
    y_values, labels, nums, cols = get_columns( data, col_y) 
    # Plot the dat
    size = 3.3
    plt.figure(figsize=(size, size), dpi = 300)
    plt.minorticks_on()
    plt.tick_params(direction='in', which='minor',length=size)
    plt.tick_params(direction='in', which='major',length=2*size)
    labeled = False
    
    for j, (y_v,lab, n, col) in enumerate (zip(y_values,labels,nums, cols) ):
        if 'MAE' in col :
            units = 'meV'
            if not labeled:
                ylabel+=' MAE ' + units
            mult = 0.0433641153*1e3
        else:
            units = r'meV$^2$'
            if not labeled:
                ylabel+=' MSE ' + units
            mult = (0.0433641153*1e3)**2
        
        if n ==0:
            lab = ' '
            n=1
            y = np.arry(y_v)*mult
        else:
            y = np.array(y_v)*mult/n
            if not labeled:
                ylabel+='/atom'
        labeled= True
        plt.plot(x_values, y, marker='o', linestyle='-', label = lab)
        print( 'System {:s} --> Last Point Error = {:4.3f} meV/atom --> last 5 points average = {:4.3f} meV/atom'.format(lab,y[-1],y[-5:].mean() ) )
    if title !='':
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.gca()  # Get the current axes

    # Set minor ticks at every even number
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
    plt.yscale('log')
    plt.grid(True)
    plt.xticks([int(x) for x in x_values if (x-1)%2==0])
    plt.legend(frameon=False, fontsize=2*size)
    plt.savefig(f'PREDICTION.png', bbox_inches='tight')

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

