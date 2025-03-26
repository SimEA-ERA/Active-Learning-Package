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

def find_tot_err(columns,err='MAE', s='train'):
    return  [ c for c in columns if err in c and s in c and  'energy' in c and not contains_number(c) and 'norm' not in c] 
def find_err_per_system(columns,err='MAE'):
    return  [ c for c in columns if err in c and  'energy' in c and  contains_number(c) and 'norm' not in c] 

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

def get_csv_data(file_path,err='MAE',per_system=False ):
    data = pd.read_csv(file_path)
    if file_path =='COSTS.csv':
        datasets = ['train','dev']
    elif file_path == 'predictCOSTS.csv':
        datasets = ['all']

    cols = ['AL_iteration']
    for s in datasets:
        if not per_system:
            cols.extend(find_tot_err(data.columns, err , s) )
        else:
            cols.extend(find_err_per_system(data.columns,err) )
    data = data[cols]
    for c in data:
        print(file_path, c)
    return data

def plot_learning_rate(err='MAE'):
    train_dev = get_csv_data('COSTS.csv',err)
    pred = get_csv_data('predictCOSTS.csv',err)
    print(len(train_dev), len(pred))
    data = pd.DataFrame()
    
    mult = 0.0433641153*1e3
    data['AL_iteration' ] = train_dev['AL_iteration'] + 1
    data['train'] = train_dev[err+'_train_energy'].to_numpy()*mult 
    data['dev'] = train_dev[err+'_dev_energy'].to_numpy()*mult 
    data['pred'] =  pred[err+'_all_energy'].to_numpy()*mult
   

    size = 3.3
    plt.figure(figsize=(size, size), dpi = 300)
    plt.minorticks_on()
    plt.tick_params(direction='in', which='minor',length=size)
    plt.tick_params(direction='in', which='major',length=2*size)
    plt.xlabel('AL iteration',fontsize=2.5*size)
    plt.ylabel(f'{err} (meV)',fontsize=2.5*size)
    ax = plt.gca()  # Get the current axes
    x_values = data['AL_iteration'].to_numpy()
    colors = ['#1b9e77','#d95f02','#7570b3']
    markers = ['s','o','*']
    lst = ['-','--',':']
    for j,col in enumerate(['train','dev','pred']):
        y = data[col].to_numpy()
        plt.plot(x_values, y, label=col, color = colors[j], ls = lst[j],marker=markers[j],
                 markersize=1.05*size, lw=0.3*size)
        print(f'{col} --> {y[-1]} , last 5 = {y[-5:].mean()}')
    # Set minor ticks at every even number
    odd_numbers = np.arange(1, x_values[-1]+1, 2)  # Adjust range as needed
    ax.xaxis.set_minor_locator(ticker.FixedLocator(odd_numbers))

    plt.yscale('log')
    plt.grid(True,alpha=0.35)
    e = 2 if len(x_values) < 30 else 4
    if len(x_values)>60: e = 6
    plt.xticks([int(x) for x in x_values if (x)%e == 0])
    plt.legend(frameon=False, fontsize=2*size)
    plt.savefig(f'{err}_learning_curve.png', bbox_inches='tight')
    return

def label_nicely(s):
    s_nop_= s.split('(')[0]
    element_pattern = re.findall(r'([A-Z][a-z]*)(\d*)|\(([^)]+)\)(\d*)', s)
    sorting = ['Ag','C',1,'N','S','O']
    eln = [ (ele[0],ele[1]) for ele in element_pattern if len(ele[0]) !=0 ]
    l = []
    for e in sorting:
        for el in eln:
            if el[0] == e:
                l.append(e)
                if len(el[1]) >0:
                    l.append(r'$_{'+ el[1] + '}$')
    new_s = ''.join(l)
    print(new_s)
    return new_s

                
def plot_pred_per_system(err='MAE'):
    pred = get_csv_data('predictCOSTS.csv',err,per_system=True)
    data = pd.DataFrame()
    
    mult = 0.0433641153*1e3
    data['AL_iteration' ] = pred['AL_iteration']
    systems = [c.split('_')[1] for c in pred.columns if 'AL_iteration' != c]
    print(systems)
    for s in systems: 
        data[s] =  pred[err+f'_{s}_energy'].to_numpy()*mult/identify_natoms(s)
   

    size = 3.3
    plt.figure(figsize=(size, size), dpi = 300)
    plt.minorticks_on()
    plt.tick_params(direction='in', which='minor',length=size)
    plt.tick_params(direction='in', which='major',length=2*size)
    plt.xlabel('AL iteration',fontsize=2.5*size)
    plt.ylabel(f'Prediction {err} (meV/atom)',fontsize=2.5*size)
    ax = plt.gca()  # Get the current axes
    x_values = data['AL_iteration'].to_numpy()
    colors = ['#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac']
    markers = ['s','o','*','v']
    lst = ['-','--',':']
    for j,col in enumerate(systems):
        plt.plot(x_values, data[col], label = label_nicely(col), 
                color = colors[j%len(colors)], ls = lst[j%len(lst)],
                marker=markers[j%len(markers)], markersize=1.05*size,lw=0.3*size )
        y = data[col].to_numpy()
        print(f'{col} --> {y[-1]} , last 5 = {y[-5:].mean()}')
    # Set minor ticks at every even number
    odd_numbers = np.arange(1, x_values[-1]+1, 2)  # Adjust range as needed
    ax.xaxis.set_minor_locator(ticker.FixedLocator(odd_numbers))
    plt.yscale('log')
    plt.grid(True,alpha=0.35)
    e = 2 if len(x_values) < 30 else 4
    if len(x_values)>60: e = 6
    plt.xticks([int(x) for x in x_values if (x)%e==0])
    plt.legend(frameon=False, fontsize=2*size,ncol = int(j/5)+1)
    plt.savefig(f'Prediction_{err}_per_system.png', bbox_inches='tight')
def main():
    # Set up command-line argument parsing
    err = 'MAE'

    plot_learning_rate(err)
    plot_pred_per_system(err)
if __name__ == "__main__":
    main()

