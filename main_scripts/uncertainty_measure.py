import sys
from time import perf_counter
import os
import pandas as pd
import FF_Develop as ff
import numpy as np
import argparse
from matplotlib import pyplot as plt 
import matplotlib.ticker as ticker
import re

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

def main():
    
    t0 = perf_counter()
    
    argparser = argparse.ArgumentParser(description="Uncertainy post calculation, created by Nikolaos Patsalidis")
    
    hnum ='Total number of iterations number'
    
    argparser.add_argument('-n',"--maxnum",metavar=None,
            type=int, required=True, help=hnum)
    argparser.add_argument('-in',"--inputfile",metavar=None,
            type=str, required=True, help=hnum)
    
    argparser.add_argument('-bs',"--beta_sampling",metavar=None,
            type=float, required=True, help=hnum)
    args = argparser.parse_args()
    
    setup = ff.Setup_Interfacial_Optimization(args.inputfile)

    data_dict = dict()
    
    for n in range(args.maxnum+1):    
        print('Reading iteration {:d} '.format(n))
        sys.stdout.flush()
        path = 'data'
        path_xyz ='{:s}/D{:d}'.format(path,n)
        df = ff.al_help.data_from_directory(path_xyz)
        
        
        ff.al_help.make_absolute_Energy_to_interaction(df,setup)
        
        df = ff.al_help.clean_data(df,setup, args.beta_sampling)
        
        ff.al_help.make_interactions(df,setup)
        data_dict[n] = df
    
    ud = dict()
    setup.opt_models = setup.init_models

    for n in range(args.maxnum):
        print(f'Iteration {n}')
        candidate_data = pd.DataFrame()
        existing_data = pd.DataFrame()
        #for n1 in range(args.maxnum-1,n,-1):
         #   print(n1)
         #   candidate_data = candidate_data.append(data_dict[n1], ignore_index=True)
        candidate_data = data_dict[n+1]
        for n2 in range(0,n+1):
            print(n2)
            existing_data = existing_data.append(data_dict[n2], ignore_index=True)
        
        uns = np.unique(candidate_data['sys_name'])
        print(uns)
        sd = dict()
        for s in uns:
            fc = candidate_data['sys_name'] == s
            fex = existing_data['sys_name'] == s
            uncertainty_norm, uncertainty = ff.al_help.find_histogram_uncertainty( candidate_data [fc], existing_data[fex]  , setup )
            sd[s] = {'mean':uncertainty.mean(),'std': uncertainty.std(), 'min': uncertainty.min(), 'max':uncertainty.max()}
        ud[n] = sd

    data = {s:{k:[]  for k in ['mean', 'std', 'min', 'max']} for s in uns  }
    for s in uns:
        iters = []
        for u,d in ud.items():
            iters.append(u)
            try:
                x = d[s]
                print(u,s,d[s])
            except KeyError:
                d[s] = { k: np.nan for k in ['mean','std','min','max'] }
            for k in data[s]:
                data[s][k].append(d[s][k])
    iters = np.array(iters) +1
    size = 3.3
    _ = plt.figure(figsize=(size, size),dpi=300)
    plt.minorticks_on()
    plt.tick_params(direction='in', which='minor',length = size)
    plt.tick_params(direction='in', which='major',length = 2*size)
    plt.xlabel('AL iteration', fontsize=2.5*size)
    plt.ylabel('u', fontsize=2.5*size)
    ax = plt.gca()
    colors = ['#1b9e77','#d95f02','#7570b3']
    if len(data)>3:
        colors = ['#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac']*10
    markers = ['s','o','*','v']*10
    ls = ['-','--','-.',':']*10
    e =2 if args.maxnum < 30 else 4 
    for i,s in enumerate(data):
        label = label_nicely(s)
        c = colors[i]
        m = markers[i]
        lst = ls[0]
        mm = np.array( [ data[s]['min'], data[s]['max'] ] ).reshape((2, len(data[s]['max'])) )
        d = np.array(data[s]['mean'])
        
        #f = np.logical_not(np.isnan(d))
        plt.errorbar(iters,d,yerr=mm, color=c,elinewidth = 0.2*size,capthick=0.2*size,
                 capsize=0.3*size,  lw=0.,markersize=0)
        plt.errorbar(iters,d,yerr=data[s]['std'], capthick=0.6*size, elinewidth = 0.4*size, capsize=0.6*size,
                label= label, color = c, marker = m, ls = lst , lw=0.4*size,markersize=1.4*size)
    
    
    odd_numbers = np.arange(1, iters[-1]+1, 2)  # Adjust range as needed
    ax.xaxis.set_minor_locator(ticker.FixedLocator(odd_numbers))
    plt.yscale('log')
    plt.grid(True,alpha=0.35)
    e= 6 if len(iters) >60 else 4
    if len(iters)<30: e = 2
    plt.xticks([int(x) for x in iters if (x)%e == 0])
    plt.legend(frameon=False, fontsize= 2*size, ncol=int(i/5) +1)
    plt.savefig('unc-log.png', bbox_inches='tight')
    plt.close()

    _ = plt.figure(figsize=(size, size),dpi=300)
    plt.minorticks_on()
    plt.tick_params(direction='in', which='minor',length = size)
    plt.tick_params(direction='in', which='major',length = 2*size)
    ax = plt.gca()
    
    for i,s in enumerate(data):
        label = label_nicely(s)
        c = colors[i]
        m = markers[i]
        lst = ls[0]
        mm = np.array( [ data[s]['min'], data[s]['max'] ] ).reshape((2, len(data[s]['max'])) )
        d = data[s]['mean']
        plt.errorbar(iters,d,yerr=mm, color=c,elinewidth = 0.2*size,capthick=0.2*size,
                 capsize=0.3*size,  lw=0.,markersize=0)
        plt.errorbar(iters,d,yerr=data[s]['std'], capthick=0.6*size, elinewidth = 0.4*size, capsize=0.6*size,
                label= label, color = c, marker = m, ls = lst , lw=0.4*size,markersize=1.4*size)
    odd_numbers = np.arange(1, iters[-1]+1, 2)  # Adjust range as needed
    ax.xaxis.set_minor_locator(ticker.FixedLocator(odd_numbers))
    plt.ylim((0,data[ list(data.keys())[0] ]['max'][-1]*3) ) 
    plt.grid(True,alpha=0.35)
    e= 6 if len(iters) >60 else 4
    if len(iters)<30: e = 2
    plt.xticks([int(x) for x in iters if (x)%e == 0])
    plt.savefig('unc-linear.png', bbox_inches='tight')
    plt.close()


if __name__=='__main__':
    main()
    
