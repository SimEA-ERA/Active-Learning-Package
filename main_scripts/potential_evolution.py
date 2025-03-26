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
import math
def label_nicely(s,last=False):
    if s=='th0':
        return r'$\theta_0$'
    elif 'k' in s and len(s)>1:
        if last:
            l = r'$y_L$'
        else:
            l = r'$y_{'+str(int(s[1:])-1)+'}$'
        return l 
    elif s =='re':
        return r'$r_e$'
    elif s =='De':
        return r'$D_e$'
    elif s =='alpha':
        return r'$\alpha$'
    else:
        return f'${s}$'
def ylabel(key):
    ylabel = f'{key[-1]} parameters for {key[0]} interactions of type {key[1]}'
    return ylabel
def main():
    
    t0 = perf_counter()
    
    argparser = argparse.ArgumentParser(description="Uncertainy post calculation, created by Nikolaos Patsalidis")
    
    hnum ='Total number of iterations number'
    
    argparser.add_argument('-n',"--maxnum",metavar=None,
            type=int, required=True, help=hnum)
    
    args = argparser.parse_args()

    setup_dict = dict()
    
    for n in range(args.maxnum+1):    
        print('Reading iteration {:d} '.format(n))
        setup_dict[n] = ff.Setup_Interfacial_Optimization(f'Results/{n}/runned.in')
    models = setup_dict[0].init_models
    model_parameters = {(model.category,model.type,model.model):{ model.pinfo[k].name :[] for k in model.pinfo } for key,model  in models.items()  }
    
    for n in range(args.maxnum+1):
        for key,model in setup_dict[n].init_models.items():
            for k in model.pinfo:
                ty = (model.category,model.type,model.model)
                pn = model.pinfo[k].name
                v = model.pinfo[k].value
                print(ty,pn,v)
                model_parameters[ty][pn].append( v )
    print(model_parameters)
    iters = np.arange(1,args.maxnum+2,1,dtype=int)
    size = 3.3
    _ = plt.figure(figsize=(size, size),dpi=600)
    plt.minorticks_on()
    plt.tick_params(direction='in', which='minor',length = size)
    plt.tick_params(direction='in', which='major',length = 2*size)
    ax = plt.gca()

    colors = ['#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac']*10
    markers = ['s','o','*','v','^']*10
    ls = ['-','--','-.',':']*10
    naxis = len(model_parameters)
    cols = math.ceil(math.sqrt(naxis))
    rows = math.ceil(naxis / cols)

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # Flatten axes for easy iteration (in case of 2D array)
    axes = axes.flatten()

    odd_numbers = np.arange(1, iters[-1]+1, 2)  # Adjust range as needed
    e= 6 if len(iters) >60 else 4
    if len(iters)<30: e = 2
    xticks = [int(x) for x in iters if (x)%e == 0]
    for i,(key, params) in enumerate(model_parameters.items()):
        ax = axes[i]
        ax.xaxis.set_minor_locator(ticker.FixedLocator(odd_numbers))
        ax.set_title(ylabel(key), fontsize=2.3*size)
        ax.set_ylabel('Parameter Values',fontsize=2.8*size)
        ax.set_xlabel('AL iteration',fontsize=2.8*size)
        ax.grid(True,alpha=0.35)
        ax.set_xticks(xticks)
        mx,mn = -1e16,1e16
        for j,(pn,v) in enumerate(params.items()):
            maxp = np.max(v)
            minp = np.min(v)
            if maxp > mx:
                mx = maxp
            if minp < mn:
                mn = minp
            last = j+1 == len(params)
            ax.plot(iters, v,label = label_nicely(pn, last), color= colors[j],lw=0.2*size,marker=markers[j] )
        r = mx - mn
        ax.set_ylim(mn-0.1*r,mx+0.35*r)
        #ax.set_yscale('log')
        ax.legend(frameon=False,ncol=int(len(params)/3)+1,fontsize=2.1*size)
    for i in range(naxis, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('pev.png', bbox_inches='tight')
    plt.close()

if __name__=='__main__':
    main()
    
