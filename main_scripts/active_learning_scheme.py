# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:24:31 2022

@author: n.patsalidis
"""
import sys
from time import perf_counter
sys.path.insert(0, '/Users/n.patsalidis/Desktop/PHD/REPOSITORIES/FFDevelop')
import os
import pandas as pd
import FF_Develop as ff
import numpy as np
import argparse
import math



def main():
    
    t0 = perf_counter()
    
    argparser = argparse.ArgumentParser(description="FF development step in Active learning scheme: created by Nikolaos Patsalidis")
    
    hnum ='Iteration number'
    hdatapath='main path of data'
    hinputfile = 'input for FF develop file'
    hbatchsize = 'size of the batch'
    hsigma ='sigma value for random petrubation'
    hexisting_data = 'iteration of already existing data'
    hm="mass map. Provide a python dictionary in as a string"
    hc="charge map. Provide a python dictionary in as a string"
    htt="Target sampling temperature"
    hbs="Input sampling beta"


    argparser.add_argument('-n',"--num",metavar=None,
            type=int, required=True, help=hnum)
    argparser.add_argument('-dp',"--datapath",metavar=None,
            type=str, required=False,default='data', help=hdatapath)
    argparser.add_argument('-f',"--inputfile",metavar=None,
            type=str, required=True, help=hinputfile)
    argparser.add_argument('-b',"--batchsize",metavar=None,
            type=int, required=False,default=100, help=hbatchsize)
    argparser.add_argument('-s',"--sigma",metavar=None,
            type=float, required=True,default=0.01, help=hsigma)
    argparser.add_argument('-exd',"--existing_data",metavar=None,
            type=int, required=False,default=-1, help=hexisting_data)
    argparser.add_argument('-m',"--sampling_method",metavar=None,
            type=str, default='perturbation', help='sampling method')
    
    argparser.add_argument('-cm',"--charge_map",metavar=None,
            type=str, required=True, help=hc)
    
    argparser.add_argument('-mm',"--mass_map",metavar=None,
            type=str, required=True, help=hm)
    argparser.add_argument('-t',"--Ttarget",metavar=None,
            type=float, required=False,default=300.0, help=htt)
    
    argparser.add_argument('-bs',"--beta_sampling",metavar=None,
            type=float, required=False,default=1.67, help=htt)
    
    parsed_args = argparser.parse_args()
    
    beta_sampling = parsed_args.beta_sampling

    kB = 0.00198720375145233 # kcal/mol
    beta_target = 1.0/(kB*parsed_args.Ttarget)

    setup = ff.Setup_Interfacial_Optimization(parsed_args.inputfile)
    al = ff.al_help()

    path = parsed_args.datapath
    num = parsed_args.num
    
    path_log = '{:s}/L{:d}'.format(path,num)
    path_xyz ='{:s}/D{:d}'.format(path,num)
    
    al.log_to_xyz(path_log, path_xyz)
    data = pd.DataFrame()
    
    ff.GeneralFunctions.make_dir('distr')
    for n in range(num+1):
        print('Reading iteration {:d} '.format(n))
        sys.stdout.flush()
        
        path_xyz ='{:s}/D{:d}'.format(path,n)
        df = al.data_from_directory(path_xyz)

        al.make_absolute_Energy_to_interaction(df,setup)
        data = data.append(df , ignore_index=True)
        #ff.Data_Manager.distribution(data['Energy'],'distr/data{:d}.png'.format(n))

    data = al.clean_data(data,setup)
    ff.Data_Manager(data,setup).distribution('Energy')
    setup.run = num

    
    batchsize= parsed_args.batchsize
    
    t1 = perf_counter()
    
    data, model_costs, setup, optimizer = al.solve_model(data,setup)
    print('solving time = {:.3e} '.format(perf_counter()-t1))
    
    sys.stdout.flush()

    al.write_errors(model_costs,num) 
    if num >= parsed_args.existing_data:
        
        t2 = perf_counter()
        if parsed_args.sampling_method == 'perturbation':
            candidate_data = al.make_random_petrubations(data, sigma = parsed_args.sigma)
        elif parsed_args.sampling_method == 'md':
            parsed_args.writing_path='lammps_working'
            candidate_data, beta_sampling = al.sample_via_lammps(data,setup, parsed_args, beta_sampling)
        elif parsed_args.sampling_method == 'mc':
            candidate_data, beta_sampling = al.MC_sample(data, setup, parsed_args.sigma, beta_sampling)
        else:
            raise NotImplementedError(f'Candidate Sampling Method "{parsed_args.sampling_method}" is unknown')
        #if num != 0:
           # al.plot_candidate_distribution(candidate_data,setup )
        
        with open('beta_sampling_value','w') as f:
            f.write(f'{beta_sampling}')
            f.closed
        kB = 0.00198720375145233
        tsamp = 1/(kB*beta_sampling)
        print(f'AL iteration {num} beta_sampling =  {beta_sampling}, Tsample = {tsamp} K ')
        
        print('Candidate sampling time = {:.3e} '.format(perf_counter()-t2))
        #selected_data = al.disimilarity_selection(data, setup, candidate_data,batchsize  )
        
        t2 = perf_counter()
        if len(candidate_data) <= batchsize:
            selected_data = candidate_data
        else:
            selected_data = al.random_selection(data, setup, candidate_data,batchsize  )
        selected_data = selected_data.reset_index(drop=True)
 
        print('selecting time = {:.3e} '.format(perf_counter()-t2))
        sys.stdout.flush()

        size = len(selected_data)
        t3 = perf_counter()
        al.write_drun('{:s}/J{:d}'.format(path,num+1), selected_data,size,num+1)
        print('writing time = {:.3e} '.format(perf_counter()-t3))
   
    return
    
    
if __name__=='__main__':
    main()
