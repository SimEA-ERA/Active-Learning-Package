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
    
    
    parsed_args = argparser.parse_args()
    
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

    necessary_columns = ['sys_name','natoms','coords','at_type','bodies', 'Energy']
    
    batchsize= parsed_args.batchsize
    
    t1 = perf_counter()
    
    data, model_costs, setup, optimizer = al.solve_model(data,setup)
    print('solving time = {:.3e} '.format(perf_counter()-t1))
    
    sys.stdout.flush()

    al.write_errors(model_costs,num) 
    
    if num >= parsed_args.existing_data:
        t2 = perf_counter()
        parsed_args.ffinputfile='Results/{:d}/runned.in'.format(num)
        r_setup = ff.Setup_Interfacial_Optimization( parsed_args.ffinputfile )
        if parsed_args.sampling_method == 'perturbation':
            possible_data = al.make_random_petrubations(data[necessary_columns],
                                                 sigma=parsed_args.sigma, method=setup.perturbation_method)
            selected_data = al.disimilarity_selection(data,r_setup,possible_data,batchsize)
        
        elif parsed_args.sampling_method == 'md':
            parsed_args.writing_path='lammps_working'
            possible_data = al.sample_via_lammps(data,r_setup,parsed_args)
            selected_data = al.disimilarity_selection(data,r_setup,possible_data,batchsize)
        elif parsed_args.sampling_method == 'mc':
            possible_data = al.MC_sample(data, r_setup, parsed_args)
            selected_data = al.disimilarity_selection(data,r_setup,possible_data,batchsize)
        
        selected_data = selected_data.reset_index(drop=True)
   #     for j in selected_data.index:
    #        print(selected_data.loc[j,'coords'])
 
        print('selecting time = {:.3e} '.format(perf_counter()-t2))
        sys.stdout.flush()

        size = len(selected_data)
        t3 = perf_counter()
        al.write_drun('{:s}/J{:d}'.format(path,num+1), selected_data,size,num+1)
        print('writing time = {:.3e} '.format(perf_counter()-t3))
   
    return
    
    
if __name__=='__main__':
    main()
