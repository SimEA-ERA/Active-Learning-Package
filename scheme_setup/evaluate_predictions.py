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
    hinputfile = 'input for FF develop file'
    hdatapath='main path of data'
    
    argparser.add_argument('-n',"--num",metavar=None,
            type=int, required=True, help=hnum)
    argparser.add_argument('-dp',"--datapath",metavar=None,
            type=str, required=False,default='data', help=hdatapath)
    
    argparser.add_argument('-f',"--inputfile",metavar=None,
            type=str, required=True, help=hinputfile)
    
    parsed_args = argparser.parse_args()
    
    path = parsed_args.datapath
    num = parsed_args.num
    
    infile = parsed_args.inputfile
    setup = ff.Setup_Interfacial_Optimization(infile)
    
    path_log = '{:s}/L{:d}'.format(path,num)
    path_xyz ='{:s}/D{:d}'.format(path,num)
    al = ff.al_help()
    al.log_to_xyz(path_log, path_xyz)
    
    data = al.data_from_directory(path_xyz)
    al.make_absolute_Energy_to_interaction(data,setup)
   
    #data = al.clean_data(data,setup,prefix='evaluation data')
    nld =setup.nLD
    print('Evaluating for nld = {:d} models'.format(setup.nLD))
    err = dict()
    err[nld]= al.predict_model(data,setup)
    errors = al.rearrange_dict_keys(err)
    e = dict()
    for k in errors.keys():
        e['pred_'+k.lower()] = [v for v in errors[k].values()]
    al.write_errors(e,num) 

    print('FF development Prediction Evaluation Time --> {:.3e} sec'.format(perf_counter()-t0))
    
    
if __name__=='__main__':
    main()
