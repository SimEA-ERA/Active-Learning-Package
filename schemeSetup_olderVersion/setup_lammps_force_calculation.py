import sys
from time import perf_counter
import os
import pandas as pd
import FF_Develop as ff
import numpy as np
import argparse
 
def main():
    
    t0 = perf_counter()
    
    argparser = argparse.ArgumentParser(description="Making a FF developed structure to lammps.dat file -- Step in Active learning scheme-- !! created by Nikolaos Patsalidis !!")
    hfile="file to read the structure"
    hpath="writing path"
    
    
    
    argparser.add_argument('-df',"--datafile",metavar=None,
            type=str, required=True, help=hfile)
    argparser.add_argument('-wp',"--writing_path",metavar=None,
            type=str, required=True, help=hpath)

    argparser.add_argument('-in',"--ffinputfile",metavar=None,
            type=str, required=True, help='ff develop input file')
    ht="types map. Provide a python dictionary in as a string"
    hm="mass map. Provide a python dictionary in as a string"
    hc="charge map. Provide a python dictionary in as a string"
    
    
    
    argparser.add_argument('-cm',"--charge_map",metavar=None,
            type=str, required=True, help=hc)
    
    argparser.add_argument('-mm',"--mass_map",metavar=None,
            type=str, required=True, help=hm)
    
    
    parsed_args = argparser.parse_args()
    ff.al_help.lammps_force_calculation_setup(parsed_args)
if __name__=='__main__':
    main()
