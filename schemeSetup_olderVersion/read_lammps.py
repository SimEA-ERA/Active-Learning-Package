# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 18:57:27 2024

@author: n.patsalidis
"""

import lammpsreader as lr
import numpy as np
import FF_Develop as ff
import argparse

def main():
    hpath="working lammps path"
    argparser = argparse.ArgumentParser(description="Comparing forces from Lammps -- Step in Active learning scheme-- !! created by Nikolaos Patsalidis !!")
    argparser.add_argument('-wp',"--writing_path",metavar=None,
            type=str, required=True, help=hpath)
    argparser.add_argument('-l',"--logfile",metavar=None,
            type=str, required=True, help='Gaussian log file')

    args = argparser.parse_args()

    a  = lr.LammpsTrajReader('{:s}/forces.lammpstrj'.format(args.writing_path))
    a.readNextStep()
    forces = np.array([ np.array(a.data[f],dtype=float) for f in ['fx','fy','fz']])
    forces = forces.reshape(forces.shape[::-1] )
    
    #df = ff.Data_Manager.read_Gaussian_output(args.logfile,read_forces=True)
    #for f1,f2 in zip(forces,df.loc[0,'Forces']):
    #print(f1,f2)
    
    return

if __name__=='__main__':
    main()
