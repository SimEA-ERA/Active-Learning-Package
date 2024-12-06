"""
Created on Fri Mar 19 18:49:40 2021

@author: nikolas
This is a library to develop FF. 
Currently it supports FF development for physisorption of small molecules to surfaces
"""
from numba import jit,prange
#from numba.experimental import jitclass
import os.path
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import copy
from matplotlib import pyplot as plt
from scipy.optimize import minimize, differential_evolution,dual_annealing
from time import perf_counter
import coloredlogs
import logging
import itertools

import collections
import six

import lammpsreader as lammps_reader
class GeometryTransformations:

    @staticmethod
    def rotation_matrix_x(angle_rad):
        """Generate a rotation matrix for rotating about the x-axis."""
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])

    @staticmethod
    def rotation_matrix_y(angle_rad):
        """Generate a rotation matrix for rotating about the y-axis."""
        return np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])

    @staticmethod
    def rotation_matrix_z(angle_rad):
        """Generate a rotation matrix for rotating about the z-axis."""
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])

    @staticmethod
    def rotate_coordinates(coords, angle_rad_x, angle_rad_y, angle_rad_z):
        """
        Rotate coordinates around all three axes (x, y, z) by specified angles in degrees.

        Args:
        coords (np.array): Nx3 numpy array of x, y, z coordinates.
        angle_rad_x, angle_rad_y, angle_rad_z (float): Rotation angles in radians for each axis.

        Returns:
        np.array: Rotated coordinates.
        """

        # Get rotation matrices
        rot_matrix_x = GeometryTransformations.rotation_matrix_x(angle_rad_x)
        rot_matrix_y = GeometryTransformations.rotation_matrix_y(angle_rad_y)
        rot_matrix_z = GeometryTransformations.rotation_matrix_z(angle_rad_z)

        # Combined rotation matrix; order of multiplication is important
        rot_matrix = rot_matrix_x @ rot_matrix_y @ rot_matrix_z

        # Apply the combined rotation matrix to coordinates
        return np.dot(coords, rot_matrix.T)

class al_help():
    def __init__(self):
        lammps_style_map = {'Morse':'morse', 
                                'MorseBond':'morse',
                                'Bezier': 'table linear 50000',
                                'harmonic':'harmonic',
                                }
        self.lammps_style_map = lammps_style_map
        return

    def map_to_lammps_style(self,style):
        return self.lammps_style_map[style]

    @staticmethod
    def decompose_data_to_structures(df,structs):
        nc = [] ; nt = [] ; nn = [] ; ns = []
        oid = [] ; 
        for j, dat in df.iterrows():
            c = np.array(dat['coords'])
            tys = np.array(dat['at_type'])
            for i,struct in enumerate(structs):
                f = np.zeros(len(tys),dtype=bool)
                for t in struct:
                    f = np.logical_or(t == tys, f)
                newc =  c[f]
                newtys = tys[f]
                natoms = newtys.shape[0]
                sys_name =str(j)+'-'+ dat['sys_name']+'__ref'+str(i)
                nc.append(newc)
                nt.append(list(newtys))
                nn.append(natoms)
                ns.append(sys_name)
                oid.append(j)
        decdata = pd.DataFrame({'coords':nc,'at_type':nt,'natoms':nn,'sys_name':ns,'original_index':oid})
        return decdata 

    @staticmethod
    def read_lammps_structs(fname,inv_types):
        a  = lammps_reader.LammpsTrajReader(fname)
        coords = []
        types = []
        natoms = []
        while( a.readNextStep() is not None):
            crds = np.array([ np.array(a.data[c], dtype=float) for c in ['x','y','z']])
            crds = crds.T 
            cbox = a.box_bounds
            tricl = cbox[:,2] != 0.0
            if tricl.any():
                raise NotImplementedError('Triclinic boxes are not implemented')
            offset = cbox[:,0]
            box  = cbox[:,1]-offset
            b2 =box/2

            crds -= offset
            # Applying periodic condition
            cref = crds[0]
            r = crds - cref
            for j in range(3):
                fm = r[:,j] < - b2[j]
                fp = r[:,j] >   b2[j]
                crds[:,j][fm] +=box[j]
                crds[:,j][fp] -=box[j]
            #print('Reading Lammps')
            crds -= crds.mean(axis=0)
            tys =  [inv_types[t] for t in a.data['type'] ]
            coords.append(crds)
            types.append(tys)
            natoms.append(crds.shape[0])
        return pd.DataFrame({'coords':coords,'at_type':types,'natoms':np.array(natoms)})

    @staticmethod
    def sample_via_lammps(data,setup,parsed_args):
        al = al_help()
        cols = data.columns
        possible_data = pd.DataFrame()

        #select randomly 0.01% of the index
        unq_sys = np.unique(data['sys_name'])
        #old_bonded_inters = 'dumb variable'
        for sname in unq_sys:
            fs = data['sys_name'] == sname
            udat = data [ fs ]
            us = data['Uclass'].to_numpy()[ fs ]
            us = us - us.min()
            ps = np.exp(-us/setup.bS)
            ps /= ps.sum()
            print('Choosing initial configuration for {:s}'.format(sname))
            sys.stdout.flush()
            t0 = perf_counter()
            chosen_index = np.random.choice(np.arange(0,len(udat),1,dtype=int) ,1, replace=False,p=ps)
            print('Choice complete time --> {:.3e} sec'.format(perf_counter()-t0))
            sys.stdout.flush()
            #print('beginning md condifiguration index',chosen_index)
            for ii,j in enumerate(udat.index[chosen_index]):
                #print('running lammps of data point {:d}, sys_name = {:s}'.format(j, sname))
                df = udat.loc[j,cols]
            
                maps = al.get_lammps_maps(df,parsed_args)
                inv_types = {v:k for k,v in maps['types_map'].items()}

                bonded_inters = al.write_Lammps_dat(df,setup,maps['types_map'],maps['mass_map'],maps['charge_map'])       
                al_help.write_potential_files(setup,df, parsed_args,bonded_inters)

                al.write_rigid_fixes(setup,'lammps_working',maps['types_map'])
                sn = sname.replace('(','_')
                sn = sn.replace(')','_')
                naming = 'iter{:d}_{:d}-{:s}'.format(parsed_args.num+1,ii,sn)
                c1 = 'cd lammps_working' 
                c2 = 'bash lammps_sample_run.sh'
                c3 = 'mkdir {0:s} ; cp samples.lammpstrj {0:s}/ ;  mv structure.dat {0:s}'.format(naming)
                c4 = 'mv potential.inc {0:s}/ ;  mv rigid_fixes.inc {0:s}'.format(naming)
                c5 = ' cd - '
                command  = '  ;  '.join([c1,c2,c3,c4,c5])
                os.system(command)

                new_data = al_help.read_lammps_structs('lammps_working/samples.lammpstrj',inv_types)
                
                #select randomly 0.01% of the new_data
                
                chosen_index = new_data.index #[ np.random.choice( new_data.index, max(1,int(0.1*len(new_data))), replace=False) ]
                #print('from runned data chose configurations',chosen_index)
                new_data['sys_name'] = sname
                dfn = new_data.loc[chosen_index]
                possible_data = possible_data.append(dfn, ignore_index=True)
        
        print('Lammps Simulations complete')
        sys.stdout.flush()
        return possible_data

    @staticmethod
    def write_potential_files(setup,data_point,parsed_args,bonded_inters):
        maps = al_help.get_lammps_maps(data_point,parsed_args)

        al_help.write_Lammps_potential_file(setup, data_point, bonded_inters,maps['types_map'],maps['charge_map'])
        return 

    @staticmethod
    def write_rigid_fixes(setup,path,types_map):
        with open('{:s}/rigid_fixes.inc'.format(path),'w') as f:
            for j,k in enumerate(setup.rigid_types):
                f.write('group rigid_group{:d} type {:s}\n'.format(j,' '.join( [ str(types_map[v]) for v in k] ))  )
                f.write('fix {:d} rigid_group{:d} rigid {:s} langevin 1 500 $(1*dt) 15 \n'.format(j+3,j,setup.rigid_style) )

            f.closed
    @staticmethod
    def make_interactions(data,setup):

        manager = Data_Manager(data,setup)
        manager.setup_bonds(setup.distance_map) 

        intersHandler = Interactions(data,setup,atom_model = setup.representation, 
                                        find_vdw_connected = True,
                                        find_vdw_unconnected=True,
                                        find_densities=True,
                                        vdw_bond_dist=0,
                                        rho_r0=setup.rho_r0,rho_rc=setup.rho_rc)
        
        intersHandler.InteractionsForData(setup)
        intersHandler.calc_interaction_values()
        return
    
    @staticmethod
    def evaluate_potential(data,setup):
        al_help.make_interactions(data,setup)
        
        
        train_indexes, dev_indexes = Data_Manager(data,setup).train_development_split()
        
        setup.optimize = False

        optimizer = Interfacial_FF_Optimizer(data,train_indexes, dev_indexes, setup)
        #print('dataevaluations')
        #results
        optimizer.optimize_params()
        optimizer.UvectorizedContribution_ondata(which='opt',dataset='all')
        
        return optimizer

    @staticmethod
    def get_lammps_maps(data_point,parsed_args):
        maps= dict()
        maps['types_map'] = { k:j+1 for j,k in enumerate(np.unique(data_point['at_type'])) }
        
        for name in ['mass_map','charge_map']:
            a = getattr(parsed_args,name)
            a = a.split(',')
            #print(a)
            d = dict()
            for j in a:
                l = j.split(':')
                k = str(l[0])
                v = float(l[1]) 
                d[k]=v
            maps[name] = d
        return maps

    @staticmethod
    def lammps_force_calculation_setup(parsed_args):
        al = al_help()
        
        setup = Setup_Interfacial_Optimization(parsed_args.ffinputfile)
        
        data = Data_Manager.read_xyz(parsed_args.datafile)
        manager = Data_Manager(data,setup)
        manager.setup_bonds(setup.distance_map) 
        al_help.make_absolute_Energy_to_interaction(data,setup)
        al_help.evaluate_potential(data,setup)
        for j,df in data.iterrows():
            maps = al.get_lammps_maps(df,parsed_args)
            bonded_inters = al.write_Lammps_dat(df,setup,maps['types_map'],maps['mass_map'],maps['charge_map'])
            al_help.write_potential_files(setup,df,parsed_args,bonded_inters)
        
        return
    @staticmethod
    def get_molidmap(Bonds,atom_types):

        Bonds = Interactions.bonds_to_python(Bonds)
        neibs = Interactions.get_neibs(Bonds,len(atom_types))
        at_types=atom_types
        connectivity = Interactions.get_connectivity(Bonds,at_types,[])
            
        neibs = Interactions.get_neibs(connectivity,len(atom_types))
        molecule_ids = Interactions.get_unconnected_structures(neibs)
       
        mol_id_map = dict()
        nummols=1
        for molecule in molecule_ids:
            for j in molecule:
                mol_id_map[j] = nummols
            nummols+=1
       
        return mol_id_map

    @staticmethod
    def get_Lammps_needed_info(df):
        ty = df['at_type']
        coords = np.array(df['coords'])
        box = 10*(np.max(coords,axis=0)-np.min(coords,axis=0))+100
        #print(box,coords.shape)
        cm = np.mean(coords,axis=0)
        coords += box/2 - cm

        mol_id_map = al_help.get_molidmap(df['Bonds'],ty)
        return ty, coords, box, mol_id_map
    
    @staticmethod
    def write_Lammps_dat(df,setup,types_map,mass_map,charge_map):
        def n_inters(x):
            ntypes = len(x)
            ni = np.sum([v.shape[0] for v in x.values()])
            return ntypes, ni
        def map_to_lammps_types(x):
            return { k:j+1  for j,k in enumerate(x.keys()) }
        wp = 'lammps_working/'
        GeneralFunctions.make_dir(wp)
        ty, coords, box,  mol_id_map = al_help.get_Lammps_needed_info(df)
        unty = np.unique(ty)
        natoms = len(ty)
        
        bonds = df['interactions']['connectivity']
        nbond_types, nbonds = n_inters(bonds)
        bonds_lmap = map_to_lammps_types(bonds)

        angles = df['interactions']['angles']
        nangle_types, nangles = n_inters(angles)
        angles_lmap = map_to_lammps_types(angles)
        
        dihedrals = df['interactions']['dihedrals']
        ndihedral_types, ndihedrals = n_inters(dihedrals)
        dihedrals_lmap = map_to_lammps_types(dihedrals)

        bonded_inters = {'bonds':bonds_lmap, 'angles':angles_lmap, 'dihedrals':dihedrals_lmap }

        if natoms != coords.shape[0]:
            raise ValueError('number of atoms and coordinates do not much')
        
        with open('{:s}/structure.dat'.format(wp),'w') as f:
            f.write('# generated from FF_develop kind of dataframes\n# created within AL scheme step by Nikolaos Patsalidis [ EDFT = {:4.6f}, Upred ={:4.6f} ]  \n'.format(df['Energy'],df['Uclass']))
            
            f.write('{:d} atoms\n'.format(natoms))
            if nbonds >0:
                f.write('{:d} bonds\n'.format(nbonds))
            if nangles >0:
                f.write('{:d} angles\n'.format(nangles))
            if ndihedrals >0:
                f.write('{:d} dihedrals\n'.format(ndihedrals))
 
            f.write('\n')
            f.write('{:d} atom types\n'.format(len(unty)))
            if nbond_types >0:
                f.write('{:d} bond types\n'.format(nbond_types))
            if nangle_types > 0:
                f.write('{:d} angle types\n'.format(nangle_types))
            if ndihedral_types > 0:
                f.write('{:d} dihedral types\n'.format(ndihedral_types))

            for b,s in zip(box,['x','y','z']):
                f.write('\n{:4.2f} {:4.2f} {:s}lo {:s}hi'.format(0.0,b,s,s) )

            f.write('\n\n\n\nMasses\n\n')
            for t in unty:
                mt=types_map[t]
                mass = mass_map[t]
                f.write('{} {:4.6f}\n'.format(mt,mass) )
            f.write('\n\nAtoms\n\n')
            
            for iat,(t,c) in enumerate(zip(ty,coords)):
                tmd =types_map[t]
                charge = charge_map[t]
                mol_id = mol_id_map[iat]
                f.write('{:d}  {:d}  {:d}  {:4.6f}   {:.16e}   {:.16e}   {:.16e}   \n'.format(iat+1,mol_id,tmd,charge,*c)  )
            
            if nbonds >0:
                f.write('\nBonds\n\n')
                j = 1
                for k,b in bonds.items():
                    ty = bonds_lmap[k] 
                    for v in b:
                        f.write('{:d} {:d} {:d} {:d}\n'.format(j,ty,v[0]+1,v[1]+1))
                        j+=1
 
            if nangles > 0:
                f.write('\nAngles\n\n')
                j = 1
                for k,a in angles.items():
                    ty = angles_lmap[k]
                    for v in a:
                        f.write('{:d} {:d} {:d} {:d} {:d}\n'.format(j, ty, v[0]+1,v[1]+1,v[2]+1))
                        j += 1
            
            if ndihedrals > 0:
                f.write('\nDihedrals\n\n')
                j = 1
                for k, d in dihedrals.items():
                    ty = dihedrals_lmap[k]
                    for v in d:
                        f.write('{:d} {:d} {:d} {:d} {:d} {:d}\n'.format(j, ty, v[0]+1,v[1]+1,v[2]+1,v[3]+1))
                        j += 1


            f.closed


        return bonded_inters
    
    @staticmethod
    def write_Lammps_potential_file(setup,data_point,bonded_inters,types_map,charge_map):
        
        with open('lammps_working/potential.inc','w') as f:
            f.write('# generated from FF_develop results\n# created within AL scheme step by Nikolaos Patsalidis  \n')
            for k,t in types_map.items():
                f.write('group {:s} type {:}\n'.format(k,t) )
            f.write('\n\n')
            for k,t in types_map.items():
                f.write('set type {:} charge {:}\n'.format(t,charge_map[k]) )
            f.write('\n\n')
            
            f.write('\n\n')

            try:
                models = getattr(setup,'opt_models')
            except AttributeError:
                models = getattr(setup,'init_models')
                which ='init'
            else:
                which ='opt'
            
            # 1,0 categorize models, clear models irrelevant to struct
            classified_models = dict()
            for name,model in models.items():
                lc = model.lammps_class
                if model.type not in data_point['values'][model.feature]:
                    continue
                if model.type[0] not in types_map or model.type[1] not in types_map:
                    continue
                if lc not in classified_models:
                    classified_models[lc] = [model]
                else:
                    classified_models[lc].append(model)
            # 1 end
            # 1.1 handle extra_pair coeff
            epc = setup.extra_pair_coeff
            
            added_extra = ' '
            for ty,v in epc.items():
                if ty[0] in types_map and ty[1] in types_map:
                    added_extra += v[0] + ' 20' if v[0] not in added_extra else ' '
                    write_extra_pairs =True
            # 1.1 end
            f.write('\n')

            added_ld =' '
            if 'pair' in classified_models:
                # 1.2 add ld to hybrid style if you find even one LD potential
                ##### The function writing the LD potential will handle multiple LDs 
                for model in classified_models['pair']:
                    if model.category == 'LD' and model.num<setup.nLD:
                        added_ld = 'local/density'
                # 1.2 end
                f.write('pair_style hybrid/overlay   {:s}   morse 20.0 table linear 50000 {:s}\n'.format(added_extra,added_ld))
                for  model in classified_models['pair']:
                    ty = model.type
                    m = model.model
                    c = model.category
                    n = model.num
                    tyl = (types_map[ty[0]], types_map[ty[1]])
                    if int(tyl[0]) > int(tyl[1]):
                        tyl = (tyl[1],tyl[0])
                    if n>= getattr(setup,'n'+c):
                        continue
                    if m =='Morse':
                        pars = [ model.pinfo['De'].value, model.pinfo['alpha'].value, model.pinfo['re'].value]
                        args = (*tyl, *pars)
                        s = 'pair_coeff {:} {:} morse {:.16e}  {:.16e}  {:.16e} \n'.format(*args)
                        f.write(s)
                    if m =='Bezier' and c =='PW' :
                        args = (*tyl,)
                        s = 'pair_coeff {:} {:}  table tablePW.tab {:}-{:} \n'.format(args[0],args[1],ty[0],ty[1])
                        f.write(s)
                
                setup.write_PWtable(types_map,50000,which=which)
            elif write_extra_pairs:
                f.write('pair_style hybrid/overlay {:s} {:s} \n'.format(added_extra,added_ld))

            for ty,v in epc.items():
                if ty[0] in types_map and ty[1] in types_map:
                    t0 = str(types_map[ty[0]])
                    t1 = str(types_map[ty[1]])
                    t = ' '.join([t0,t1]) if  int(t0) <= int(t1) else ' '.join([t1,t0])
                    va = ' '.join(v)
                    s = 'pair_coeff {:s} {:s} \n'.format(t,va)
                    f.write(s)
            if setup.nLD > 0 and 'pair' in classified_models:
                s = 'pair_coeff * * local/density frhos{:d}.ld \n'.format(setup.nLD)
                f.write(s)
                setup.write_LDtable(types_map,50000,which=which)
            f.write('\n')

            for lc, models in classified_models.items():
                if lc =='bond':

                    f.write('\n')

                    styles = np.unique([ model.model for model in models])
                    styles_per_type = { m1.type : np.unique([ m2.model for m2 in models if m2.type==m1.type]) for m1 in models }
                    written_types = set()
                    hybrid=False
                    
                    if len(styles) >1:
                        hybrid_styles = []
                        for k,spt in styles_per_type.items():
                            if  'Bezier' in spt:
                                hybrid_styles.append('table linear 50000')
                            elif 'MorseBond' in spt or 'Morse' in spt:
                                hybrid_styles.append('morse') 
                        if len( np.unique(hybrid_styles) )>1: 
                            hybrid=True
                            f.write('bond_style hybrid {:s} \n'.format('  '.join(hybrid_styles)) )
                            print('Using hybrid bond model')

                    if not hybrid:
                        if len(styles) ==1:
                            f.write('bond_style {:s} \n'.format(models[0].lammps_style))
                        else:
                            f.write('bond_style {:s} \n'.format(hybrid_styles[0]))
                    for model in models:
                        st = styles_per_type[model.type]
                        typ = bonded_inters['bonds'][model.type]
                        if len(st) ==1:
                            style = model.lammps_style if hybrid else ''
                            if st[0] =='morse':
                                pars = [ model.pinfo['De'].value, model.pinfo['alpha'].value, model.pinfo['re'].value]
                                f.write('bond_coeff {:d} {:s} {:.16e}  {:.16e}  {:.16e} \n'.format(typ, style, *pars) )

                            if st[0] =='harmonic':
                                pars = [ model.pinfo['k'].value , model.pinfo['r0'].value ]
                                f.write('bond_coeff {:d} {:s} {:.16e}  {:.16e} \n'.format(typ, style, *pars) )
                            if st[1] =='Bezier':
                                f.write('bond_coeff {:d} {:s} tableBO.tab {:s}-{:s} \n'.format(typ,style,ty[0],ty[1]))
                        else:
                            # add both style to a table 
                            style = 'table' if hybrid else ''
                            ty = model.type
                            if ty not in written_types:
                                f.write('bond_coeff {:d} {:s} tableBO.tab {:s}-{:s} \n'.format(typ,style,ty[0],ty[1]))
                      
                        written_types.add(model.type)
                    setup.write_BOtable(types_map,50000,which=which)
            
            for lc, models in classified_models.items():
            
                if lc=='angle':

                    f.write('\n')

                    styles = np.unique([ model.model for model in models])
                    styles_per_type = { m1.type : np.unique([ m2.model for m2 in models if m2.type==m1.type]) for m1 in models }
                    written_types = set()
                    hybrid=False
                    
                    if len(styles) >1:
                        hybrid_styles = []
                        for k,spt in styles_per_type.items():
                            if  'Bezier' in spt:
                                hybrid_styles.append('table linear 50000')
                            elif 'MorseBond' in spt or 'Morse' in spt:
                                hybrid_styles.append('morse') 
                            elif 'harmonic' in spt:
                                hybrid_styles.append('harmonic')
                        if len( np.unique(hybrid_styles) )>1: 
                            hybrid=True
                            f.write('angle_style hybrid {:s} \n'.format('  '.join(hybrid_styles)) )
                            print('Using hybrid angle model')

                    if not hybrid:
                        if len(styles) ==1:
                            f.write('angle_style {:s} \n'.format(models[0].lammps_style))
                        else:
                            f.write('angle_style {:s} \n'.format(hybrid_styles[0]))
                    for model in models:
                        st = styles_per_type[model.type]
                        typ = bonded_inters['angles'][model.type]
                        ty = model.type
                        if len(st) ==1:
                            style = model.lammps_style if hybrid else ''
                            if st[0] =='morse':
                                pars = [ model.pinfo['De'].value, model.pinfo['alpha'].value, model.pinfo['re'].value]
                                f.write('angle_coeff {:d} {:s} {:.16e}  {:.16e}  {:.16e} \n'.format(typ, style, *pars) )

                            if st[0] =='harmonic':
                                pars = [ model.pinfo['k'].value , model.pinfo['th0'].value*180/np.pi ]
                                f.write('angle_coeff {:d} {:s} {:.16e}  {:.16e} \n'.format(typ, style, *pars) )
                            if st[0] =='Bezier':
                                f.write('angle_coeff {:d} {:s} tableAN.tab {:s}-{:s}-{:s} \n'.format(typ,style,ty[0],ty[1],ty[2]))
                        else:
                            # add both style to a table 
                            style = 'table' if hybrid else ''
                            ty = model.type
                            if ty not in written_types:
                                f.write('angle_coeff {:d} {:s} tableAN.tab {:s}-{:s}-{:s} \n'.format(typ,style,ty[0],ty[1],ty[2]))
                      
                        written_types.add(model.type)
                    setup.write_ANtable(types_map,50000,which=which)

                    f.write('\n')
                f.write('\n')
            f.write('\n')
            for line in setup.lammps_potential_extra_lines:
                f.write(line +' \n')
            f.closed
        return
    @staticmethod
    def rearrange_dict_keys(dictionary):
        '''
        Changes the order of the keys to access data
        Parameters
        ----------
        dictionary : Dictionary of dictionaries with the same keys.
        Returns
        -------
        x : Dictionary with the second set of keys being now first.

        '''
        x = {k2 : {k1:None for k1 in dictionary} for k2 in dictionary[list(dictionary.keys())[0]]}
        for k1 in dictionary:
            for k2 in dictionary[k1]:
                x[k2][k1] = dictionary[k1][k2]
        return x
 
    @staticmethod
    def predict_model(data,setup):
        #t0 =perf_counter()

        GeneralFunctions.make_dir(setup.runpath)
        
        al_help.evaluate_potential(data,setup)
        
        dataeval = Interfacial_Evaluator(data,setup,prefix='all')
        #print(data)
        #funs = ['MAE','relMAE','BIAS','STD','MSE']
        funs = ['MAE','MSE','relMAE','relMSE','MAX','relBIAS']
        
        cols = ['sys_name']
        df = cols + ['relMAE','relMSE','relBIAS','MAE','MSE']
        if setup.costf not in funs: funs.append(setup.costf)
        
        trev = dataeval.make_evaluation_table(funs,cols,save_csv='predict.csv')
        print('prediction errors \n----------\ns',trev[df])
        
        dataeval.plot_predict_vs_target(path = setup.runpath,title='prediction dataset',
                                          fname='predict.png',size=2.35,compare='sys_name')
        
        
        dataeval.plot_eners(subsample=1,fname='predicteners.png')
       
        return  trev.loc['TOTAL',['MAE','MSE',setup.costf]].to_dict()

    @staticmethod
    def set_L_toRhoMax(dm,setup):
        models = copy.deepcopy(setup.init_models)
        for k, model in models.items():   
            if 'LD' not in k:
                continue
            ty=model.type
            rho_max = dm.get_distribution(ty,'rhos').max()
            kp='L'
               
            if model.pinfo[kp].opt == False:
                continue
            else:
                model.pinfo[kp].low_bound = 1.2*rho_max
                model.pinfo[kp].upper_bound = 1.5*rho_max
            #print('setting L = {:4.3f} for LD{:d} and type {}'.format(rho_max,i,ty))
            model.pinfo[kp].value = 1.35*rho_max
            setattr(setup,'init_models',models)
        return


    @staticmethod
    def data_from_directory(path):
        data = pd.DataFrame()
        for fname in os.listdir(path):
            df = Data_Manager.read_xyz('{:s}/{:s}'.format(path,fname))
            data = data.append(df,ignore_index=True) 
        return data

    @staticmethod
    def solve_model(data,setup):

        t0 =perf_counter()
        
        GeneralFunctions.make_dir(setup.runpath)
        
        al_help.make_interactions(data,setup)
        dataMan = Data_Manager(data,setup)
        train_indexes, dev_indexes = dataMan.train_development_split()
        
        al_help.set_L_toRhoMax(dataMan, setup)
               
        optimizer = Interfacial_FF_Optimizer(data,train_indexes, dev_indexes,setup)
         
        optimizer.optimize_params()
        
        #print('evaluations')
        #results
        optimizer.UvectorizedContribution_ondata(which='opt',dataset='all')
        
        setup.write_running_output()
        
        funs = ['MAE','MSE','MAX','BIAS','STD']
        
        cols = ['sys_name']
        #df = cols + ['relMAE','relMSE','relBIAS','MAE','MSE']
        if setup.costf not in funs: funs.append(setup.costf)
        

        train_eval = Interfacial_Evaluator(data.loc[train_indexes],setup,prefix='train')
        dev_eval = Interfacial_Evaluator(data.loc[dev_indexes],setup,prefix='development')
        
        train_eval.plot_predict_vs_target(path = setup.runpath,title='train dataset',
                                          fname='train.png',size=2.35,compare='sys_name')
        
        dev_eval.plot_predict_vs_target(path = setup.runpath,title='development dataset',
                                          fname='development.png',size=2.35,compare='sys_name')
        
        
        #trev = train_eval.make_evaluation_table(funs,cols,save_csv='train.csv')
        tv = dev_eval.make_evaluation_table(funs,cols,save_csv='development.csv')
        errors = {'mae': tv.loc['TOTAL','MAE'],
                  'mse': tv.loc['TOTAL','MSE']
                  }
        errors[setup.costf.lower()] = tv.loc['TOTAL',setup.costf]

        train_eval.plot_eners(subsample=1,path=setup.runpath,fname='eners.png')
        
    
        dataMan.save_selected_data(setup.runpath+'/frames.xyz', data,labels=['sys_name','Energy'])
        
        setup.plot_models(which='opt')
        
        tf =perf_counter() - t0
        print('Time consumed for solving the model --> {:4.3e} min'.format(tf/60))
        return  data, errors , optimizer

    @staticmethod
    def make_random_petrubations(data,nper=10,sigma=0.05,method='atoms'):
        possible_data= pd.DataFrame()
        for j,dat in data.iterrows():
            for k in range(nper):
                d = dat.copy()
                new_coords = al_help.petrube_coords(dat,sigma,method)
                d['coords'] = new_coords
                possible_data = pd.concat([possible_data, d.to_frame().T], ignore_index=True)
        return possible_data
    @staticmethod
    def petrube_coords(dat,sigma,method):
        
        c = np.array(dat['coords'])
        if method == 'atoms':
            c += np.random.normal(0,sigma,c.shape)
        elif method =='rigid':
            bodies = dat['bodies']
            for j,body in bodies.items():
                b = body
                c[b] = al_help.rottrans_randomly(c[b],sigma)
        else:
            raise Exception('method {:s} is not Implemente. Give prober name for petrubation method'.format(method))
        return c
    @staticmethod
    def rottrans_randomly(c,sigma):
        rtrans = np.random.normal(0,sigma,3)
        rrot = np.pi*np.random.normal(0,sigma,3)
        #cm = c.mean(axis=0)
        cc = c.copy()
        cc = GeometryTransformations.rotate_coordinates(cc,*rrot)
        cc +=  rtrans
        return cc

    @staticmethod
    def similarity_vector(dval):
        vec = []
        for k,v1 in dval.items():
            try:
                v1.values()
            except AttributeError:
                continue
            for v2 in v1.values():
                v = v2.copy()
                v.sort()
                vec.extend(v)
        return np.array(vec)

    @staticmethod
    def disimilarity_selection(data,setup,possible_data,batchsize,method='sys_name'):
        al_help.make_interactions(possible_data,setup) 
        possible_data = al_help.clean_possible_data(possible_data,setup)
        dis = []
        for k1,d1 in possible_data.iterrows():
            disim = 0
            vec1 = al_help.similarity_vector(d1['values'])
            synm = d1['sys_name']
            jdat=0
            for k2,d2 in data[ data['sys_name'] == synm].iterrows():
                vec2 = al_help.similarity_vector(d2['values'])
                rvec = vec1-vec2
                disim+= np.sum(rvec*rvec)**0.5
                jdat+=1
            dis.append(disim)
        dis = np.array(dis)
        possible_data['disimilarity'] = dis
        fcheck = np.isnan(possible_data['disimilarity'])
        for j,d in possible_data[fcheck].iterrows():
            print('Warning: Removing nan values', j,d['sys_name'],d['disimilarity'])
            sys.stdout.flush()
        possible_data = possible_data[~fcheck]
        print('evaluating in selection step with nld = {:d}'.format(setup.nLD))
        possible_data['Energy']=np.zeros(len(possible_data))
        al_help.evaluate_potential(possible_data,setup)

        possible_data['Prob. select'] = possible_data['disimilarity']/dis.sum()
        
        Ucls = possible_data['Uclass'].to_numpy()
        n = len(possible_data)
        indx = np.arange(0,n,1,dtype=int)
        if n<=batchsize:
            batchsize=min(n,batchsize)
            method = None

        if method is None:
            chosen = np.random.choice(indx,size=batchsize,replace=False,p=possible_data['Prob. select'])
        else:
            col = possible_data[method].to_numpy()
            colvals = np.unique(col)
            Props = [1/np.count_nonzero(col == c) for c in colvals]
            Props = np.array(Props)/np.sum(Props)

            names = np.random.choice(  colvals,size=batchsize,replace=True,p = Props  )
            nums = {name: np.count_nonzero(names==name) for name in colvals} 
            
            chosen = []
            psel = possible_data['Prob. select'].to_numpy()
            for name,num in nums.items():
                fc = col == name 
                pc = psel[ fc ] 
                norm = pc.sum()
                if norm ==0:
                    print ('Warning: system {:s} prosum is zero numbers = {:} '.format(name,pc) )
                    continue
                pc /= norm
                #pc[np.isnan(pc)] = 1e-2
                ix = indx[ fc ] 
                ux = Ucls[fc]
                
                snum = min(ix.shape[0] , num)
                ux -= ux.min()
                n_chosen = 0
                infinit_index =0
                serial_ix = [i for i in range(ix.shape[0]) ]
                select_any = False
                while(n_chosen <snum):
                    infinit_index+=1
                    if infinit_index>(snum*100):
                        args = (name,infinit_index,n_chosen,ix.shape[0])
                        print( 'Selecting randomly! system {:s} --> Made {:d} tries, selected  {:d}/{:d} data'.format(*args))
                        select_any = True
                    
                    if infinit_index>(snum*102):
                        raise Exception('Infinite while loop in selection algorithm')
                    six = np.random.choice(serial_ix,replace=False,p = pc)
                    ec = ux[six]
                    canditate = ix [six]
                    if (np.exp(-ec/setup.bS) > np.random.uniform(0,1) and canditate not in chosen) or select_any:
                        n_chosen+=1
                        chosen.append( canditate )
                print( 'system selection {:s} --> Made {:d} tries, selected  {:d}/{:d} data'.format(name,infinit_index,n_chosen,ix.shape[0]))
                sys.stdout.flush()
        select = np.zeros(n,dtype=bool)
        for j in indx:
            if j in chosen:
                select[j] = True
        possible_data['select'] = select

        return  possible_data [ possible_data ['select'] ] 

    @staticmethod
    def write_gjf(path,fname,atom_types,coords,
                  multiplicity=1):
        file = '{:s}/{:s}'.format(path,fname)
        #lines = ['%nprocshared=16\n%mem=16000MB\n#p wb97xd/def2TZVP scf=xqc scfcyc=999\n\nTemplate file\n\n']    
        lines = ['%nprocshared=16\n%mem=16000MB\n#p wb97xd/def2TZVP scf(xqc,Conver=6) scfcyc=999\n\nTemplate file\n\n']    
        lines.append(' 0 {:1d}\n'.format(multiplicity))
        for i in range(len(atom_types)):
            at = atom_types[i]
            c = coords[i]
            lines.append(' {:s}    {:.16e}  {:.16e}  {:.16e}\n'.format(at,c[0],c[1],c[2]))
        lines.append('\n')
        with open(file,'w') as f:
            for line in lines:
                f.write(line)
            f.close()
        return

    @staticmethod
    def write_array_batch(path,size,num):
        bash_script_lines = [
        "#!/bin/bash  ",
        "#SBATCH --job-name=R{:d}  ".format(num),
        "#SBATCH --output=output  ",
        "#SBATCH --error=error  ",
        "#SBATCH --array=0-{0:d}%{0:d}  ".format(size),
        "#SBATCH --nodes=1  ",
        "#SBATCH --ntasks-per-node=16  ",
        "#SBATCH --partition=milan  ",
        "#SBATCH --time=1:19:00  ",
        "",
        "module load Gaussian/16.C.01-AVX2  ",
        "source $g16root/g16/bsd/g16.profile  ",
        'GAUSS_SCRDIR="/nvme/scratch/npatsalidis/"  ',
        "",
        '# get the name of the directory to process from the directories file  ',
        "parent='.'  ",
        "",
        'linecm=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${parent}/runlist.txt)  ',
        "",
        '# change into the directory  ',
        "",
        'echo "${linecm}"  ',
        "eval $linecm  "
        ]
                    
        with open('{:s}/run.sh'.format(path),'w') as f:
            for line in bash_script_lines:
                f.write(line +'\n')
            f.closed
        return
    
    @staticmethod
    def rw(source_path,destination_path):
        with open(source_path, 'r') as source_file:
            file_contents = source_file.read()

        # Write the contents to the destination file
        with open(destination_path, 'w') as destination_file:
            destination_file.write(file_contents)
        return

    @staticmethod
    def write_drun(path,selected_data,size,num):
        selected_data = selected_data.sort_values(by='natoms', ascending=False)
        GeneralFunctions.make_dir(path)
        sys_iter =0
        lst = []
        sys_iter=0
        names = []
        for j, dat in selected_data.iterrows():
            sys_name =  dat['sys_name']
       
            mult = 2 if '(2)' in sys_name else 1
            name = sys_name.replace('(2)','') + '_'+str(sys_iter)
            names.append(name)
            fname = name +'.gjf'
            exec_command = 'cd {0:s} ; g16 < {0:s}.com > {0:s}.log'.format(name)
            lst.append(exec_command)
            al_help.write_gjf(path,fname,dat['at_type'],dat['coords'],mult)
            sys_iter +=1
        with open('{:s}/runlist.txt'.format(path),'w') as f:
            for line in lst:
                f.write(line+' \n')
            f.closed
        al_help.write_array_batch(path,size,num)
        
        return names

    @staticmethod
    def log_to_xyz(input_path,output_path):
        GeneralFunctions.make_dir(output_path)
        
        for fname in [x for x in os.listdir(input_path) if x[-4:] =='.log'] :
            try:
                data = Data_Manager.read_Gaussian_output('{:s}/{:s}'.format(input_path,fname))
            except Exception as ve:
                print('warning: DFT null data --> {}'.format(ve))
                continue
            data['gaussianfile'] = data['filename']
         
            labels = [c for c in data.columns if c not in ['coords','natoms','at_type','filename'] ]
            Data_Manager.save_selected_data('{:s}/{:s}.xyz'.format(output_path,fname.split('.')[0]), data,labels=labels)
        return

    @staticmethod
    def make_absolute_Energy_to_interaction(data,setup):
        
        references = setup.reference_energy
        data['Absolute Energy'] = data['Energy'].copy()
        
        data['Eref'] = np.zeros(len(data),dtype=float)
        
        #print(data['Energy'])
        for k, ref in references.items():
            if k =='atoms':
                for j,dat in data.iterrows():
                    tys = np.array(dat['at_type'])
                    for a,val in ref.items():
                        data.loc[j,'Eref'] += np.count_nonzero(tys == a)*val
            elif k=='reference':
                # 0 initialize reference data file if not exist
                path = 'reference_data'
                GeneralFunctions.make_dir(path)
                
                try:
                    references =  pd.read_pickle('reference_data/reference_data.pickle')
                except FileNotFoundError:
                    references = pd.DataFrame(columns=['sys_name','natoms','at_type','coords'])
                    references.to_pickle('reference_data/reference_data.pickle')
                #0 end

                # 1 decompose data into reference structure based on setup
                df_structs = al_help.decompose_data_to_structures(data,setup.struct_types)
                # 1 end

                # 2 check which are almost identical to previous to avoid doing multiple DFT for references
                nexrd, exrd =  al_help.check_for_existing_reference_data(df_structs)
                print('Existing referenced data = {:d}, Non existing reference data = {:d}'.format(len(exrd),len(nexrd)))
                sys.stdout.flush()
                # 2 end
                
                # 3 if the non existing are more than 0 write files and do DFT
                if len(nexrd) >0:
                    names = al_help.write_drun(path,nexrd,len(nexrd),99)

                    for gjfile in [x for x in os.listdir(path) if x[-4:]=='.gjf']:
                        name = gjfile.split('.')[0]
                        dr = '{:s}/{:s}'.format(path,name)
                        GeneralFunctions.make_dir(dr)
                        os.system('mv {:s}/{:s} {:s}/{:s}.com'.format(path,gjfile,dr,name))
                    os.system('cp batch_refs.sh {:s}/'.format(path))
                    os.system('cd  {:s}/ ; bash batch_refs.sh'.format(path))
                else:
                    names = []
                 # 3 end

                # 4 find from the stored reference data (exrd) or the new reference data (nexrd) and add the reference energy
                exrd_oi = exrd['original_index'].to_numpy()
                #exrd_sys_names = exrd['sys_name'].to_numpy()
                #nexrd_sys_names = nexrd['sys_name'].to_numpy()
                for j,dat in data.iterrows():
                    eref = 0
                    eref += np.sum(exrd['Energy'] [ j == exrd_oi ])
                    dirs = [dr for dr in os.listdir(path) if dr in names and str(j) == dr.split('-')[0] ]
                    for dr in dirs:
                         df = Data_Manager.read_Gaussian_output('{0:s}/{1:s}/{1:s}.log'.format(path,dr))
                         references = references.append(df,ignore_index=True)
                         eref += df['Energy'].to_numpy()[-1]
                    data.loc[j,'Eref'] += eref
                    #print('Reference energy:', j, eref, dirs)
                
                sys.stdout.flush()
                # 4.1 store anew reference data
                references.to_pickle('reference_data/reference_data.pickle')
                # 4.1 end 
                # 4 end 
            elif k=='value':
                data['Eref'] = +ref
            else:
                raise NotImplementedError('{:s} method is not implemented'.format(k))
        # final step: remove reference energy
        data['Energy'] -= data['Eref']
        #print(data[['sys_name','Energy','Eref']])
        # final step end
        return

    @staticmethod
    def invariant(dat,pickdat):
        ty1 = dat['at_type']
        ty2 = pickdat['at_type']
        for i1,i2 in zip(ty1,ty2):
            if i1 !=i2: return False
        c1 = dat['coords']
        c2 = pickdat['coords']
        c1m = np.mean(c1,axis=0)
        c2m = np.mean(c2,axis=0)
        t1 = c1 - c1m 
        t2 = c2 - c2m
        d1 = np.sum(t1*t1,axis=1)**0.5
        d2 = np.sum(t2*t2,axis=1)**0.5
        r = d1-d2
        d = np.dot(r,r.T)**0.5/r.size
        return d < 1e-4

    @staticmethod
    def check_for_existing_reference_data(data):
        pickled_data = pd.read_pickle('reference_data/reference_data.pickle')
        existing_indexes = []
        data['Energy'] = np.zeros(len(data),dtype=float)
        for j,dat in data.iterrows():
            for p,pickdat in pickled_data.iterrows():
                if  al_help.invariant(dat,pickdat):
                    existing_indexes.append(j)
                    data.loc[j,'Energy'] = pickdat['Energy']
                    break
        
        nexrd = data [ ~data.index.isin(existing_indexes) ]
        if len(existing_indexes) == 0:
            exrd = pd.DataFrame(columns=data.columns)
        else:
            exrd = data.iloc[ existing_indexes ]
        return nexrd, exrd
    
    
    @staticmethod
    def write_errors(err,num):
        GeneralFunctions.make_dir('evaluations')
        for k,e in err.items():
            with open('evaluations/{:s}.dat'.format(k),'a') as f:
                f.write('{:d} {:s} \n'.format(num,'   '.join([str(i) for i in e])))
                f.closed
        return
    @staticmethod
    @jit(nopython=True)
    def calc_dmin(c1,c2):
        dmin = 1e16
        for i in range(c1.shape[0]):
            for j in range(c2.shape[0]):
                r = c1[i] - c2[j]
                d = np.dot(r,r)**0.5
                if d <dmin:
                    dmin = d
        return dmin
    @staticmethod
    def clean_possible_data(data,setup):
        rc = setup.rclean
        #sc = setup.struct_types
        keep_index = []
        for j,df in data.iterrows():
            c = df['coords']
            st = df['structs']
            dmax = 0
            for k1,i1 in st.items():
                for k2,i2 in st.items():
                    if k1 == k2: continue
                    dmin = al_help.calc_dmin(c[i1],c[i2])
                    if dmin > dmax:
                        dmax = dmin
            if dmax < rc:        
                keep_index.append(j)
        return data.loc[keep_index]
    @staticmethod
    def clean_data(data,setup,prefix=''):
        n = len(data)
        sysnames = data['sys_name'].to_numpy()
        
        ener = data['Energy'].to_numpy()
        
        me = {js:np.min(ener[js == sysnames] ) for js in np.unique(sysnames)}
        
        mineners = np.empty(n,dtype=float)
        for i in range(n):
            mineners[i] = me[ sysnames[i] ]
        
        re = np.abs(mineners - ener)
        pe = np.exp(-re/(2*setup.bS))
        if n > 300:
            indexes = data.index
            #ix = np.random.choice(data.index,int(setup.clean_perc*n),replace=False,p=ps)
            ix = []
            for i,p in enumerate(pe):
                if p > np.random.uniform(0,1):
                    ix.append(indexes[i])

        else:
            ix = data.index
        data = data.iloc[ix] 
        print('Kept {:4.3f} % of the {:s} data'.format(len(data)*100/n,prefix))
        return data

class logs():
    '''
    A class for modifying the loggers
    '''
    def __init__(self):
        self.logger = self.get_logger()
        
    def get_logger(self):
    
        LOGGING_LEVEL = logging.CRITICAL
        
        logger = logging.getLogger(__name__)
        logger.setLevel(LOGGING_LEVEL)
        logFormat = '%(asctime)s\n[ %(levelname)s ]\n[%(filename)s -> %(funcName)s() -> line %(lineno)s]\n%(message)s\n --------'
        formatter = logging.Formatter(logFormat)
        

        if not logger.hasHandlers():
            logfile_handler = logging.FileHandler('FF_develop.log',mode='w')
            logfile_handler.setFormatter(formatter)
            logger.addHandler(logfile_handler)
            self.log_file = logfile_handler
            stream = logging.StreamHandler()
            stream.setLevel(LOGGING_LEVEL)
            stream.setFormatter(formatter)
            
            logger.addHandler(stream)
         
        fieldstyle = {'asctime': {'color': 'magenta'},
                      'levelname': {'bold': True, 'color': 'green'},
                      'filename':{'color':'green'},
                      'funcName':{'color':'green'},
                      'lineno':{'color':'green'}}
                                           
        levelstyles = {'critical': {'bold': True, 'color': 'red'},
                       'debug': {'color': 'blue'}, 
                       'error': {'color': 'red'}, 
                       'info': {'color':'cyan'},
                       'warning': {'color': 'yellow'}}
        
        coloredlogs.install(level=LOGGING_LEVEL,
                            logger=logger,
                            fmt=logFormat,
                            datefmt='%H:%M:%S',
                            field_styles=fieldstyle,
                            level_styles=levelstyles)
        return logger
    
    def __del__(self):
        try:
            self.logger.handlers.clear()
        except:
            pass
        try:    
            self.log_file.close()
        except:
            pass
        return
logobj = logs()        
logger = logobj.logger



class GeneralFunctions:
    
    @staticmethod
    def iterable(arg):
        return (
            isinstance(arg, collections.Iterable) 
            and not isinstance(arg, six.string_types)
        )
    @staticmethod
    def make_dir(name):
        name = name.replace('\\','/')
        n = name.split('/')
        lists = ['/'.join(n[:i+1]) for i in range(len(n))]  
        a = 0 
        for l in lists:
            if not os.path.exists(l):
                a = os.system('mkdir {:s}'.format(l))
                if a!=0:
                    s = l.replace('/','\\')
                    a = os.system('mkdir {:s}'.format(s))
        
        return a
    
    @staticmethod
    def weighting(data,weighting_method,T=1,w=1):
        E= np.array(data['Energy'])
        unsys = np.unique(data['sys_name'])
        wts = np.empty_like(E)
        for s1 in unsys:
            f = data['sys_name'] == s1
            E = data['Energy'] [f ]
    
            if weighting_method =='constant':    
                weights = np.ones(len(E))*w
            elif weighting_method == 'Boltzman':
                weights = w*np.exp(-E/T)/np.sum(np.exp(-E/T))
                weights /=weights.mean()
            elif weighting_method == 'linear':
                weights = w*(E.max()-E+1.0)/(E.max()-E.min()+1.0)
            elif weighting_method =='gmin':
                weights = np.ones(len(E))*w
                weights[E.argmin()] = 3
            wts [f] = weights
    
        return wts
    
    @staticmethod
    def get_colorbrewer_colors(n):
        colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
        nc = len(colors)
        ni = int(nc/n)
        if ni ==0:
            colors = [colors[0] for i in range(nc)]
        else:
            colors = [colors[i] for i in range(0,nc,ni)]
        return colors
    
    
class VectorGeometry:
    
    @jit(nopython=True,fastmath=True)
    def calc_dist(r1,r2):
        r = r2 - r1
        d = np.sqrt(np.dot(r,r))
        return d
    
    @jit(nopython=True,fastmath=True)
    def calc_unitvec(r1,r2):
        r = r2 - r1
        d = np.sqrt(np.dot(r,r))
        return r/d
    
    @jit(nopython=True,fastmath=True)
    def calc_angle(r1,r2,r3):
        d1 = r1 -r2 ; d2 = r3-r2
        nd1 = np.sqrt(np.dot(d1,d1))
        nd2 = np.sqrt(np.dot(d2,d2))
        cos_th = np.dot(d1,d2)/(nd1*nd2)
        return np.arccos(cos_th)
    
    @jit(nopython=True,fastmath=True)
    def calc_dihedral(r1,r2,r3,r4):
        d1 = r2-r1
        d2 = r3-r2
        d3 = r4-r3
        c1 = np.cross(d1,d2)
        c2 = np.cross(d2,d3)
        n1 = c1/np.sqrt(np.dot(c1,c1))
        n2 = c2/np.sqrt(np.dot(c2,c2))
        m1= np.cross(n1,d2/np.sqrt(np.dot(d2,d2)))
        x= np.dot(n1,n2)
        y= np.dot(m1,n2)
        dihedral = np.arctan2(y, x)
        return dihedral
            

class MathAssist:
    def __init__(self):
        return
    
    @staticmethod
    def numba_combinations(n,r):
        a = 1
        for i in range(r+1,n+1):
            a*=i
        b = MathAssist().numba_factorial(n-r)
        return float(a/b)
    
    @staticmethod
    def numba_factorial(n):
        if n==0:
            return 1
        f = 1
        for i in range(1,n+1):
            f *= i
        return f
    
    @staticmethod
    def Atridag(n):
        A =np.zeros((n,n))
        A[0,0]=1 ; A[n-1,n-1]=1
        for i in range(1,n-1):
            A[i,i]=4
            A[i,i+1]=1
            A[i,i-1]=1#
        return A
    
    @staticmethod
    def norm2(r1):
        return np.sqrt(np.dot(r1,r1))
    
    @staticmethod
    def norm1(r1):
        r = np.sum(np.abs(r1))
        return r
    
    @staticmethod
    def most_min(arr,m):
        return arr.argsort()[0:m]

    @staticmethod
    def norm2squared(r1):
        r = np.dot(r1,r1)
        return r


class harmonic3:
    def __init__(self,r,params):
        self.r = r
        self.params = params
        return 
    
    def u_vectorized(self):
        r0, k1, k2, k3 = self.params
        r = self.r
        
        r_r0 = r - r0
        r_r0m2 = r_r0*r_r0
        r_r0m3 = r_r0m2*r_r0
        r_r0m4 = r_r0m2*r_r0m2
        
        u = k1*r_r0m2 + k2*r_r0m3 + k3*r_r0m4 
        
        return u
    
    def find_dydx(self):
        r0, k1, k2, k3 = self.params
        r = self.r
        r_r0 = r - r0
        r_r0m2 = r_r0*r_r0
        r_r0m3 = r_r0m2*r_r0
        
        g = 2*k1*r_r0 + 3*k2*r_r0m2 + 4*k3*r_r0m3
        
        self.dydx = g
        
        return g
    
    def find_params_gradient(self):
        
        r0, k1, k2, k3 = self.params
        r = self.r
        r_r0 = r - r0
        r_r0m2 = r_r0*r_r0
        r_r0m3 = r_r0m2*r_r0
        r_r0m4 = r_r0m2*r_r0m2
        
        g = np.empty((4,r.shape[0]), dtype=np.float64)
        
        
        g[0] = - (2*k1*r_r0 + 3*k2*r_r0m2 + 4*k3*r_r0m3)
        g[1] = r_r0m2 
        g[2] = r_r0m3 
        g[3] = r_r0m4 
        
        self.params_gradient = g
        
        return g
    


class harmonic:
    
    def __init__(self,r,params):
        self.r = r
        self.params = params
        return 
    
    def u_vectorized(self):
        r0, k = self.params
        r = self.r
        u = k*(r-r0)**2
        return u
    
    def find_dydx(self):
        r0, k = self.params
        r = self.r
        g = 2*k*(r-r0)
        self.dydx = g
        return g
    
    def find_params_gradient(self):
        
        r0, k = self.params
        r = self.r
        
        g = np.empty((2,r.shape[0]), dtype=np.float64)
        r_r0 = r - r0
        g[0] = -2*k*(r_r0)
        g[1] = r_r0 * r_r0
        
        self.params_gradient = g
        return g




class LJ:
    
    def __init__(self,r,params):
        self.r = r
        self.params = params
        return 
    
    def u_vectorized(self):
        sigma, epsilon = self.params
        r = self.r
        
        so_r6 = (sigma/r)**6
        so_r12 = so_r6*so_r6
        
        u = 4 * epsilon * (so_r12 - so_r6) 
        return u
        
    
    def find_dydx(self):
        sigma, epsilon = self.params
        r = self.r
        
        so_r6 = (sigma/r)**6
        so_r12 = so_r6*so_r6
        
        g = 4 * epsilon * ( 6*so_r6 - 12*so_r12 )/r
        
        self.dydx = g
        return g
       
    def find_params_gradient(self):
        sigma, epsilon = self.params
        r = self.r
        
        so_r6 = (sigma/r)**6
        so_r12 = so_r6*so_r6
        
        g = np.zeros((2,r.shape[0]))
        
        g[0] = 4 * epsilon * (12*so_r12  - 6*so_r6 )/sigma 
        g[1] = 4 * (so_r12 - so_r6) 
        
        self.params_gradient = g
        return g


class MorseBond:
    
    def __init__(self,r,params):
        self.r = r
        self.params = params
        return 
    def u_vectorized(self):
        x = self.params
        r = self.r
        
        re = x[0]
        De = x[1]
        alpha = x[2]
        t1 = -alpha*(r-re)
        e1 = np.exp(t1)
        me1 = 1 - e1
        u = De*me1*me1
        return u
        
    
    def find_dydx(self):
        x = self.params
        r = self.r
        
        re = x[0]
        De = x[1]
        alpha = x[2]
        
        t1 = - alpha*(r-re)
        e1 = np.exp(t1)
        g = 2 * alpha* De *(1-e1)*e1
        self.dydx = g
        return g
       
    def find_params_gradient(self):
        r = self.r
        x = self.params
        re, De, alpha = x
        nr = r.shape[0]
        n = self.params.shape[0]
        
        g = np.zeros((n,nr))
        
        r_re = r-re
        t1 = -alpha*r_re
        e1 = np.exp(t1)
        me1 = 1 - e1
        me1_e1 = me1*e1
        g[0] = -2 * De * alpha * me1_e1 #dudre
        g[1] = me1*me1 # dudDe
        g[2] = 2 * De * r_re * me1_e1 # dudalpha
        
        self.params_gradient = g
        return g

    
    
class Morse:
    def __init__(self,r,params):
        self.r = r
        self.params = params
        return 
    def u_vectorized(self):
        x = self.params
        r = self.r
        
        re = x[0]
        De = x[1]
        alpha = x[2]
        t1 = -alpha*(r-re)
        u = De*(np.exp(2.0*t1)-2.0*np.exp(t1))
        
        return u
    
    def find_dydx(self):
        x = self.params
        r = self.r
        
        re = x[0]
        De = x[1]
        alpha = x[2]
        t1 = - alpha*(r-re)
        g = -2 * alpha* De * (np.exp(2*t1) - np.exp(t1))
        self.dydx = g
        return g
       
    def find_params_gradient(self):
        r = self.r
        x = self.params
        re, De, alpha = x
        nr = r.shape[0]
        n = self.params.shape[0]
        
        g = np.zeros((n,nr))
        
        r_re = r-re
        t1 = -alpha*r_re
        e1 = np.exp(t1)
        e2 = np.exp(2*t1)
        rr = (e2  - e1 )
        g[0] = 2 * De * alpha * rr #dudre
        g[1] = rr - e1 # dudDe
        g[2] = -2 * De * r_re * rr # dudalpha
        
        self.params_gradient = g
        return g
    

class Bezier(MathAssist):
    def __init__(self,xvals, params,  M = None ):
        self.xvals = xvals
        self.params = params
        self.L = params[0]
        self.y0 = params[1]
        self.ye = params[-1]
        # 1 y defines bezier y points.y[0] = y[1] = y[ 2 ] := ycontrol[1], y[-3]=y[-2] =y[-1] := ycontrol[-1]
        ### L := y[0] and x[i] is equidistant from 0 to L
        y = np.empty(params.shape[0]+3,dtype=np.float64)
        y[2:-2] = params[1:]
        y[0] = self.y0
        y[1] = self.y0
        y[-1] = self.ye
        y[-2] = self.ye
        self.ycontrol = y
        self.npoints = y.shape[0]
        dx = self.L/float(self.npoints-1)
        
        x = np.empty_like(y)
        for i in range(self.npoints):
            x[i] = float(i)*dx
        self.xcontrol = x
        
        if M is None:
            self.M = self.matrix(self.npoints)
        else:
            self.M = M
            
        self.find_taus()
        
        return
    
    def matrix_coef(self,i,j,N):
        s = (-1)**(j-i)
        nj = self.numba_combinations(N, j)
        ij = self.numba_combinations(j,i)
        mij = s*nj*ij
        return mij

    def matrix(self,Npoints):
        N = Npoints - 1 # Npoints = (N+1 sum is from 0 to N)
        M = np.zeros((Npoints,Npoints))
        for i in range(Npoints):
            for j in range(i,Npoints):      
                M[i,j] = self.matrix_coef(i,j,N)
        return M
    
    def find_taus(self):
        xv = self.xvals
        L = self.L
        
        fup = xv > L
        self.taus = self.xvals/L
        self.taus[fup] = 1.0

    def u_serial(self):
        ny = self.npoints
        y = self.ycontrol
        M = self.M
        taus = self.taus
        yr = np.zeros((taus.size,))
        for i in range(ny):
            yv= y[i]
            for j in range(i,ny):
                yr += yv*M[i,j]*taus**j 
        self.ycurve = yr
        return yr
    
    def u_vectorized(self):
        # 1  find taus using newton raphson from x positions(rhos)
        ny = self.npoints
        y = self.ycontrol
        M = self.M
        taus = self.taus
        
        coeff_y_tj = np.zeros((ny,))
        for i in range(ny):
            ry = y[i]
            for j in range(i,ny):
                mij = M[i,j]
                coeff_y_tj[j] += ry * mij
        yr = np.zeros((taus.size,))  # Initialize yr with the same shape as taus
        taus_power = np.ones((taus.size,))
    
        for j in range(ny):
            yr += coeff_y_tj[j] * taus_power
            taus_power *= taus  
        self.ycurve = yr
        return yr
    
    def find_dydx(self):
        y = self.ycontrol
        M = self.M
        n = self.npoints
        if not hasattr(self,'taus_power'):
            self.find_taus_power()
        
        taus_power = self.taus_power
        coeff_tj = np.zeros((n,))
        
        for i in range(n):
            ry = y[i]
            for j in range(i,n):
                mij = M[i,j]
                coeff_tj[j] += ry * mij * j
        
        g0_1 = np.dot(taus_power[0:-1].T,coeff_tj[1:])   
        
        self.dydt = g0_1
        self.dydx = g0_1/self.L # dydt*dtdx
        
        return self.dydx
    
    def find_params_gradient(self):
        
        if not hasattr(self,'dxdxc'):
            C = self.find_dydyc_vectorized()
        else:
            C = self.dxdxc # it is the same as self.dydyc = C
        if not hasattr(self,'dydt'):
           _ = self.find_dydx()
       
        taus = self.taus
        
        nt = taus.shape[0]
        g = np.zeros((self.params.shape[0],nt))
        
        g[0] = self.dydt*(-taus/self.L) # dydt*dtdL
        
        g[1] = C[0] + C[1] + C[2]
        g[-1] = C[-1] + C[-2] + C[-3]
        
        g[2:-1] = C[3:-3]
        self.params_gradient = g
        
        return g
    
    
    def find_dydyc_numerically(self,epsilon=1e-3):
        n = self.npoints
        C = np.zeros(( n, self.taus.shape[0]))
        self.ycontrol_copy = self.ycontrol.copy()
        for i in range(n):
            self.ycontrol = self.ycontrol_copy.copy()
            self.ycontrol[i] +=epsilon
            bu = self.u_vectorized()
            self.ycontrol = self.ycontrol_copy.copy()
            self.ycontrol[i] = self.ycontrol[i]-epsilon 
            bd = self.u_vectorized()
            C[i,:] = (bu-bd)/(2*epsilon)
        return C
    
    def find_taus_power(self):
        n = self.npoints
        taus = self.taus
         
        nt = taus.shape[0]
        taus_power = np.ones((n,nt))
         
        taus_power[1] = taus.copy()
        for j in range(2,n):
            taus_power[j] = taus_power[j-1]*taus
        self.taus_power = taus_power
        return

    def find_dydyc_vectorized(self):
        if not hasattr(self,'taus_power'):
            self.find_taus_power()
        C = np.dot(self.M,self.taus_power)
        self.dydyc = C
        self.dxdxc = C
        return C 
   
    def find_dydyc_serial(self):
        n = self.npoints
        M = self.M
        taus = self.taus
        nt = taus.shape[0]
        C = np.zeros((n, nt))
        taus_power = np.ones((n,nt))
        taus_power[1] = taus
        for j in range(2,n):
            taus_power[j] = taus_power[j-1]*taus
        for i in range(n):
            for j in range(i,n):
                C[i] += M[i,j]*taus_power[j]
        self.dydyc = C
        return C


class TestPotentials:
    def __init__(self,function_name,params,min_value,max_value,dv=0.001,fargs=(),fkwargs={},
                 plot=False,ignore_high_u=1e16):
        self.f = globals()[function_name] # it will actually get the class
        self.vals = np.arange(min_value,max_value,dv)
        self.params = np.array(params,dtype=np.float64) 
        self.b = self.f(self.vals, params,*fargs,**fkwargs)
        self.u = self.b.u_vectorized()
        self.filt =  self.u < ignore_high_u
        self.dv = dv
        self.fargs = fargs
        self.fkwargs = fkwargs
        if plot:
            _ = plt.figure(figsize=(3.3,3.3),dpi=250)
            plt.title(f'Function: {function_name}')
            filt = self.filt
            plt.plot(self.vals[filt],self.u[filt])
            plt.legend(fontsize=6,frameon=False)
            plt.show()
        return
    
    def derivative_check(self,tol=1,plot=False,verbose=False):
        
        b = self.b
        u = self.u
        filt = self.filt
        dv = self.dv
        # Analytical calculation of dudx
        dudx = b.find_dydx()
        
        # numerical calculation of dudx
        dudx_num = np.empty_like(u)
        dudx_num[0] = (-3 * u[0] + 4 * u[1] - u[2]) / (2 * dv)
        dudx_num[-1] = (3 * u[-1] - 4 * u[-2] + u[-3]) / (2 * dv)
        dudx_num[1:-1] = (u[2:] - u[:-2])/(2*dv)
        
        diff = np.abs(dudx_num[filt]-dudx[filt])
        diff_max = diff.max()
        
        if verbose:
            i = diff.argmax()
            print('Maximizing diff at {:d}, x = {:4.5f}'.format(i,self.vals[i]))
            print(diff[i:i+3])
        if plot:
            _ = plt.figure(figsize=(3.3,3.3),dpi=250)
            plt.title('Derivative of dydx')
            plt.plot(self.vals[filt], dudx_num[filt], label='numerical')
            plt.plot(self.vals[filt], dudx[filt], label='analytical',ls='--')
            plt.legend(fontsize=6,frameon=False)
            plt.show()
        if diff_max > tol*dv:
            print("max difference {:4.3e}".format(diff_max))
            print('Derivative Test not passed')
            return
        print("Derivative check ok, max difference {:4.3e}".format(diff_max))
        return
    
    def time_cost(self, Nt=100,verbose=True):
        
        b = self.b
        times  = dict()
        bs = [ copy.deepcopy(self.b) for _ in range(Nt)]
        t_overhead = perf_counter()
        for b,_ in zip(bs,range(Nt)):
            pass
        t_overhead = perf_counter()-t_overhead
        
        t0 = perf_counter()
        for b,_ in zip(bs,range(Nt)):
            u = b.u_vectorized()
        tf = 1000*(perf_counter() - t0 ) - t_overhead

        times['func'] = tf/Nt 
        
        t0 = perf_counter()
        for b,_ in zip(bs,range(Nt)):
            _ = b.find_params_gradient()
        tf = 1000*(perf_counter() - t0 ) - t_overhead
        
        times['grads'] = tf/Nt 
        
        t0 = perf_counter()
        for b,_ in zip(bs,range(Nt)):
            _ = b.find_dydx()
        tf = 1000*(perf_counter() - t0 ) - t_overhead
        
        times['dydx'] = tf/Nt 
        if verbose:
            line = ", ".join(["{:s}  --> {:4.3e} ms".format(k,v) for k,v in times.items() ]) 
            print(" Npoints: {:d} -- -- > {:s} ".format(u.shape[0],line))
        return times
    
    def vectorization_scalability(self,Nt=30,verbose=False,plot=True):
        x0 = 1e-8
        if hasattr(self.b,"L"):
            L = self.b.L
        else:
            L=10.0
        xe =L+x0
        func = [] ; grads = [] ; dydx = []
        Npoints = []
        for dx in [1.0,1e-1,1e-2,1e-3,1e-4,1e-5]:
            xvals = np.arange(x0,xe,dx)
            Npoints.append(xvals.shape[0])
            
            self.b = self.f(xvals, self.params,*self.fargs,**self.fkwargs)
            
            times = self.time_cost(Nt,verbose)
            
            func.append(times['func'])
            grads.append(times['grads'])
            dydx.append(times['dydx'])
            
        func = np.array(func)
        grads = np.array(grads)
        dydx = np.array(dydx)
        Npoints = np.array(Npoints)
        if plot:
            _ = plt.figure(figsize=(3.3,3.3),dpi=250)
            plt.title('vectorization scaling')
            plt.ylabel('ms/point')
            plt.xlabel('Npoints')
            plt.xscale('log')
            plt.yscale('log')
            plt.plot(Npoints, func/Npoints, marker='o', label='func',ls='-')
            plt.plot(Npoints, grads/Npoints,marker='s', label='grads',ls='--')
            plt.plot(Npoints, dydx/Npoints,marker='v', label='dydx',ls=':')
            plt.legend(fontsize=6,frameon=False)
            plt.show()
        return 
    
    def gradient_check(self, epsilon=1e-4,tol=1,plot=False,verbose=False):
        tol=tol*epsilon
        b = self.b
        params = self.params.copy()
        vals = self.vals
        filt = self.filt
        
        g = b.find_params_gradient()
        for i in range(params.shape[0]):
            # Create copies of params and perturb only the i-th parameter for 4th order central difference
            p1p, p1m = params.copy(), params.copy()
            p1p[i] += epsilon
            p1m[i] -= epsilon
            p2p, p2m = params.copy(), params.copy()
            p2p[i] += 2 * epsilon
            p2m[i] -= 2 * epsilon
    
            # Compute function values at perturbed points
            u1p = self.f(vals, p1p, *self.fargs, **self.fkwargs).u_vectorized()
            u1m = self.f(vals, p1m, *self.fargs, **self.fkwargs).u_vectorized()
            u2p = self.f(vals, p2p, *self.fargs, **self.fkwargs).u_vectorized()
            u2m = self.f(vals, p2m, *self.fargs, **self.fkwargs).u_vectorized()
            
            # 4th-order central difference approximation
            g_num = (-u2p + 8 * u1p - 8 * u1m + u2m) / (12 * epsilon)
    
            # Compute the max difference between numerical and analytical gradients
            diff = np.abs(g_num[filt] - g[i][filt]).max()
            if verbose:
                print("parameter {:d} --> max difference {:4.3e}".format(i, diff))
            
            if plot:
                _ = plt.figure(figsize=(3.3,3.3),dpi=250)
                plt.title(f'Gradient of x_{i}')
                plt.plot(vals[filt], g_num[filt], label='numerical')
                plt.plot(vals[filt], g[i][filt], label='analytical',ls='--')
                plt.legend(fontsize=6,frameon=False)
                plt.show()
            if diff > tol:
                
                print('Gradient Test not passed')
                return
        print('Gradient check ok')
        return

    @staticmethod
    def demonstrate_bezier(y=None, dpi=300,size=3.4,fname=None,
                          show_points=True,seed=None,illustration='y'):
        if seed is not None:
            np.random.seed(seed)
        if y is None:
            y = np.array([10.0,0,0,5.0,-12.0,-13.0,-23,0,0])
        elif type(y) is list:
            y = np.array(y,dtype=float)
        if y.shape[0]<=3:
            raise Exception('Number of points should be at least 3 (y.shape[0]>3)')
        if y[0] ==0:
            raise ValueError('First point should not be zero')
        y1 = y.copy()
            
        y1[2:-1] += np.random.normal(0,3.0,y1[2:-1].shape[0])
        y2 = y.copy() 
        y2[0] = 13.0
        y3 = y.copy()
        y3[0] = 7.0
        data = {0:[y,y1],1:[y,y2,y3]}
    
        
        drho=0.01
        
        colors = ['#1b9e77','#d95f02','#7570b3']
        figsize = (2*size,size) 
        fig = plt.figure(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        markers =['*','s','x','v']
        plt.ylabel(r'$f(\rho)$', fontsize=2.5*size,labelpad=8*size)
        plt.xlabel(r'$\rho$', fontsize=2.5*size,labelpad=4*size)
        plt.xticks([])
        plt.yticks([])
        gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
        ax = gs.subplots(sharex='col', sharey='row')
        fig.suptitle(r'Illustration of the Bezier curve construction ',fontsize=2.7*size)
        #ax[1].set_title(r'Varying the  positions of the Bezier CPs ',fontsize=2.5*size)
        
        for i,(k,d) in enumerate(data.items()):
            
            
            for j,yb in enumerate(d):
                #print(drho,yb)
                rh = np.arange(drho,yb[0]+drho,drho)
                b = Bezier(rh,yb)
                u = b.u_serial()
                label=None
                ax[i].plot(rh,u,color=colors[j],lw=0.6*size,label=label)
                if show_points:
                    ax[i].plot(b.xcontrol,b.ycontrol,ls='none',marker=markers[j],markeredgewidth=0.5*size,
                        markersize=2.0*size,fillstyle='none',color=colors[j])
            ax[i].minorticks_on()
            ax[i].tick_params(direction='in', which='minor',length=size*0.6,labelsize=size*1.5)
            ax[i].tick_params(direction='in', which='major',length=size*1.2,labelsize=size*1.5)
            ax[i].tick_params(axis='both', labelsize=size*2.5)
            ax[i].set_xticks([x for x in range(2,int(y[0]+2),2)])
            ax[i].legend(frameon=False,fontsize=2.5*size)
        if fname is not None:
             plt.savefig(fname,bbox_inches='tight')
        #plt.show()
        
        return
        

                
class Setup_Interfacial_Optimization():
    
    defaults = {
        'representation':'AA',
        'storing_path':'FFresults',
        'run': '0',
        'dpi':300,
        'figsize':(3.5,3.5),
        'runpath_attributes':['run'],
        'uncertainty_method':'no',
        'max_ener':1.0,
        'rclean':6.0,
        'max_force':0.003,
        'central_LD_atoms': ['',],
        'split_method':'random',
        'colsplit':'label',
        'train_colvars': ['',],
        'optimization_method':'DA',
        'opt_disp':False,
        'optimize':True,
        'maxiter':30,
        'max_moves':10,
        'nepochs':100,
        'batchsize':10,
        'maxiter_i':50,
        'tolerance':1e-4,
        'sampling_method':'random',
        'seed':1291412,
        'train_perc':0.8,
        'validation_set':'colname1:colv1 , colv2, colv3 & colname2: colv1 , colv2',
        'development_set':'colname1:colv1 , colv2, colv3 & colname2: colv1 , colv2',
        'regularization_method': 'ridge',
        'increased_stochasticity':0.0,
        'reg_par': 0.0,
        'polish':False,
        'popsize':30,
        'mutation':(0.5,1.0),
        'recombination':0.7,
        'initial_temp':5230.0,
        'restart_temp_ratio':2e-5,
        'local_search_scale':1.0,
        'accept':-5.0,
        'visit':2.62,
        'weighting_method':'constant',
        'w':10.0,
        'bT':15.0,
        'bC':30.0,
        'bS':15.0,
        'clean_perc':0.8,
        'costf':'MSE',
        'nLD':1,
        'nPW':1,
        'nBO':2,
        'nAN':2,
        'LD_model':'polynomial',
        'LD_types' : [''],
        'rho_r0' : 1.1,
        'rho_rc': 6.0,
        'distance_map':"dict()",
        'reference_energy':"dict()",
        'struct_types':"[('type1'),('type2','type3')]",
        'rigid_types':"[]",
        'perturbation_method':'atoms',
        'lammps_potential_extra_lines':"['']",
        'rigid_style':'single', 
        'extra_pair_coeff':"{('DUMP','DUMP'):['morse','value','value','value']}"
        }

    executes = ['distance_map','reference_energy','struct_types','rigid_types',
            'lammps_potential_extra_lines','extra_pair_coeff']
    
    def __init__(self,fname):
        '''
        A Constructor of the setup of Interfacial Optimization
        
        Parameters
        ----------
        fname : string
            File to read the setup.
        Raises
        ------
        Exception
            If you initialize wrongly the parameters.
            
        Returns
        -------
        None.        
        '''
        
        #print('setting algorithm from file "{:s}"'.format(fname))
        def my_setattr(self,attrname,val,defaults):
            if attrname not in defaults:
                raise Exception('Uknown input variable "{:s}"'.format(attrname))
            ty = type(defaults[attrname])
            if ty is list or ty is tuple:
                
                tyi = type(defaults[attrname][0])
                attr = ty([tyi(v) for v in val])
            else:
                attr = ty(val)
            setattr(self,attrname,attr)

            return
        # Defaults

        
        defaults = self.defaults
        
        with open(fname,'r') as f:
            lines = f.readlines() 
            f.closed
        
        #strip comments
        for j,line in enumerate(lines):
            for i,s in enumerate(line):
                if '#' == s: 
                    lines[j]=lines[j][:i]+'\n'
                lines[j] = lines[j].strip()
                
        section_lines = dict()
        for j,line in enumerate(lines):
            if '&' in line:
                section_lines[line.split('&')[-1].strip()] = j
        
        #get_attributes
        for j,line in enumerate(lines):
            
            if '&' in line:
                break 
            
            if '=' in line:
                li = line.split('=')
                var = li[0].strip() ; value = li[1].strip()
                if value.isdigit():
                    value = int(value)
                elif value.replace('.','',1).isdigit():
                    value = float(value)
                elif var in self.executes:
                    value = str(value)
                   #print(var,'=',value)
            elif ':' in line:
                li = line.split(':')
                var = li[0].strip() ; value = [] 
                for x in li[1].split():
                    if x.isdigit(): y = int(x)
                    elif x.replace('.','',1).isdigit(): y = float(x)
                    else: y = x
                    value.append(y)
            else:
                continue
   
            my_setattr(self,var,value,defaults)
            
        for atname,de in defaults.items():
            if not hasattr(self,atname):
                setattr(self,atname,de)
        #Get initial conditions
        models = dict()
        for key,sl in section_lines.items():
            name = key
            #print(name)
            obj = self.model_interaction(lines[sl:],key)
            attrname = 'init' +name 
            setattr(self,attrname,obj)
            models[name] = obj
        self.init_models = models
        self.nonchanging_init_models = copy.deepcopy(models)
        #execute the string related commands
        for e in self.executes:
            #print(e)
            exec_string = "setattr(self,e, {:})".format(getattr(self,e))
            #print(exec_string)
            exec(exec_string)
        return 

    def excess_model(self,cat,num):
        return num>=getattr(self,'n'+cat)
    def plot_models(self,which='init',path=None):
        models = getattr(self,which+'_models')
        size=3.3
        if path is None:
            path = self.runpath
        unique_categories = np.unique ( [model.category for k,model in models.items() if not self.excess_model(model.category,model.num) ] ) 
        unique_types = set([model.type for k,model in models.items() if not self.excess_model(model.category,model.num) ] )
        figsize = (len(unique_categories)*size,size) 
        fig,ax = plt.subplots(1,len(unique_categories),figsize=figsize,dpi=300)
        fig.suptitle( r'Potential Function',fontsize=3*size, y=1.18+0.02/(size/2))
        colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
        
        cmap = {t:colors[j] for j,t in enumerate(unique_types) }
        
        dr = 0.01
        if len(unique_categories) == 1: ax = (ax,)
        for j,c in enumerate(unique_categories):
            if c =='PW':
                r = np.arange(0.9,7.0,dr)
                filt_u = 50
                xlabel =r'r \ $\AA$' 
            if c =='BO':
                r = np.arange(0.9,3,dr)
                filt_u= 100 
                xlabel =r'r \ $\AA$' 
            if c =='LD':
                r = np.arange(0.0,7.0,dr)
                filt_u = 1e16
                xlabel =r'$\rho$' 
            if c =='AN':
                r = np.arange(0,2*np.pi+dr,dr)
                filt_u= 70
                xlabel =r'$\theta$ \ degrees' 
            for ty in unique_types:
                current_models = { model for model in models.values() 
                        if model.category == c and model.type == ty and  not self.excess_model(model.category,model.num)
                                 }
                utot = np.zeros(r.shape[0],dtype=float)
                for k,model in enumerate(current_models):
                    if model.model =='Bezier' and c == 'LD':
                        r = np.arange(1e-12,model.pinfo['L'].value,dr)
                    fu = model.function
                    ty = ' '.join( model.type)
                    print(model.name, model.parameters)
                    b = fu(r,model.parameters,*model.model_args)
                    u = b.u_vectorized()
                    if 'PW1' in model.name:
                        print(u.min(),u.max())
                    if c in ['PW','BO','AN'] and len(current_models)>1:
                        if k == 0 : lstyle = '--' 
                        if k == 1: lstyle =':'
                        if k== 2: lstyle='-.' 
                        if k>2:  '-'
                        utot += u
                    else:
                        lstyle='-'
                    
                    label = '{:s} ({:s})'.format(ty,model.model)
                    
                    f = u<filt_u
                    #for i in range(u.argmin(),f.shape[0]):
                    #    if f[i-1] == False: f[i] = False
                    rf = r[f]
                    if c =='AN': rf*=180/np.pi
                 
                    uf = u[f]
                    ax[j].plot(rf,uf,label=label,ls=lstyle,lw=0.35*size,color=cmap[model.type])
                
                if len(current_models)>1 and c in ['PW','BO','AN']:
                    f = utot < filt_u
                    rf = r[f]
                    if c =='AN': rf*=180/np.pi
                    utotf = utot[f] 
                    ax[j].plot(rf,utotf,label=ty +' (tot)',ls='-',lw=0.5*size,color=cmap[model.type])
            ax[j].set_xlabel(xlabel) 
            ax[j].minorticks_on()
            ax[j].tick_params(which='minor',direction='in',axis='x',length=0.75*size)
            ax[j].tick_params(direction='in', which='minor',axis='y',length=0.75*size,)
            ax[j].tick_params(direction='in', which='major',length=1.2*size)
            ax[j].tick_params(axis='both', labelsize=size*1.7)
            nlab = len(ax[j].get_legend_handles_labels()[1])
            ncol = 1 if nlab < 4 else 2
            ax[j].legend(frameon=False,fontsize=1.3*size,loc='upper center', bbox_to_anchor=(0.5, 1.15), shadow=True, ncol=ncol)
        plt.savefig('{:s}/potential.png'.format(path),bbox_inches='tight')
        plt.show()


    class model_interaction():
        def __init__(self,lines,prefix):
            parameter_information =dict()
            self.name = prefix
            inter = tuple(prefix.split()[1:]) # types of atoms
            self.category = prefix.split()[0][:2]
            self.num = int(prefix[2])
            
            if self.category in ['PW',]:
                inter = self.sort_type(inter)
            else:
                inter = tuple(inter)
            self.type = inter
            
            if 'PW' == self.category:
                self.feature = 'vdw'
                self.lammps_class = 'pair'
            elif 'LD' == self.category:
                self.feature =  'rhos'
                self.lammps_class = 'pair'
            elif 'BO' == self.category:
                self.feature = 'connectivity'
                self.lammps_class = 'bond'
            elif 'AN' == self.category:
                self.feature = 'angles'
                self.lammps_class ='angle'
            
            for j,line in enumerate(lines):
                if 'FUNC' in line:
                    self.model = line.split('FUNC')[-1].strip()
                    self.function = globals()[self.model] # it is a class now
                    self.lammps_style = al_help().map_to_lammps_style(self.model)
                if ':' in line:
                    li = line.split(':')
                    var = li[0].strip() ; v = [float(x) for x in li[1].split()]
                    if not (len(v) == 4 or len(v) ==5):
                        raise Exception('In line "{:s}..." Cannot determine parameters. Expected 4 values {:1d} were given " \n Give "value optimize_or_not low_bound upper_bound'.format(line[0:30],len(v)))
                    obj = self.param_info(var,*v)
                    parameter_information[var] = obj
                if '/' in line:
                    break
            #if self.model =='Bezier':
             #   ny = len(parameter_information) + 3
              #  M = Bezier(np.arange(0.1,1,0.1),np.array([1.0 for _ in range(ny)]) ).matrix(ny) 
             #   self.model_args = (M,)
            #else:
            self.model_args = tuple()

            self.pinfo = parameter_information
            return
        
        class param_info:
            def __init__(self,name,value,opt,low_bound,upper_bound,regul=1.0):
                self.name = name
                self.value = float(value)
                self.opt = bool(opt)
                self.low_bound = float(low_bound)
                self.upper_bound = float(upper_bound)
                self.regul = float(regul)
                return
        
        @property
        def parameters(self):
            return np.array([ v.value for v in self.pinfo.values()])
        @property
        def isnotfixed(self):
            return np.array([ v.opt for v in self.pinfo.values()])
        @property
        def low_bounds(self):
            return np.array([ v.low_bound for v in self.pinfo.values()])
        @property
        def upper_bounds(self):
            return np.array([ v.upper_bound for v in self.pinfo.values()])

        @property
        def number_of_parameters(self):
            return len(self.pinfo)
        @property
        def names_of_parameters(self):
            return list(self.pinfo.keys())
    
        @property
        def regular_consts(self):
            return np.array([ v.regul for v in self.pinfo.values()])
    
    
        @staticmethod
        def sort_type(t):
            if len(t) ==2:
                return tuple(np.sort(t))
            elif len(t)==3:
                if t[0]<=t[2]: 
                    ty = tuple(t)
                else:
                    ty = t[2],t[1],t[0]
                return ty
            elif len(t)==4:
                if t[0]<=t[3]: 
                    ty = tuple(t)
                else:
                    ty = t[3],t[2],t[1],t[0]
                return ty
            else:
                return NotImplemented
    
    @property                
    def runpath(self):
        r = self.storing_path
        for a in self.runpath_attributes:
            r += '/{:s}'.format(str(getattr(self,a)))
        return r
    

    def write_running_output(self):

        def type_var(v,ti,s):
            if ti is int: s+='{:d} '.format(v)
            elif ti is str: s+='{:s} '.format(v)
            elif ti is float: s+='{:8.6f} '.format(v)
            elif ti is bool:
                if v: s+='1 ' 
                else: s+='0 '
            return s

        def write(file,name,var):
            s = '{:15s}'.format(name)
            t = type(var)

            if t is list or t is tuple:
                try:
                    ti = type(var[0])
                except IndexError:
                    s+= ' : '
                else:
                    s +=' : '
                    for v in var:
                        s = type_var(v,ti,s)
            else:
                s+= ' = '
                s = type_var(var,t,s)
            s+='\n'
            
            file.write(s)
            return
        
        fname = '{:s}/runned.in'.format(self.runpath)
        with open(fname,'w') as file:
            add_empty_line = [6,12,27,31] 
            
            for i,(k,v) in enumerate(self.defaults.items()):
                var = getattr(self,k)
                if k in self.executes:
                    var = str(var)
                write(file,k,var)
                if i in add_empty_line: 
                    file.write('\n')
            file.write('\n')
            
            for k,model in self.opt_models.items() :
                file.write('&{:s}\n'.format(k))
                file.write('FUNC {:s}\n'.format(model.model))
                for k,p in model.pinfo.items():
                    file.write('{:10s} : {:8.8f}  {:d}  {:8.8f}  {:8.8f}    {:8.8f} \n'.format(
                                    p.name, p.value, int(p.opt),p.low_bound, p.upper_bound,p.regul))
                
                file.write('/\n\n')
                
            file.closed
        return
                    
    def __repr__(self):
        x = 'Attribute : value \n--------------------\n'
        for k,v in self.__dict__.items():
            x+='{} : {} \n'.format(k,v)
        x+='--------------------\n'
        return x

    def write_BOtable(self,typemap,Nr,which='opt'):
        path = 'lammps_working'
        lines = []

        try:
            models = getattr(self,which+'_models')
        except AttributeError as e:
            raise e
        line1 = "# DATE: 2024-03-20  UNITS: real  CONTRIBUTOR: Nikolaos Patsalidis,"
        line2 = "# This table was generated to describe flexible pairwise potentials for bond interactions by Nikolaos Patsalidis FF_Develop AL library\n"
        lines.append(line1) ; lines.append(line2)
        # 1 gather models of same type together
        models_per_type = {m.type: [m1 for m1 in models.values() if m1.type == m.type]  for k,m in models.items() if m.category =='BO' }
        rmin = 0.7 ; rmax = 3
        dr = (rmax -rmin)/Nr
        r = np.array([rmin+dr*i for i in range(Nr)])
        for ty, models_of_ty in models_per_type.items(): 
            keyword = '-'.join(ty)
            lines.append(keyword)
            lines.append('N {:d}  \n'.format(Nr))
            u = np.zeros(r.shape[0],dtype=float)
            for model in models_of_ty:
                fu = model.function
                u += fu(r,model.parameters,*model.model_args)

            du = np.empty_like(u)
            du[0] = -(u[1]-u[0])/dr
            du[1:-1] = -(u[2:]-u[:-2])/(2*dr)
            du[-1] = -(u[-1]-u[-2])/dr
            for i in range(Nr):
                s = '{:d} {:.16e} {:.16e} {:.16e}'.format(i+1,r[i],u[i],du[i])
                lines.append(s)
            lines.append(' ')
        with open('{:s}/tableBO.tab'.format(path),'w') as f:
            for line in lines:
                f.write('{:s}\n'.format(line))
            f.closed
        return

    def write_ANtable(self,typemap,Nr,which='opt'):
        path = 'lammps_working'
        lines = []

        try:
            models = getattr(self,which+'_models')
        except AttributeError as e:
            raise e
        line1 = "# DATE: 2024-03-20  UNITS: real  CONTRIBUTOR: Nikolaos Patsalidis,"
        line2 = "# This table was generated to describe flexible pairwise potentials for bond interactions by Nikolaos Patsalidis FF_Develop AL library\n"
        lines.append(line1) ; lines.append(line2)
        # 1 gather models of same type together
        models_per_type = {m.type: [m1 for m1 in models.values() if m1.type == m.type]  for k,m in models.items() if m.category =='AN' }
        rmin = 0.0 ; rmax = np.pi
        dr = (rmax -rmin)/(Nr-1)
        r = np.array([rmin+dr*i for i in range(Nr)])
        for ty, models_of_ty in models_per_type.items(): 
            keyword = '-'.join(ty)
            lines.append(keyword)
            lines.append('N {:d} \n'.format(Nr,))
            u = np.zeros(r.shape[0],dtype=float)
            for model in models_of_ty:
                fu = model.function
                u += fu(r,model.parameters,*model.model_args)

            du = np.empty_like(u)
            du[0] = -(u[1]-u[0])/dr
            du[1:-1] = -(u[2:]-u[:-2])/(2*dr)
            du[-1] = -(u[-1]-u[-2])/dr
            c = 180/np.pi
            for i in range(Nr):
                s = '{:d} {:.16e} {:.16e} {:.16e}'.format(i+1,r[i]*c,u[i],du[i]/c)
                lines.append(s)
            lines.append(' ')
        with open('{:s}/tableAN.tab'.format(path),'w') as f:
            for line in lines:
                f.write('{:s}\n'.format(line))
            f.closed
        return

    
    def write_PWtable(self,typemap,Nr,which='opt'):
        path = 'lammps_working'
        lines = []

        try:
            models = getattr(self,which+'_models')
        except AttributeError as e:
            raise e
        line1 = "# DATE: 2024-03-20  UNITS: real  CONTRIBUTOR: Nikolaos Patsalidis"
        line2 = "# This table was generated to describe flexible pairwise potentials between atoms by Nikolaos Patsalidis FF_Develop AL library\n"
        lines.append(line1) ; lines.append(line2)
        for k,model in models.items():
            if model.category != 'PW' or model.model =='Morse': 
                continue
            ty = model.type
            keyword = '-'.join(ty)
            lines.append(keyword)
            rmax = model.pinfo['L'].value
            rmin = 1e-10
            dr = (rmax -rmin)/(Nr-1)
            r = np.array ( [ rmin + dr*i for i in range(Nr) ] )
            lines.append('N {:d} R {:.16e} {:.16e}\n'.format(Nr,r[0],r[-1]))

            fu = model.function
            
            u = fu(r,model.parameters,*model.model_args)
            du = np.empty_like(u)
            du[0] = -(u[1]-u[0])/dr
            du[1:-1] = -(u[2:]-u[:-2])/(2*dr)
            du[-1] = -(u[-1]-u[-2])/dr
            for i in range(Nr):
                s = '{:d} {:.16e} {:.16e} {:.16e}'.format(i+1,r[i],u[i],du[i])
                lines.append(s)
            lines.append(' ')
        with open('{:s}/tablePW.tab'.format(path),'w') as f:
            for line in lines:
                f.write('{:s}\n'.format(line))
            f.closed
        return

    def write_LDtable(self,typemap,N_rho,which='opt'):
        N_LD = 0 
        
        
        lc1 = '     # lower and upper cutoffs, single space separated'
        lc2 = '     # central atom types, single space separated'
        lc3 = '     # neighbor-types (neighbor atom types single space separated)'
        lc4 = '     # min, max and diff. in tabulated rho values, single space separated'
        
        path = 'lammps_working'
        lines = []

        try:
            models = getattr(self,which+'_models')
        except AttributeError as e:
            raise e
       
        for k,model in models.items():
            if model.category != 'LD': 
                continue
            if model.num >= self.nLD: 
                continue
            ty = model.type
            if ty[0] not in typemap or ty[1] not in typemap:
                continue
            #if 'LD2' in model.name: continue
            #ni = model.num
            r0 = self.rho_r0
            rc = self.rho_rc
            fmodel = model.function
            #num_pars = model.number_of_parameters

            x  =  model.parameters
            
            N_LD+=1
            lines.append('{:4.9f} {:4.9f}  {:s}\n'.format(r0,rc,lc1))
            tyc = typemap[ty[0]]
            tynei = typemap[ty[1]]
            lines.append('{:d}  {:s}\n'.format(tyc,lc2))
            lines.append('{:d}  {:s}\n'.format(tynei,lc3))
           #print('saving LD type {} on the same file'.format(ty))
            rho_max=model.pinfo['L'].value ; rho_min = 0
            diff = (rho_max-rho_min)/(N_rho-1)
            rhov = np.array([rho_min +i*diff for i in range(N_rho)])
            if rhov.shape[0] != N_rho:
                raise Exception('model = {:s} not prober discretization rho_shape = {:d} != N_rho = {:d}'.format(model.name,rhov.shape[0],N_rho))
            #lines.append('{:.8e} {:.8e} {:.8e} {:s}\n'.format(rho_min,rho_max+diff,diff,lc4))
            lines.append('{:.16e} {:.16e} {:.16e} {:s}\n'.format(rho_min,rho_max,diff,lc4))
            Frho = fmodel(rhov,x,*model.model_args)
            for i,fr in enumerate(Frho):
                #lines.append('{:.8e}\n'.format(fr))
                lines.append('{:.16e}\n'.format(fr))
            lines.append('\n')
        #del lines[-1]  
        with open(path +'/frhos.ld','w') as f:
            f.write('{0:s}\n{0:s}Written by Force-Field Develop{0:s}\n{1:d} {2:d} # of LD potentials and # of tabulated values, single space separated\n\n'.format('******',N_LD,N_rho))
            for line in lines:
                f.write(line)
            f.closed
        return
    

class Interactions():
    
    def __init__(self,data,setup,atom_model = 'AA',
            vdw_bond_dist=4, find_vdw_unconnected = True,
            find_bonds=False, find_vdw_connected=True,
            find_dihedrals=False,find_angles=True,find_densities=False,
            excludedBondtypes=[],**kwargs):
        self.setup = setup
        self.data = data
        self.atom_model = atom_model
        self.vdw_bond_dist = vdw_bond_dist
        self.find_bonds = find_bonds
        self.find_vdw_unconnected = find_vdw_unconnected
        self.find_vdw_connected = find_vdw_connected
        self.find_angles = find_angles
        self.find_dihedrals = find_dihedrals
        self.find_densities = find_densities
        if find_densities:
            for rh in ['rho_r0','rho_rc']:
                if rh  not in kwargs:
                    raise Exception('{:s} is not given'.format(rh))
            if kwargs['rho_r0'] >= kwargs['rho_rc']:
                raise Exception('rho_rc must be greater than rho_r0')
        
        for i in range(len(excludedBondtypes)):
            excludedBondtypes.append((excludedBondtypes[i][1],excludedBondtypes[i][0]))
        
        self.excludedBondtypes = excludedBondtypes
        for k,v in kwargs.items():
            setattr(self,k,v)
        return
    
    @staticmethod
    def bonds_to_python(Bonds):
        if len(Bonds) == 0:
            return np.array(Bonds)
        bonds = np.array(Bonds,dtype=int)
        min_id = bonds[:,0:2].min()
        logger.debug('min_id = {:d}'.format(min_id))
        return bonds
    
    @staticmethod
    def get_connectivity(bonds,types,excludedBondtypes):
        conn = dict()     
        for b in bonds:
            i,t = Interactions.sorted_id_and_type(types,(b[0],b[1]))
            if t in excludedBondtypes:
                continue
            conn[i] = t
        return conn
    
    @staticmethod
    def get_angles(connectivity,neibs,types):
        '''
        Computes the angles of a system in dictionary format
        key: (atom_id1,atom_id2,atom_id3)
        value: object Angle
        Method:
            We search the neihbours of bonded atoms.
            If another atom is bonded to one of them an angle is formed
        We add in the angle the atoms that participate
        '''
        t0 = perf_counter()
        angles = dict()
        for k in connectivity.keys():
            #"left" side angles k[0]
            for neib in neibs[k[0]]:
                if neib in k: continue
                ang_id ,ang_type = Interactions.sorted_id_and_type(types,(neib,k[0],k[1]))
                if ang_id[::-1] not in angles.keys():
                    angles[ang_id] = ang_type
            #"right" side angles k[1]
            for neib in neibs[k[1]]:
                if neib in k: continue
                ang_id ,ang_type = Interactions.sorted_id_and_type(types,(k[0],k[1],neib))
                if ang_id[::-1] not in angles.keys():
                    angles[ang_id] = ang_type  
        tf = perf_counter()
        logger.info('angles time --> {:.3e} sec'.format(tf-t0))
        return angles
    
    @staticmethod
    def get_dihedrals(angles,neibs,types):
        '''
        Computes dihedrals of a system based on angles in dictionary
        key: (atom_id1,atom_id2,atom_id3,atom_id4)
        value: object Dihedral
        Method:
            We search the neihbours of atoms at the edjes of Angles.
            If another atom is bonded to one of them a Dihedral is formed is formed
        We add in the angle the atoms that participate
        '''
        t0 = perf_counter()
        dihedrals=dict()
        for k in angles.keys():
            #"left" side dihedrals k[0]
            for neib in neibs[k[0]]:
                if neib in k: continue
                dih_id,dih_type = Interactions.sorted_id_and_type(types,(neib,k[0],k[1],k[2]))
                if dih_id[::-1] not in dihedrals:
                    dihedrals[dih_id] = dih_type
            #"right" side dihedrals k[2]
            for neib in neibs[k[2]]:
                if neib in k: continue
                dih_id,dih_type = Interactions.sorted_id_and_type(types,(k[0],k[1],k[2],neib))
                if dih_id[::-1] not in dihedrals:
                    dihedrals[dih_id] = dih_type
        tf = perf_counter()
        logger.debug('dihedrals time --> {:.3e} sec'.format(tf-t0))
        return dihedrals
    
    @staticmethod
    def sorted_id_and_type(types,a_id):
        t = [types[i] for i in a_id]
        if t[0] <= t[-1]:
            t = tuple(t)
        else:
            t = tuple(t[::-1])
        if a_id[0]<=a_id[-1]:
            a_id = tuple(a_id)
        else:
            a_id = tuple(a_id[::-1])
        return a_id,t
    
    @staticmethod
    def get_neibs(connectivity,natoms):
        '''
        Computes first (bonded) neihbours of a system in dictionary format
        key: atom_id
        value: set of neihbours
        '''
        neibs = dict()
        if type(connectivity) is dict:
            for k in connectivity.keys(): 
                neibs[k[0]] = set() # initializing set of neibs
                neibs[k[1]] = set()
            for j in connectivity.keys():
                neibs[j[0]].add(j[1])
                neibs[j[1]].add(j[0])
        else:
            for i in range(connectivity.shape[0]):
                k1 =connectivity[i,0]
                k2 = connectivity[i,1]
                neibs[k1] = set()
                neibs[k2] = set()
            for i in range(connectivity.shape[0]):
                k1 =connectivity[i,0]
                k2 = connectivity[i,1]
                neibs[k1].add(k2)
                neibs[k2].add(k1)
        for ii in range(natoms):
            if ii not in neibs:
                neibs[ii] = set()
            
        return neibs
    
    @staticmethod
    def get_unconnected_structures(neibs):
        '''
        The name could be "find_unbonded_structures"
        Computes an array of sets. The sets contain the ids of a single molecule.
        Each entry corresponds to a molecule
        '''
        unconnected_structures=[]
        for k,v in neibs.items():
            #Check if that atom is already calculated
            in_unstr =False
            for us in unconnected_structures:
                if k in us:
                    in_unstr=True
            if in_unstr:
                continue
            sold = set()
            s = v.copy()  #initialize the set
            while sold != s:
                sold = s.copy()
                for neib in sold:
                    s = s | neibs[neib]
            s.add(k)
            unconnected_structures.append(s)
        try:
            sort_arr = np.empty(len(unconnected_structures),dtype=int)
            for i,unst in enumerate(unconnected_structures):
                sort_arr[i] =min(unst)
            x = sort_arr.argsort()
            uncs_struct = np.array(unconnected_structures)[x]
        except ValueError as e:
            logger.error('{}'.format(e))
            uncs_struct = np.array(unconnected_structures)
        return uncs_struct
    
    @staticmethod
    def get_at_types(at_types,bonds):
        at_types = at_types.copy()
        n = len(at_types)
        db = np.zeros(n,dtype=int)
        for b in bonds:
            if b[2] ==2:
                db [b[0]] +=1 ; db [b[1]] +=1
        for i in range(n):
            if db[i] == 0 :
                pass
            elif db[i] ==1:
                at_types[i] +='='
            elif db[i]==2:
                at_types[i] = '='+at_types[i]+'='
            else:
                logger.debug('for i = {:d} db = {:d} . This is not supported'.format(i,db[i]))
                raise Exception(NotImplemented)
                
        return at_types
    
    @staticmethod
    def get_united_types(at_types,neibs):
        united_types = np.empty(len(at_types),dtype=object)
        for i,t in enumerate(at_types):
           #print(i,t)
            nh=0
            if i in neibs:
                for neib in neibs[i]:
                    if at_types[neib].strip()=='H':nh+=1
                
            if  nh==0: united_types[i] = t
            elif nh==1: united_types[i] = t+'H'
            elif nh>1 : united_types[i] = t+'H'+str(nh)
        
        return united_types
    
    @staticmethod
    def get_itypes(model,at_types,bonds,neibs):
        logger.debug('atom model = {:s}'.format(model))
        if model.lower() in ['ua','united-atom','united_atom']:
            itypes = Interactions.get_united_types(at_types,neibs)
        elif model.lower() in ['aa','all-atom','all_atom']:
            itypes = Interactions.get_at_types(at_types, bonds)
        else:
            logger.debug('Model "{}" is not implemented'.format(model))
            raise Exception(NotImplementedError)
        
        return itypes

    @staticmethod
    def get_vdw_unconnected(types,bond_d_matrix):
        vdw_uncon = dict()
        n = bond_d_matrix.shape[0]
        for i1 in range(n):
            for i2 in range(n):
                if i2 <= i1:
                    continue
                if bond_d_matrix[i1,i2] == -1:
                    vid, ty = Interactions.sorted_id_and_type(types,(i1,i2) )
                    vdw_uncon[vid] = ty
        return vdw_uncon
    
    @staticmethod
    def get_vdw_connected(types,bond_d_matrix,vdw_bond_dist):
        vdw_con = dict()
        n = bond_d_matrix.shape[0]
        for i1 in range(n):
            for i2 in range(n):
                if i2 <= i1:
                    continue
                bd = bond_d_matrix[i1,i2]
                if bd > vdw_bond_dist:
                    vid, ty = Interactions.sorted_id_and_type(types,(i1,i2) )
                    vdw_con[vid] = ty
        return vdw_con
    
    
    @staticmethod
    def get_vdw(types,bond_d_matrix,find_vdw_connected,
                 find_vdw_unconnected,
                 vdw_bond_dist=4):
        if find_vdw_unconnected:
            vdw_unc = Interactions.get_vdw_unconnected(types,bond_d_matrix)
        else:
            vdw_unc = dict()
        if find_vdw_connected:
            vdw_conn = Interactions.get_vdw_connected(types,bond_d_matrix,vdw_bond_dist)
        else:
            vdw_conn = dict()
                
        vdw = dict()
        for d in [vdw_conn,vdw_unc]:
            vdw.update(d)
        return vdw
    
    @staticmethod
    def inverse_dictToArraykeys(diction):
        arr =  list( np.unique( list(diction.values()) , axis =0) )
       
        inv = {tuple(k) : [] for k in arr }
        for k,val in diction.items():
            inv[val].append(k)
        return {k:np.array(v) for k,v in inv.items()}
  
    @staticmethod
    def clean_hydro_inters(inters):
        #Remove Hydrogen atoms if you have to
        def del_inters_with_H(it):
            dkeys =[]
            for k in it.keys():
                if 'H' in k: dkeys.append(k)
            for k in dkeys:
                del it[k]
            return it
        for k,vv in inters.copy().items():
            inters[k] = del_inters_with_H(vv)
        
        return inters
    
    
    @staticmethod
    def get_rho_pairs(bdmatrix, at_types,vd):
        s = bdmatrix.shape
        rhos = dict()
        for i in range(s[0]):
            ty1 = at_types[i]
            for j in range(s[1]):
                if i==j:
                    continue
                #b = bdmatrix[i,j]
                ty2 = at_types[j]
                ty = (ty1,ty2)
                if ty not in rhos:
                    rhos[ty] = []
                rhos[ty].append([i,j])
            
        rhos = {k:np.array(v) for k,v in rhos.items()}
        rhos = {k: [v[v[:,0]==i] for i in np.unique(v[:,0])] for k,v in rhos.items()}
        return rhos
    
    def find_bond_distance_matrix(self,ids):
        '''
        takes an array of atom ids and finds how many bonds 
        are between each atom id with the rest on the array

        Parameters
        ----------
        ids : numpy array of int

        Returns
        -------
        distmatrix : numpy array shape = (ids.shape[0],ids.shape[0])
        
        '''
        
        size = ids.shape[0]
        distmatrix = np.zeros((size,size),dtype=int)
        for j1,i1 in enumerate(ids):
            nbonds = self.bond_distance_id_to_ids(i1,ids)
            distmatrix[j1,:] = nbonds
        return distmatrix
    
    def bond_distance_id_to_ids(self,i,ids):
        '''
        takes an atom id and find the number of bonds between it 
        and the rest of ids. 
        If it is not connected the returns a very
        large number
        
        Parameters
        ----------
        i : int
        ids : array of int

        Returns
        -------
        nbonds : int array of same shape as ids
            number of bonds of atom i and atoms ids

        '''
        chunk = {i}
        n = ids.shape[0]
        nbonds = np.ones(n)*(-1)
        incr_bonds = 0
        new_neibs = np.array(list(chunk))
        while new_neibs.shape[0]!=0:
            f = np.zeros(ids.shape[0],dtype=bool)
            numba_isin(ids,new_neibs,f)
            nbonds[f] = incr_bonds
            new_set = set()
            for ii in new_neibs:
                for neib in self.neibs[ii]:
                    if neib not in chunk:
                        new_set.add(neib)
                        chunk.add(neib)
            new_neibs = np.array(list(new_set))
            incr_bonds+=1
        return nbonds

    def find_configuration_inters(self,Bonds,atom_types,struct_types):
        Bonds = Interactions.bonds_to_python(Bonds)
        natoms = len(atom_types)
        neibs = Interactions.get_neibs(Bonds,natoms)
        #at_types = Interactions.get_at_types(atom_types.copy(),Bonds)
        at_types=atom_types
        connectivity = Interactions.get_connectivity(Bonds,at_types,self.excludedBondtypes)
        
        neibs = Interactions.get_neibs(connectivity,natoms)

        self.neibs =  neibs 
        bond_d_matrix = self.find_bond_distance_matrix( np.arange(0,natoms,1,dtype=int) ) 
        
        bodies = Interactions.get_unconnected_structures(neibs)
        bodies = {j:np.array(list(v),dtype=int) for j,v in enumerate(bodies) }
        structs = dict()
        for k in  struct_types:
            stk = np.array([j  for j,t in enumerate(at_types) if t in k],dtype=int)
            if stk.shape[0] > 0:
                structs[k] = stk


        types = Interactions.get_itypes(self.atom_model,at_types,Bonds,neibs)
        types = Interactions.get_at_types(types.copy(),Bonds)
        
        #1.) Find 2 pair-non bonded interactions
        vdw = Interactions.get_vdw(types, bond_d_matrix, self.find_vdw_connected,
                      self.find_vdw_unconnected,
                      self.vdw_bond_dist)      

        if self.find_angles or self.find_dihedrals:
            angles = Interactions.get_angles(connectivity,neibs,types)
            
            
            if self.find_dihedrals:
                dihedrals = Interactions.get_dihedrals(angles,neibs,types)
    
            else:
                dihedrals = dict()
        else:
            angles = dict()
            dihedrals= dict()
            

        inters = {k : Interactions.inverse_dictToArraykeys(d) 
                  for k,d in zip(['connectivity','angles','dihedrals','vdw'],
                              [connectivity, angles, dihedrals, vdw])
                 }
        if self.find_densities:
           # print(at_types)
            rhos = Interactions.get_rho_pairs(bond_d_matrix,at_types,self.vdw_bond_dist)
        else:
            rhos = dict()
        inters['rhos'] = rhos
        if self.atom_model.lower() in ['ua','united-atom','united_atom']:    
            inters = Interactions.clean_hydro_inters(inters)
            
        return inters, bodies, structs
     
    def InteractionsForData(self,setup):
        #t0 = perf_counter()
        dataframe = self.data
        
        All_inters = np.empty(len(dataframe),dtype=object)
        all_bodies = np.empty(len(dataframe),dtype=object)
        all_structs = np.empty(len(dataframe),dtype=object)
        first_index = dataframe.index[0]
        Bonds = dataframe['Bonds'][first_index].copy()
        atom_types = dataframe['at_type'][first_index].copy()
        inters, bodies, structs = self.find_configuration_inters(Bonds,atom_types,setup.struct_types)
        for i,(j,data) in enumerate(dataframe.iterrows()):
            b_bool =dataframe.loc[j,'Bonds'] != Bonds
            t_bool = dataframe.loc[j,'at_type'] != atom_types
            if b_bool or t_bool:
                Bonds = data['Bonds'].copy()
                atom_types = data['at_type'].copy()
                inters, bodies, structs = self.find_configuration_inters(Bonds,atom_types,setup.struct_types)
                logger.info('Encountered different Bonds or at_type on j = {}\n Different bonds --> {}  Different types --> {}'.format(j,b_bool,t_bool))
            All_inters[i] = inters
            all_bodies[i] = bodies   
            all_structs[i] = structs
        #print(j,all_bodies)
        dataframe['interactions'] = All_inters
        dataframe['bodies'] = all_bodies
        dataframe['structs'] = all_structs
        return
    
    @staticmethod
    def activation_function_illustration(r0,rc):
        def get_rphi(r0,rc):
            c = Interactions.compute_coeff(r0, rc)
            r = np.arange(0,rc*1.04,0.001)
            phi = np.array([Interactions.phi_rho(x,c,r0,rc) for x in r])
            return r,phi
        
        size = 3.0
        _ = plt.figure(figsize=(size,size),dpi=300)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size)
        plt.tick_params(direction='in', which='major',length=size*2)
        plt.ylabel(r'$\phi(r)$')
        plt.xlabel(r'$r$ $(\AA)$')
        plt.xticks([i for i in range(int(max(rc))+1)])
        colors = ['#1b9e77','#d95f02','#7570b3']
        plt.title('Illustration of the activation function',fontsize=3.0*size)
        styles =['-','-','-']
        if type(r0) is tuple or type(r0) is list:
            for i,(rf0,rfc) in enumerate(zip(r0,rc)):
                r,phi = get_rphi(rf0,rfc)
                label = r'$r_c$={:1.1f} $\AA$'.format(rfc)
                plt.plot(r,phi,label=label,color=colors[i%3],lw=size/2.0,ls=styles[i%3])
            plt.legend(frameon=False,fontsize=2.5*size)
        else:
            r,phi = get_rphi(r0,rc)
            plt.plot(r,phi)
        plt.savefig('activation.png',bbox_inches='tight')
        #plt.show()
    @staticmethod
    def compute_coeff(r0,rc):
        c = np.empty(4,dtype=float)
        r2 = (r0/rc)**2

        r3 = (1.0 -r2)**3

        c[0] = (1.0-3.0*r2)/r3
        c[1] =(6.0*r2)/r3/rc**2
        c[2] = -3.0*(1.0+r2)/r3/rc**4
        c[3] = 2.0/r3/rc**6
        return c
    
    @staticmethod
    def phi_rho(r,c,r0,rc):
        if r<=r0:
            return 1
        elif r>=rc:
            return 0
        else:    
            return c[0] + c[1]*r**2 + c[2]*r**4 + c[3]*r**6
        
    @staticmethod
    def dphi_rho(r,c,r0,rc):
        if r<=r0:
            return 0
        elif r>=rc:
            return 0
        else:
            return 2*c[1]*r + 4*c[2]*r**3 + 6*c[3]*r**5
    
    def calc_rhats(self):
        n = len(self.data)
        atom_confs = np.array(self.data['coords'])
        all_rhats = np.empty(n,dtype=object)
        for m in range(n):
            ac = np.array(atom_confs[m])
            
            natoms = ac.shape[0]
            rhats  = np.zeros( (natoms,natoms,3) , dtype=float)
            for i in range(natoms):
                xi = ac[i]
                for j in range(natoms):
                    if j==i: continue
                    xj = ac[j]
                    rh = VectorGeometry.calc_unitvec(xi,xj)
                    rhats[i,j] = rh
            all_rhats[m] = rhats
        self.data['rhats'] = all_rhats
        return
    
    def calc_neibs_lists(self):
        n = len(self.data)
        
        all_neibs = np.empty(n,dtype=object)

        interactions = np.array(self.data['interactions'])
        
        for m,  inters in zip(range(n), interactions):
            neibs = dict()
            for intertype, pairs_dict in inters.items():
                d = {t:dict() for t in pairs_dict}
                for t, pairs in pairs_dict.items():
                    if intertype in ['connectivity','vdw']:
                        for im, p in enumerate(pairs):
                            i , j = p
    
                            if i not in d[t]:
                                d[t][i] = [ j ]
                            else:
                                d[t][i].append(j)
                    elif intertype =='rhos':
                        for im,ltpa in enumerate(pairs):
                            for o,p in enumerate(ltpa):
                                i , j = p
                                if i not in d[t]:
                                    d[t][i] = [ j ]
                                else:
                                    d[t][i].append(j)
                    
                    else:
                        raise NotImplementedError
                neibs[intertype] = {t: {k: np.array(idxs,dtype=int) for k,idxs in v.items()} for t,v in d.items()}

                
            all_neibs[m] = neibs

        self.data['neibs'] = all_neibs       
        return
    
    def calc_interaction_values(self):
        n = len(self.data)
        
        all_Values = np.empty(n ,dtype=object)
        
        atom_confs = np.array(self.data['coords'])
        interactions = np.array(self.data['interactions'])
        
        for i, ac, inters in zip(range(n), atom_confs, interactions):
            
            values = dict(keys=inters.keys())
            
            
            for intertype,vals in inters.items():
                
                d =  {t : None for t in vals.keys()}
                
                for t,pairs in vals.items():
                    
                    temp = np.empty(len(pairs),dtype=float)
                    if intertype in ['connectivity','vdw']:
                        
                        for im,p in enumerate(pairs):
                            
                            r1 = np.array(ac[p[0]]) ; r2 = np.array( ac[p[1]])
                            
                            temp[im] =  VectorGeometry.calc_dist(r1,r2)
                    
                    elif intertype=='angles':

                        for im,p in enumerate(pairs):
                            
                           
                            
                            r1 = np.array(ac[p[0]]) ; 
                            r2 = np.array(ac[p[1]]) ; 
                            r3 = np.array(ac[p[2]])
                            
                            temp[im] =  VectorGeometry.calc_angle(r1,r2,r3)
                    
                    elif intertype =='dihedrals':
                        
                        for im,p in enumerate(pairs):
 
                            r1 = np.array(ac[p[0]]) ; 
                            r2 = np.array(ac[p[1]]) ; 
                            r3 = np.array(ac[p[2]])
                            r4 = np.array(ac[p[3]])
                            
                            temp[im] =  VectorGeometry.calc_dihedral(r1, r2, r3, r4)
                        
                    elif intertype=='rhos':
                        r0 = self.rho_r0
                        rc = self.rho_rc
                        
                        temp = np.empty(len(pairs),dtype=float)
    
                        c = self.compute_coeff(r0, rc)
                        for im,ltpa in enumerate(pairs):
                            rho = 0
                            for o,p in enumerate(ltpa):
                                        
                                r1 = np.array(ac[p[0]]) 
                                r2 = np.array( ac[p[1]])
                               
                                r12 =  VectorGeometry.calc_dist(r1,r2)
                                rho += self.phi_rho(r12,c,r0,rc)

                            temp[im] = rho
                            
                    else:
                        raise Exception(NotImplemented)
                    
                    d[t] = temp.copy()
                           
                values[intertype] = d.copy()

                
            all_Values[i] = values

        self.data['values'] = all_Values
        self.calc_neibs_lists()
        self.calc_rhats()
        self.calc_force_info()
        return
    
    def calc_force_info(self):
        
        n = len(self.data)
        
        all_force_info = np.empty(n ,dtype=object)
        
        atom_confs = np.array(self.data['coords'])
        interactions = np.array(self.data['interactions'])
        
        for m, ac, inters in zip(range(n), atom_confs, interactions):
            
            force_info = dict(keys=inters.keys())
            
            
            for intertype,vals in inters.items():
                
                d =  {t : None for t in vals.keys()}
                
                for t,pairs in vals.items():
                    
                   
                    if intertype in ['connectivity','vdw']:
                        
                        npairs = len(pairs) 
                        
                        r  = np.empty((npairs,1), dtype=float)
                        rhats_ij = np.empty( (npairs,3), dtype=float)
                        i_index = np.empty(npairs,dtype=int)
                        j_index = np.empty(npairs,dtype=int)
                                    
                        for ip,p in enumerate(pairs):
                            i, j = p
                            
                            r1 = np.array(ac[i]) 
                            r2 = np.array( ac[j])
                            
                            i_index[ip] = i
                            j_index[ip] = j
                            r[ip] =  VectorGeometry.calc_dist(r1,r2)
                            rhats_ij[ip] =  VectorGeometry.calc_unitvec(r1,r2)
                       
                        temp = {'r':r, 'rhats_ij':rhats_ij,
                                'i_index':i_index,'j_index':j_index}
                        
                    elif intertype=='angles':
                        npairs = len(pairs) 
                        
                        angles  = np.empty(npairs, dtype=float)
                        ra = np.empty( (npairs,3), dtype=float)
                        rc = np.empty( (npairs,3), dtype=float)
                        i_index = np.empty(npairs,dtype=int)
                        j_index = np.empty(npairs,dtype=int)
                        k_index = np.empty(npairs,dtype=int)
                        
                        for ip,p in enumerate(pairs):
                            i, j, k = p
                        
                            r1 = np.array(ac[i]) ; 
                            r2 = np.array(ac[j]) ; 
                            r3 = np.array(ac[k])
                            
                            i_index[ip] = i
                            j_index[ip] = j
                            k_index[ip] = k
                            
                            ra[ip] = VectorGeometry.calc_angle_ra(r1,r2,r3)
                            rc[ip] = VectorGeometry.calc_angle_rc(r1,r2,r3)
                            angles[ip] =  VectorGeometry.calc_angle(r1,r2,r3)
                        
                        temp = {'angles':angles, 'ra':ra, 'rc': rc,
                                'i_index':i_index, 'j_index':j_index,
                                'k_index': k_index}
                    
                    elif intertype =='dihedrals':
                        raise NotImplementedError('Code here is incomplete. Left for later stage')
                        npairs = len(pairs) 
                        
                        dihedrals  = np.empty(npairs, dtype=float)
                        ra = np.empty( (npairs,3), dtype=float)
                        rd = np.empty( (npairs,3), dtype=float)
                        
                        i_index = np.empty(npairs,dtype=int)
                        j_index = np.empty(npairs,dtype=int)
                        k_index = np.empty(npairs,dtype=int)
                        l_index = np.empty(npairs,dtype=int)
                        for ip,p in enumerate(pairs):
                            i, j, k, l = p
                            r1 = np.array(ac[i]) ; 
                            r2 = np.array(ac[j]) ; 
                            r3 = np.array(ac[k])
                            r4 = np.array(ac[l])
                            
                            i_index[ip] = i
                            j_index[ip] = j
                            k_index[ip] = k
                            l_index[ip] = l
                            
                            ra[ip] = VectorGeometry.calc_dihedral_ra(r1,r2,r3,r3)
                            rd[ip] = VectorGeometry.calc_dihedral_rd(r1,r2,r3,r3)
                            dihedrals[ip] =  VectorGeometry.calc_dihedral(r1, r2, r3, r4)
                            
                        temp = {'dihedral':dihedrals, 'ra':ra, 'rd': rd,
                                'i_index':i_index, 'j_index':j_index,
                                'k_index': k_index, 'l_index': l_index}
                        
                    elif intertype=='rhos':
                        r0 = self.rho_r0
                        rc = self.rho_rc
                        
                        n_rhos = len(pairs) 
                        
                        rhos  = np.empty(n_rhos, dtype=float)
                        
                        npairs = np.sum([len(v) for v in pairs])
                        
                        v_ij = np.empty( (npairs,3), dtype=float)
                        
                        i_index = np.empty(npairs,dtype=int)
                        j_index = np.empty(npairs,dtype=int)
                        to_pair_index = np.empty(npairs,dtype=int)
                        
                        c = self.compute_coeff(r0, rc)
                        
                        tot_iter = 0
                        for iv in range(n_rhos):
                            rho = 0
                            for ip,p in enumerate(pairs[iv]):
                                i,j  = p 
                                r1 = np.array(ac[i]) 
                                r2 = np.array(ac[j])
                                
                                i_index[tot_iter] = i
                                j_index[tot_iter] = j
                                to_pair_index[tot_iter] = iv
                                
                                r12 =  VectorGeometry.calc_dist(r1,r2)
                                dirvec = self.dphi_rho(r12,c,r0,rc) * VectorGeometry.calc_unitvec(r1,r2)
                                v_ij[tot_iter] = dirvec
                                
                                rho += self.phi_rho(r12,c,r0,rc)
                                
                                
                                tot_iter+=1
                            rhos[iv] = rho
                        temp = {'rhos':rhos, 'v_ij':v_ij,'to_pair_index':to_pair_index,
                                'i_index':i_index,'j_index':j_index}
                    else:
                        raise Exception(NotImplemented)
                    
                    d[t] = temp.copy()
                           
                force_info[intertype] = d.copy()
    
            all_force_info[m] = force_info
        self.data['forces_info'] = all_force_info 
        return

class Data_Manager():
    def __init__(self,data,setup):
        self.data = data
        self.setup = setup
        return
    def distribution(self,col):
        path = self.setup.runpath.split('/')[0]
        GeneralFunctions.make_dir(path)
        for s in self.data['sys_name'].unique():
            _ = plt.figure(figsize=(3.5,3.5),dpi=300)
            plt.minorticks_on()
            plt.tick_params(direction='in', which='minor',length=3)
            plt.tick_params(direction='in', which='major',length=5)
            plt.ylabel(col + ' distribution')
            f = self.data['sys_name'] == s
            plt.title('System {:s}'.format(s))
            plt.hist(self.data[col][f],bins=100,histtype='step',color='magenta')
            plt.savefig('{:s}/{:s}_{:s}_distribution.png'.format(path,s,col), bbox_inches='tight')
        #plt.show()   
        return
    
    @staticmethod
    def data_filter(data,selector=dict(),operator='and'):
        
        n = len(data)
        if selector ==dict():
            return np.ones(n,dtype=bool)
        
        if operator =='and':
            filt = np.ones(n,dtype=bool)
            oper = np.logical_and
        elif operator =='or':
            filt = np.zeros(n,dtype=bool)
            oper = np.logical_or
            
        for k,val in selector.items():
            if GeneralFunctions.iterable(val):
                f1 = np.zeros(n,dtype=bool)
                for v in val: f1 = np.logical_or(f1, data[k] == v)
            else:
                 f1 = data[k] == val
            filt = oper(filt,f1)
                
        return filt
    
    def select_data(self,selector=dict(),operator='and'):
        filt = self.data_filter(self.data,selector,operator)
        return self.data[filt]
    
    @staticmethod
    def generalized_data_filter(data,selector=dict()):
        n = len(data)
        filt = np.ones(n,dtype=bool)
        for k,val in selector.items():
            try:
                iter(val)
            except:
                s = 'val must be iterable, containing [operator or [operators], value or [values]]'
                logger.error(s)
                raise Exception(s)
            else:
                try:
                    iter(val[1])
                    iter(val[0])
                    f1 = np.zeros(n,dtype=bool)
                    for operator,v in zip(val[0],val[1]): 
                        f1 = np.logical_or(f1, operator(data[k],v))
                except:
                    f1 = val[0](data[k],val[1])
            filt = np.logical_and(filt,f1)
        return filt
    
    def clean_data(self,cleaner=dict()):
        # default are dumb values
        filt = self.generalized_data_filter(self.data,selector=cleaner)
        self.data = self.data[filt]
        return    
    
    @staticmethod
    def save_selected_data(fname,data,selector=dict(),labels=None):
        if labels is not  None:
            for j,pars in data.iterrows():
                comment = ' , '.join(['{:s}  = {:}'.format(lab, pars[lab]) for lab in labels])
                data.loc[j,'comment'] = 'index = {} , '.format(j) + comment
        dataT = data[Data_Manager.data_filter(data,selector)]
        with open(fname,'w') as f:
            for j,data in dataT.iterrows():
                at = data['at_type']
                ac = data['coords']
                na = data['natoms']
                try:
                    comment = data['comment']
                except:
                    comment = ''
                f.write('{:d} \n{:s}\n'.format(na,comment))
                for j in range(na):
                    f.write('{:3s} \t {:8.8f} \t {:8.8f} \t {:8.8f}  \n'.format(at[j],ac[j][0],ac[j][1],ac[j][2]) )
            f.closed
            
        return 
    
    @staticmethod
    def read_frames(filename):
        with open(filename,'r') as f:
            lines = f.readlines()
            f.closed
     
        line_confs =[]
        natoms = []
        comments = []
        for i,line in enumerate(lines):
            if 'iteration' in line:
                line_confs.append(i-1)
                comments.append(line.split('\n')[0])
                natoms.append(int(lines[i-1].split('\n')[0]))
        
        confs_at = []
        confs_coords = []
        for j,n in zip(line_confs,natoms):
           
            at_type = []
            coords = []
            for i in range(j+2,j+n+2):         
                li = lines[i].split('\n')[0].split()
                at_type.append( li[0] )
                coords.append(np.array(li[1:4],dtype=float))
            confs_at.append(at_type)
            confs_coords.append(np.array(coords))
       
        def ret_val(a,c):
            for j in range(len(c)):            
                if a == c[j]:
                    return c[j+2]
            return None
        
        energy = [] ; constraint = [] ; cinfo = [] ;maxf=[] ; con_val =[] 
        for comment in comments:
            c = comment.split()
            cnm = ret_val('constraint',c)
            cval = ret_val('constraint_val',c)
            cinfo.append(cnm+cval[0:5])
            constraint.append(cnm)
            con_val.append(float(cval))
            energy.append(float(ret_val('eads',c)))
            maxf.append(float(ret_val('maxforce',c)))
            
        #cons = np.unique(np.array(constraint,dtype=object))
    
        data  = {'Energy':energy,'cinfo':cinfo,'constraint':constraint,'constraint_val':con_val,'max_force':maxf,'natoms':natoms,
                 'at_type':confs_at,'coords':confs_coords,'comment':comments}
        dataframe =pd.DataFrame(data)
        #bonds
        

        return dataframe 
    
    @staticmethod
    def make_labels(dataframe):
        n = len(dataframe)
        attr_rep = np.empty(n,dtype=object)
        i=0
        for j,data in dataframe.iterrows():
            
            if 'rep' in data['cinfo']:
                attr_rep[i]='rep'
            else:
                attr_rep[i]='attr'
            i+=1 
            
        dataframe['label'] = 'inter'
        dataframe['attr_rep'] = attr_rep
        #labels
        for c in dataframe['cinfo'].unique():
            filt = dataframe['cinfo'] == c
            idxmin = dataframe['Energy'][filt].index[dataframe['Energy'][filt].argmin()]
            dataframe.loc[idxmin,'label'] = 'optimal'
        return
    
    @staticmethod
    def read_xyz(fname):
        with open(fname) as f:
            lines = f.readlines()
            f.closed
        return Data_Manager.lines_one_frame(lines)
    @staticmethod
    def lines_one_frame(lines):
        lines = [line.strip('\n') for line in lines]
        na = int(lines[0])
        
        natoms = [na]
        attrs = lines[1].split(',')
        atr = {k.split('=')[0].strip() : k.split('=')[1].strip() for k in attrs}
        newattr = dict()
        for k,v in atr.items():
            try:
                v1 = int(v)
            except:
                try:
                    v1 =float(v)
                except:
                    v1 = v
            newattr[k] = [v1]
        newattr['natoms'] = natoms
        at_types = []
        coords= []
        for line in lines[2:na+2]:
            l = line.split()
            at_types.append(l[0])
            coords.append(np.array(l[1:],dtype=float))
        newattr['coords'] = [coords]
        newattr['at_type'] = [at_types]
        
        return pd.DataFrame(newattr)
        
    
    def read_mol2(self,filename,read_ener=False,label=False):
        
        # first read all lines
        with open(filename, 'r') as f:	#open the text file for reading
            lines_list = f.readlines()
            f.closed
        logger.info("reading mol2 file: {}".format(filename))
        #Find starting point,configuration number, configuration distance
        nconfs = 0
        lns_mol = []; lns_atom = [] ; lns_bond = []
        for i,line in enumerate(lines_list):
            if line.split('\n')[0] == '@<TRIPOS>MOLECULE':
                 nconfs += 1
                 lns_mol.append(i)
            if line.split('\n')[0] == '@<TRIPOS>ATOM':
                 lns_atom.append(i)
            if line.split('\n')[0] == '@<TRIPOS>BOND':
                 lns_bond.append(i)
    
        logger.info(' Number of configurations = {:d}'.format(nconfs))
        if nconfs != len(lns_atom) or len(lns_bond) != len(lns_atom):
            logger.error('nconfs = {:d}, ncoords = {:d}, nbonds = {:d}'.format(nconfs,len(lns_atom),len(lns_bond)))
      
        natoms = 30000 ; nbonds = 50000; # default params
        Natoms = [] 
        Atom_Types =[] ; Bonds = [] ; Atom_coord = []
        if read_ener:
            energy = []
        #if label: labels = []
        
        for iconf,(im,ia,ib) in enumerate(zip(lns_mol,lns_atom,lns_bond)):
            #reading natoms,nbonds
            natoms = int (lines_list[im+2].split('\n')[0].split()[0])
            nbonds = int (lines_list[im+2].split('\n')[0].split()[1])
            at_type = []
            bonds = []
            at_coord = []
            #Reading type of atoms and coordinates
            for ja in range(0,natoms):
                line = lines_list[ia+1+ja].split('\n')[0].split()
                at_type.append( line[5].split('.')[0] )
                at_coord.append([float(line[2]), float(line[3]),float(line[4])])
            # Reading bonds
            for jb in range(0,nbonds):
                line = lines_list[ib+1+jb].split('\n')[0].split()
                bonds.append(line[1:4])
            Atom_Types.append(at_type)
            Atom_coord.append(at_coord)
            Bonds.append(bonds)
            Natoms.append(natoms)
            if read_ener:
                energy.append(float(lines_list[im+1].split('\n')[0].split()[6]))
            #.debug('conf {:d}: natoms = {:d}, nbonds = {:d}'.format(iconf,natoms,nbonds))
    
        data_dict ={'coords':Atom_coord,'at_type': Atom_Types,
                    'Bonds': Bonds,'natoms':Natoms}
        if read_ener:
            data_dict['Energy'] = energy
        return data_dict
    
    @staticmethod
    def read_Gaussian_output(filename,add_zeroPoint_corr=True,
                             units='kcal/mol',
                             clean_maxForce=None,
                             minEnergy=True,
                             enerTol=1e-6,
                             read_forces=False):
        import numpy as np
        import pandas as pd
        # first read all lines
        with open(filename, 'r') as f:	#open the text file for reading
            lines_list = f.readlines()
            f.closed
        logger.debug('reading filename "{:s}"'.format(filename))
        lc = [] ; 
        
        def key_in_line(key,lin):
            x = True
            if GeneralFunctions.iterable(key):
                for ek in key:
                    if ek not in lin:
                        x=False
                        break
            else:
                x = key in lin
            return x
        
        leners = []
        forcelines = []
        for i,line in enumerate(lines_list):
            lin = line.split()
            
            if key_in_line(['Standard','orientation:'],lin): 
                lc.append(i+5)
            if key_in_line(['SCF','Done:','='],lin):
                if ' >>>>>>>>>> Convergence criterion not met.' in lines_list[i-1]:
                    approx_ener_line = i
                    continue                
                leners.append(i)
                
            if key_in_line(['Atomic','Forces','(Hartrees/Bohr)'],lin):
                forcelines.append(i+3)

        #Count number of atoms in file
        if len(leners) ==0:
            try:
                leners.append(approx_ener_line)
            except UnboundLocalError:
                pass
        natoms = 0
        at_type = []
        for line in lines_list[lc[0]::]:
            if line[5:7]=='--':
                break
            at_type.append(line.split()[1])
            natoms+=1
            
        
        '''
        striped_lines = [line.strip('\n ') for line in lines_list]
        eners = ''.join(striped_lines).split('HF=')[-1].split('\\')[0].split(',')
        eners = np.array(eners,dtype=float)
        '''
        striped_lines = [line.strip('\n ') for line in lines_list]
        eners = np.array([float(lines_list[i].split('=')[-1].split()[0])   for i in leners])
        if add_zeroPoint_corr:
            st = 'ZeroPoint='
            if st in ''.join(striped_lines):
                zero_point_corr = ''.join(striped_lines).split(st)[-1].split('\\')[0].split(',')
                eners-= np.array(zero_point_corr,dtype=float)
        
        
        
        for line in striped_lines:
            if 'Stoichiometry' in line:
                sysname = line.split()[-1]
        
        #drop last line
        logger.debug('found eners = {:d} ,found confs = {:d}'.format(eners.shape[0],len(lc)))

       
        if eners.shape[0] +1 == len(lc):
            lc = lc[:-1]
        
        if units=='kcal/mol':
            eners *= 627.5096080305927 # to kcal/mol
        Atypes = mappers().atomic_num
        for j in range(len(at_type)):
            at_type[j] = Atypes[at_type[j]]
        #Number of configurations
        nconfs = len(lc)
        logger.info('Reading file: {:} --> nconfs = {:d} ,ener values = {:d},\
                    natoms = {:d}'.format(filename,nconfs,len(eners),natoms))
        
        #Fill atomic configuration 
        config = []
        for j,i in enumerate(lc):
            coords = []
            for k,line in enumerate(lines_list[i:i+natoms]):
                li = line.split()
                coords.append(np.array(li[3:6],dtype=float))
            config.append( np.array(coords) )
        if clean_maxForce is not None:
        #if clean_maxForce is not None and eners.shape[0]>1:
            forces_max = []
            for j,i in enumerate(forcelines):
                forces= []
                for k,line in enumerate(lines_list[i:i+natoms]):
                    li = line.split()
                    forces.append(np.array(li[3:6],dtype=float))
                m = np.abs(forces).max()
                if units =='kcal/mol': m*=627.5096080305927
                forces_max.append(m)
            forceFilt = np.array(forces_max) < clean_maxForce
            
            logger.debug('ForceFilt shape = {}'.format(forceFilt.shape))
            config = [config[i] for i,b in enumerate(forceFilt) if b]
            eners = eners[forceFilt]
            
            nconfs = len(config)
            logger.info('NEW AFTER CLEANING nconfs = {:d} ,ener values = {:d},\
                    natoms = {:d}'.format(nconfs,len(eners),natoms))
        if nconfs ==0 :
            
            raise ValueError("File {:s} does not contain any configuration. Probably it didn't run for enough time".format(filename))

        if minEnergy:
            me = eners.argmin()
            if len(config) == len(eners):
                config = [config[me]] # list
            eners = eners[[me]] #numpy array
            nconfs = eners.shape[0]
        name = filename.split('/')[-1].split('.')[0]
        data_dict ={'coords':config,'natoms':[int(natoms)]*nconfs,
                    'at_type':[at_type]*nconfs,'sys_name':[sysname]*nconfs,
                    'filename':[filename]*nconfs,'name':[name]*nconfs}
        if read_forces:
            fdat = []
            for j,i in enumerate(forcelines):
                forces= []
                for k,line in enumerate(lines_list[i:i+natoms]):
                    li = line.split()
                    forces.append(np.array(li[3:6],dtype=float))
                forces = np.array(forces)
                if units =='kcal/mol': forces*=627.5096080305927
                fdat.append(np.array(forces))
            #print (data_dict)
            data_dict['Forces'] = fdat
            #print(data_dict)
        data_dict['Energy'] = eners
        #print(data_dict)

        data = pd.DataFrame(data_dict)
        
        return data
    

    def assign_system(self,sys_name):
        self.data['sys_name'] = sys_name
        return
    
    def create_dist_matrix(self):
        logger.info('Calculating distance matrix for all configurations \n This is a heavy calculation consider pickling your data')
        i=0
        dataframe = self.data
        size = len(dataframe)
        All_dist_m = np.empty(size,dtype=object)
        at_conf = dataframe['coords'].to_numpy()
        nas = dataframe['natoms'].to_numpy()
        for i in range(size):
            natoms = nas[i]
            conf = at_conf[i]
            dist_matrix = np.empty((natoms,natoms))
            for m in range(0,natoms):
                for n in range(m,natoms):
                    r1 = np.array(conf[n])
                    r2 = np.array(conf[m])
                    rr = VectorGeometry.calc_dist(r1,r2)
                    dist_matrix[m,n] = rr
                    dist_matrix[n,m] = rr
            All_dist_m[i] = dist_matrix
            if i%1000 == 0:
                logger.info('{}/{} completed'.format(i,size))
        
        dataframe['dist_matrix'] = All_dist_m
        return  
    
    
    def sample_randomly(self,perc,data=None,seed=None):
        if data is None:
            data = self.data
        if not perc>0 and not perc <1:
            raise Exception('give values between (0,1)')  
        size = int(len(data)*perc)
        if seed  is not None: np.random.seed(seed)
        indexes = np.random.choice(data.index,size=size,replace=False)
        return data.loc[indexes]
    
    def train_development_split(self):
        
        data = self.data
        
        if len(data) < 20:
            i = data.index
            return i,i
        train_perc = self.setup.train_perc
        seed = self.setup.seed
        sampling_method = self.setup.sampling_method
        
        def get_data_via_column():
            vinfo = self.setup.devolopment_set
            
            ndata = len(data)
            f = np.ones(ndata,dtype=bool)
            
            for colval in vinfo.split('&'):
                temp =  colval.split(':')
                col = temp[0]
                vals = temp[1]
                
                fv = np.zeros(ndata,dtype=bool)
                for v in vals.split(','):
                    
                    fv = np.logical_or(fv,data[col]==v.strip())
                f = np.logical_and(fv,f)
            return f
        
        if sampling_method=='random':
            train_data = self.sample_randomly(train_perc, data, seed)
            train_indexes = train_data.index
            
            dev_data = data.loc[data.index.difference(train_indexes)]
            
        
        elif sampling_method=='column':
            
            f = get_data_via_column()
            
            dev_data = data[f]
        else:
            raise Exception(NotImplementedError('"{}" is not a choice'.format(sampling_method) ))
        
        self.train_indexes = train_indexes
        dev_indexes = dev_data.index
        self.dev_indexes = dev_indexes
        
        return train_indexes, dev_indexes
    
    def bootstrap_samples(self,nsamples,perc=0.3,seed=None,
                          sampling_method='random',nbins=100,bin_pop=1):
        if seed is not None:
            np.random.seed(seed)
        seeds = np.random.randint(10**6,size = nsamples)
        boot_data = []
        for seed in seeds: 
            if sampling_method =='random':
               boot_data.append(self.sample_randomly(perc,seed))
            elif sampling_method =='uniform_energy':
                boot_data.append(self.sample_energy_data_uniformly(nbins,bin_pop))
        return boot_data
    
    def sample_energy_data_uniformly(self,nbins=100,bin_pop=1):
        E = np.array(self.data['Energy'])
        Emin = E.min()
        Emax = E.max()
        dE = (Emax - Emin)/float(nbins)
        indexes = []
      #  ener_mean_bin = np.empty(n_bins,dtype=float)
     #   ncons_in_bins = np.empty(n_bins,dtype=int)
        for i in range(0,nbins-1):
            # Find confs within the bin
            ener_down = Emin+dE*float(i)#-1.e-15
            ener_up   = Emin+dE*float(i+1)#+1.e-15
            fe = np.logical_and(ener_down<=E,E<ener_up )
            temp_conf = np.array([j for j in self.data[fe].index])
            if temp_conf.shape[0] >= bin_pop: 
                con = np.random.choice(temp_conf,size=bin_pop)
            else:
                con = temp_conf
            indexes += con 
        return self.data.loc[indexes]
    
    def setup_bonds(self,distance_setup):
        self.create_dist_matrix()
        data = self.data
        distance_setup = {tuple(np.sort(k)):v for k,v in distance_setup.items()}
        size = len(data)
        bonds = np.empty(size,dtype=object)

        index = data.index
        for i in range(size):
            j = index[i]
            
            natoms = data.loc[j,'natoms']
            
            dists = data.loc[j,'dist_matrix']
            at_types = data.loc[j,'at_type']
            bj = []
            for m in range(0,natoms):
                for n in range(m+1,natoms):
                    ty = tuple(np.sort([at_types[m],at_types[n]]))
                    if ty not in distance_setup:
                        continue
                    else:
                        d = dists[m,n]
                        if d < distance_setup[ty][0]:
                            bj.append([m,n,2])
                        elif d<= distance_setup[ty][1]:
                            bj.append([m,n,1]) 
            bonds[i] = bj
            
        data['Bonds'] = bonds
        return
    
    @staticmethod
    def get_pair_distance_from_data(data,atom1,atom2):
        confs = data['coords'].to_numpy()
        dist = np.empty(len(data),dtype=float)
        for i,conf in enumerate(confs):
            r1 = conf[atom1]
            r2 = conf[atom2]
            dist[i] = VectorGeometry.calc_dist(r1,r2)
        return dist
            
    @staticmethod
    def get_pair_distance_from_distMatrix(data,atom1,atom2):
        distM = data['dist_matrix'].to_numpy()
        dist = np.empty(len(data),dtype=float)
        for i,cd in enumerate(distM):
            dist[i] = cd[atom1,atom2]
        return dist
    
    def get_systems_data(self,data,sys_name=None):
        if sys_name == None:
            if len(np.unique(data['sys_name']))>1:
                raise Exception('Give sys_name for data structures with more than one system')
            else:
                data = self.data
        else:
            data = self.data [ self.data['sys_name'] == sys_name ] 
            
        return data
    def plot_distribution(self,ty,inter_type='vdw',bins=100,ret_max=False):
        dists = self.get_distribution(ty,inter_type)
        _ = plt.figure(figsize=(3.5,3.5),dpi=300)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=3)
        plt.tick_params(direction='in', which='major',length=5)
        plt.ylabel('distribution')
        if inter_type in ['vdw','connectivity']:
            plt.xlabel('r({}) \ $\AA$'.format('-'.join(ty)))
        else:
            plt.xlabel(r'{:s}({:s}) \ '.format(inter_type,'-'.join(ty)))
        plt.hist(dists,bins=bins,histtype='step',color='magenta')
        #plt.show()
        if ret_max:
            return dists.max()
        return
    def get_distribution(self,ty,inter_type='vdw'):
        data = self.data
        #data = self.get_systems_data(self.data,sys_name)
        pair_dists = []
        #key = tuple(np.sort([t for t in ty]))
        for j,d in data.iterrows():
            try:
                d['values'][inter_type][ty]
            except KeyError:
                continue        
            pair_dists.extend(d['values'][inter_type][ty])
        return np.array(pair_dists)
    

    
class Optimizer():
    def __init__(self,data, train_indexes,dev_indexes,setup):
        if isinstance(data,pd.DataFrame):
            self.data = data
        else:
            raise Exception('data are not pandas dataframe')
        '''
        for t in [train_indexes,dev_indexes,valid_indexes]:
            if not isinstance(t,type(data.index)):
                s = '{} is not pandas dataframe index'.format(t)
                logger.error(s)
                raise Exception(s)
        '''
        self.train_indexes = train_indexes
        self.dev_indexes = dev_indexes
        
        if isinstance(setup,Setup_Interfacial_Optimization):
            self.setup = setup
        else:
            s = 'setup is not an appropriate object'
            logger.error(s)
            raise Exception(s)
        return
    
    @property
    def data_train(self):
        return self.data.loc[self.train_indexes]
    @property
    def data_dev(self):
        return self.data.loc[self.dev_indexes]


        
    
class Interfacial_FF_Optimizer(Optimizer):
    
    def __init__(self,data, train_indexes, dev_indexes, setup):
        super().__init__(data, train_indexes, dev_indexes, setup)
        
        return
    
    def get_indexes(self,dataset):
        if dataset=='train':
            return self.train_indexes
        elif dataset=='dev':
            return self.dev_indexes
        elif dataset =='all':
            return self.data.index
        else:
            raise Exception('available options are {"train", "dev",  "all"}')
    
    
    def get_list_of_model_information(self,models,dataset):
        
        #serialize parameters and bounds

        values_dict = self.get_dataDict(dataset,'values')
        
        fid = self.get_dataDict(dataset,'forces_info')
        #neibs_dict = self.get_dataDict(dataset,'neibs')
        n = len(values_dict)
        #rhats_dict = self.get_dataDict(dataset,'rhats')
        
        models_list = []

        for name,model in models.items():
            
            if self.check_excess_models(model.category,model.num):
                continue
            
            dists,dl,du = self.serialize_values(values_dict,model)
            
            minfo = self.Model_Info(model, dists, dl, du)
            
            if model.feature == 'vdw' or model.feature == 'connectivity':
                
                minfo.rhats_ij = {m: v[model.feature][model.type]['rhats_ij'] 
                                  for m,v in enumerate(fid.values()) }
                minfo.i_index = {m: v[model.feature][model.type]['i_index'] 
                                  for m,v in enumerate(fid.values()) }
                minfo.j_index = {m: v[model.feature][model.type]['j_index'] 
                                  for m,v in enumerate(fid.values()) }
            elif model.feature == 'rhos':
                minfo.v_ij = {m: v[model.feature][model.type]['v_ij'] 
                                  for m,v in enumerate(fid.values()) }                
                minfo.i_index = {m: v[model.feature][model.type]['i_index'] 
                                  for m,v in enumerate(fid.values()) }
                minfo.j_index = {m: v[model.feature][model.type]['j_index'] 
                                  for m,v in enumerate(fid.values()) }
                minfo.to_pair_index = {m: v[model.feature][model.type]['to_pair_index'] 
                                  for m,v in enumerate(fid.values()) }
            '''
            if model.feature in ['rhos','connectivity','vdw']:
                minfo.mij_to_value_index = self.get_mij_to_value_index(neibs_dict,model)
            
                minfo.mij_to_directionVec = self.get_mij_to_directionVec(rhats_dict, neibs_dict, model)
            else:
                raise NotImplementedError('angles and dihedrals are not supported in this function')
            '''
            models_list.append(minfo)
        
        return  models_list
    
    def UvectorizedContribution_ondata(self,which='opt',dataset='all'):
        ndata = len(self.get_Energy(dataset))
        models = getattr(self.setup,which+'_models')
        params, bounds, fixed_params, isnot_fixed,reguls = self.get_parameter_info(models)        
        models_list_info =  self.get_list_of_model_information(models,dataset)
        sys.stdout.flush()
        Uclass = self.computeUclass(params, ndata, models_list_info)
        index = self.get_indexes(dataset)
        self.data.loc[index,'Uclass'] = Uclass
        return Uclass

    @staticmethod
    def UvectorizedContribution(u_model,dists,dl,du,model_pars,*model_args):
        #compute UvectorizedContribution
        #a = (dists,model_pars,*model_args)
        Up = np.empty((dl.shape[0]), dtype=np.float64)
        pobj = u_model(dists,model_pars,*model_args)
        U = pobj.u_vectorized()
        #U = u_model(*a)
        for i in prange(Up.size):
            ui = U[dl[i] : du[i]].sum()
            Up[i] = ui 
        
        return Up
   
    @staticmethod
    def UvectorizedContribution_grad(u_model,dists,dl,du,model_pars,*model_args):
        #compute UvectorizedContribution
        #a = (dists,model_pars,*model_args)
        n_p = model_pars.shape[0]
        data_size = dl.shape[0]
        gu = np.empty((n_p,data_size), dtype=np.float64)
        
        pobj = u_model(dists,model_pars,*model_args)
        g = pobj.find_params_gradient() # shape = (n_p, dists.shape[0]) # dists.shape[0] == all the values serialized

        for j in range(n_p):
            for i in prange(data_size):
                gu[j][i] = g[j][dl[i] : du[i]].sum()
        
        return gu
    
    @staticmethod
    def computeUclass(params,ne,models_list_info):
        Uclass = np.zeros(ne,dtype=float)
        npars_old = 0
        for minf in models_list_info:
            #t0 = perf_counter()
            npars_new = npars_old  + minf.n_notfixed
            
            objparams = params[ npars_old : npars_new ]
            model_pars = Interfacial_FF_Optimizer.array_model_parameters(objparams,
                                                    minf.fixed_params,
                                                    minf.isnot_fixed,
                                                    )
            #print('overhead {:4.6f} sec'.format(perf_counter()-t0))
            #compute Uclass
            Utemp = Interfacial_FF_Optimizer.UvectorizedContribution( minf.u_model,
                    minf.dists, minf.dl, minf.du,
                    model_pars, *minf.model_args)
            #print(minf.name, Utemp)
            Uclass += Utemp
            #print('computation {:4.6f} sec'.format(perf_counter()-t0))
            npars_old = npars_new
        return Uclass
    
    @staticmethod
    def gradUclass(params,ne,models_list_info):
        n_p = params.shape[0]
        Uclass_grad = np.zeros((n_p,ne), dtype=np.float64)
        npars_old = 0
        for minf in models_list_info:
            #t0 = perf_counter()
            npars_new = npars_old  + minf.n_notfixed
            
            objparams = params[ npars_old : npars_new ]
            model_pars = Interfacial_FF_Optimizer.array_model_parameters(objparams,
                                                    minf.fixed_params,
                                                    minf.isnot_fixed,
                                                    )

            gu = Interfacial_FF_Optimizer.UvectorizedContribution_grad( 
                    minf.u_model,
                    minf.dists, minf.dl, minf.du,
                    model_pars, *minf.model_args)

            Uclass_grad[npars_old: npars_new] += gu[minf.isnot_fixed][:]
           
            npars_old = npars_new
        return Uclass_grad
    
    @staticmethod
    def computeForceClass(params, ne, natoms_per_point, models_list_info):
        
        Forces = {m: np.zeros( (natoms,3),dtype=float) for m, natoms in enumerate(natoms_per_point) }
        
        npars_old = 0
        for model_info in models_list_info:
            #t0 = perf_counter()
            print(model_info.name)
            npars_new = npars_old  + model_info.n_notfixed
            
            objparams = params[ npars_old : npars_new ]
            model_pars = Interfacial_FF_Optimizer.array_model_parameters(objparams,
                                                    model_info.fixed_params,
                                                    model_info.isnot_fixed,
                                                    )

            Interfacial_FF_Optimizer.ForcesForEachPoint(Forces, model_pars, model_info)
            npars_old = npars_new
        return Forces
    
    @staticmethod
    def ForcesForEachPoint(Forces, model_pars, model_info ):
        #compute UvectorizedContribution
        dists = model_info.dists
        compute_obj = model_info.u_model(dists,model_pars,*model_info.model_args)
        
        dl = model_info.dl
        du = model_info.du
        i_index = model_info.i_index
        j_index = model_info.j_index
        
        if 'PW' == model_info.name[:2]:
            rhats_ij = model_info.rhats_ij
        elif 'LD' == model_info.name[:2]:
            to_ij = model_info.to_pair_index
            v_ij = model_info.v_ij
        dudx_vectorized = - compute_obj.find_dydx()
        dudx_vectorized = dudx_vectorized.reshape((dudx_vectorized.shape[0],1))
        
        if 'PW' == model_info.name[:2]:
            for m in range(len(Forces)):
                chunk_m = dudx_vectorized[dl[m]:du[m]]
                #chunk_m = chunk_m.reshape( (chunk_m.shape[0],1) )
                forces_m = Forces[m]
                #npairs = len(ij_to_value_index) ; natoms = Forces[m].shape[0] ; print(m,natoms,npairs,(natoms-1)*natoms/2)
                pw = chunk_m*rhats_ij[m]
                np.add.at(forces_m,i_index[m], pw)
                np.add.at(forces_m,j_index[m], -pw)
        elif 'LD' ==   model_info.name[:2]:
            for m in range(len(Forces)):
                chunk_m = dudx_vectorized[dl[m]:du[m]][to_ij[m]]
                #chunk_m = chunk_m.reshape( (chunk_m.shape[0],1) )
                forces_m = Forces[m]
                #npairs = len(ij_to_value_index) ; natoms = Forces[m].shape[0] ; print(m,natoms,npairs,(natoms-1)*natoms/2)
                pw = chunk_m*v_ij[m]
                np.add.at(forces_m,i_index[m], pw)
                np.add.at(forces_m,j_index[m], -pw)
        return
    '''
    @staticmethod
    def ForcesForEachPoint(Forces, model_pars, model_info ):
        #compute UvectorizedContribution
        dists = model_info.dists
        compute_obj = model_info.u_model(dists,model_pars,*model_info.model_args)
        
        dl = model_info.dl
        du = model_info.du
        dudx_vectorized = - compute_obj.find_dydx()
        for m, ij_to_value_index in model_info.mij_to_value_index.items():
            chunk_m = dudx_vectorized[dl[m]:du[m]]
            forces_m = Forces[m]
            #npairs = len(ij_to_value_index) ; natoms = Forces[m].shape[0] ; print(m,natoms,npairs,(natoms-1)*natoms/2)
            for (i,j), value_index in ij_to_value_index.items():
                
                fij = chunk_m[value_index] * model_info.mij_to_directionVec[ (m, i, j ) ] 
                forces_m[i] += fij
                forces_m[j] -= fij
                
        return
    '''
    
    
    def test_ForceClass(self, which='opt', epsilon=1e-4,  seed = 2024,
                        verbose=False,random_tries=10,
                        check_only_analytical_forces=False):
        """
        Compute and compare the analytical and numerical Forces
        using second order finite difference methods.
    
        This method calculates the force on randomly selected atoms per data-point at each try. 
. 
    
        Parameters:
        ----------
        which : str, optional
            Specifies which model to use for analytical gradient computation. 
            Default is 'opt', which refers to the optimized model.
            
        epsilon : float, optional
            The step size for finite differences when calculating numerical gradients. 
            Default is 1e-4.
    
        seed : int, optional
            Random seed for reproducibility of any stochastic processes in the function. 
            Default is 2024.
    
        verbose : bool, optional
            If True, prints detailed information about the computation and comparisons. 
            Default is False.
    
        random_tries : int, optional
            The number of random configurations or trials to evaluate the gradients. 
            Default is 10.
        check_only_analytical_forces : bool, optional
            If True, it prints the mean and maximum analytical force for each point and returns
        """
        dataset='all'
        
        ndata = len(self.get_Energy(dataset))
        models = getattr(self.setup, which + '_models')
        params, bounds, fixed_params, isnot_fixed, reguls = self.get_parameter_info(models)
        models_list_info = self.get_list_of_model_information(models, dataset)
        
        t0 = perf_counter()
        _ = self.computeUclass(params, ndata, models_list_info)
        tf = perf_counter() - t0
        print('Time to compute analytical classical energy {:4.3e} ms'.format(tf*1000))
        
        # Analytical gradient calculation
        natoms_per_point = self.data['natoms'].to_numpy()
        t0 = perf_counter()
        Forces_analytical = self.computeForceClass(params, ndata, natoms_per_point, models_list_info)
        tf = perf_counter() - t0
        print('Time to compute Analytical Forces = {:4.3e}  ms, {:4.3e} ms/datapoint '.format(tf*1000,tf*1000/len(self.data)))
        if check_only_analytical_forces:
            for m, fa in Forces_analytical.items():
                max_force = np.abs(fa).max()
                min_force = np.abs(fa).min()
                mean_force = np.abs(fa).mean()
                print(m,fa.sum())
                print('point {:d} max_force = {:4.3e} mean_force = {:4.3e} min_force = {:4.3e}'.format(m,max_force,mean_force,min_force))
                if verbose:
                    print(fa)
            return
        # Numerical gradient calculation
        
        Forces_numerical =  {m: np.zeros( (natoms,3),dtype=float) for m, natoms in enumerate(natoms_per_point) }
        
        np.random.seed(42)
        
        self.data.drop(columns=['rhats','values', 'interactions' ,'neibs'], inplace=True)
        
        
        coords_copy = copy.deepcopy(self.data['coords'].to_numpy())
        if verbose:
            print(f'Calculating the Forces on a random atom, seed = {seed} ...')
        
        all_diffs = []
        for random_try in range(random_tries):
            atoms_to_modify = [np.random.randint(0,natoms) for m, natoms  in enumerate(natoms_per_point)]
            differences = []
            for dir_index in range(3): 
                
                for m,idx in enumerate(self.data.index):
                    atom_index = atoms_to_modify[m]
                    self.data['coords'][idx][atom_index][dir_index] += epsilon
                    
                al_help.make_interactions(self.data, self.setup)
                models_list_info = self.get_list_of_model_information(models, dataset)    
                up1 = self.computeUclass(params, ndata, models_list_info)
                
                self.data.drop(columns=['rhats','values', 'interactions' ,'neibs','coords'], inplace=True)
                self.data['coords'] = copy.deepcopy(coords_copy)
                
                for m,idx in enumerate(self.data.index):
                    atom_index = atoms_to_modify[m]
                    self.data['coords'][idx][atom_index][dir_index] -= epsilon
                
                al_help.make_interactions(self.data, self.setup)
                models_list_info = self.get_list_of_model_information(models, dataset)    
               
                um1 = self.computeUclass(params, ndata, models_list_info)
                
                self.data.drop(columns=['rhats','values', 'interactions' ,'neibs','coords'], inplace=True)
                self.data['coords'] = copy.deepcopy(coords_copy)
                
                for m in Forces_numerical.keys():
                    atom_index = atoms_to_modify[m]
                    Forces_numerical[m][atom_index, dir_index] = (up1[m] - um1[m]) / ( 2*epsilon)
                if verbose:
                    print(f'Numerical Forces Calculated. Comparing direction {dir_index}...')
                
                for m in Forces_numerical.keys():
                    
                    fn = Forces_numerical[m]
                    fa = Forces_analytical[m]
                    
                    atom_index = atoms_to_modify[m]
                    
                    fn_ad = fn[atom_index , dir_index]
                    fa_ad = fa[atom_index , dir_index]
                    diff = np.abs(fn_ad - fa_ad)
                    differences.append(diff)
                    if verbose:
                        print('data_point = {:d}, atom = {:d}, dir = {:d}, Fnum = {:.4e} , Fana = {:.4e} --> diff = {:.4e}'.format( m, atom_index, dir_index, fn_ad, fa_ad, diff))
            dmax = np.max(differences)
            dmean = np.mean(differences)
            print('random try {:d} --> max diff = {:4.3e}, mean diff = {:4.3e}'.format(random_try,dmax,dmean))
            all_diffs.extend(differences)
        return 
    
    def test_gradUclass(self, which='opt', dataset='all', epsilon=1e-4, order=2):
        """
        Calculate the analytical and numerical gradients of Uclass with options for higher-order finite differences.
    
        Parameters:
        - which: str, specifies which model to use (default is 'opt').
        - dataset: str, specifies the dataset to use (default is 'all').
        - epsilon: float, the step size for finite differences (default is 1e-4).
        - order: int, the order of finite difference approximation to use (2 or 4, default is 2).
                  order=2 gives a second-order central difference,
                  order=4 gives a fourth-order central difference.
        
        Returns:
        - grads_analytical: analytical gradient.
        - grads_numerical: numerical gradient.
        """
        ndata = len(self.get_Energy(dataset))
        models = getattr(self.setup, which + '_models')
        params, bounds, fixed_params, isnot_fixed, reguls = self.get_parameter_info(models)
        models_list_info = self.get_list_of_model_information(models, dataset)
    
        # Analytical gradient calculation
        grads_analytical = self.gradUclass(params, ndata, models_list_info)
        
        # Numerical gradient calculation
        n_p = params.shape[0]
        grads_numerical = np.empty((n_p, ndata), dtype=np.float64)
        
        for i in range(n_p):
            if order == 2:
                # Second-order central difference
                p1 = params.copy()
                p1[i] += epsilon
                up1 = self.computeUclass(p1, ndata, models_list_info)
                
                m1 = params.copy()
                m1[i] -= epsilon
                um1 = self.computeUclass(m1, ndata, models_list_info)
                
                grads_numerical[i] = (up1 - um1) / (2 * epsilon)
            
            elif order == 4:
                # Fourth-order central difference
                p1 = params.copy()
                p1[i] += epsilon
                up1 = self.computeUclass(p1, ndata, models_list_info)
                
                p2 = params.copy()
                p2[i] += 2 * epsilon
                up2 = self.computeUclass(p2, ndata, models_list_info)
                
                m1 = params.copy()
                m1[i] -= epsilon
                um1 = self.computeUclass(m1, ndata, models_list_info)
                
                m2 = params.copy()
                m2[i] -= 2 * epsilon
                um2 = self.computeUclass(m2, ndata, models_list_info)
                
                grads_numerical[i] = (-up2 + 8 * up1 - 8 * um1 + um2) / (12 * epsilon)
            
            else:
                raise ValueError("Order must be 2 or 4.")
    
        return grads_analytical, grads_numerical

    
    @staticmethod
    #@jit(nopython=True,fastmath=True)
    def compute_RegCost(params,reg,reguls,regf):
        return reg*regf(params*reguls)
    
    @staticmethod 
    def grad_RegCost(params,reg,reguls,grad_regf):
        return reg*grad_regf(params*reguls)
    
    @staticmethod
    def CostFunction(params,costf,grad_costf,
                     Energy,we,models_list_info,
                     reg,reguls,regf,grad_regf):   
        # serialize the params
        ne = Energy.shape[0]
        Uclass = Interfacial_FF_Optimizer.computeUclass(params,ne,models_list_info)
        # Compute cos
        cost = costf(Energy,Uclass,we)
        cost+= Interfacial_FF_Optimizer.compute_RegCost(params,reg,reguls,regf)
        return cost
    
    @staticmethod
    def gradCost(params,costf,grad_costf,
                 Energy,we,models_list_info,
                 reg,reguls,regf,grad_regf):   
        # serialize the params
        ne = Energy.shape[0]
        Uclass = Interfacial_FF_Optimizer.computeUclass(params,ne,models_list_info)
        gradU = Interfacial_FF_Optimizer.gradUclass(params,ne,models_list_info)
        # Compute cos
        grads = np.sum( grad_costf(Energy, Uclass,we) * gradU, axis = 1)
        grads += Interfacial_FF_Optimizer.grad_RegCost(params,reg,reguls,grad_regf)
        
        return grads
    
   
    @staticmethod
    def array_model_parameters(params,fixed_params,isnot_fixed):
        model_pars = []
        k1 = 0 ; k2 = 0
        for isnf in isnot_fixed:
            if isnf: 
                model_pars.append(params[k1]) 
                k1 +=1
            else:
                model_pars.append(fixed_params[k2]) 
                k2+=1
        model_pars = np.array(model_pars)
        
        return model_pars
    

    
    def get_mij_to_directionVec(self,rhats_dict, neibs_dict, model):
        
        if model.feature == 'rhos':
            r0 = self.setup.rho_r0
            rc = self.setup.rho_rc
            c = Interactions.compute_coeff(r0,rc)
            dphi_rho = Interactions.dphi_rho
            scale_rhats = True
            distances_ij = self.data['dist_matrix']
        else:
            scale_rhats = False
            
        mij_to_directionVec = dict()
        for m,(idx,val) in enumerate(neibs_dict.items()):
            for i,neibs in neibs_dict[idx][model.feature][model.type].items():
                for j in neibs:
                    if scale_rhats:
                        r_ij = distances_ij[idx][i,j]
                        scale = dphi_rho(r_ij,c,r0,rc)
                    else:
                        scale = 1.0
                    mij_to_directionVec[(m, i, j) ] = rhats_dict[idx][i,j]*scale
        return mij_to_directionVec
   
    def get_mij_to_value_index(self, neibs_dict, model):
  
        mij_to_value_index = dict()
        
        for m,(idx,val) in enumerate(neibs_dict.items()):
        
            value_index = 0
            mij_to_value_index[m] = dict()
            for i,neibs in neibs_dict[idx][model.feature][model.type].items():
                for j in neibs:
                    mij_to_value_index[m][ (i, j) ] = value_index
                    if model.feature in ['connectivity', 'vdw']:
                        value_index+=1
                if model.feature == 'rhos':
                    value_index +=1 
        return mij_to_value_index
    
    def serialize_values(self,values_dict,model):
        
        ndata = len(values_dict)
        dl = np.empty((ndata,), dtype=int)
        du = np.empty((ndata,), dtype=int)
        
        dists = [] 
       
        nd =0
        
        for m,(idx,val) in enumerate(values_dict.items()):
            if model.type  not in val[model.feature]:
                d = np.empty(0,dtype=float)
            else:
                d = val[model.feature][model.type]
                # val_to_ij
            dl[m] = nd
            nd += d.shape[0]
            du[m] = nd 
            dists.extend(d)

        dists = np.array(dists)
        
        return dists,dl,du
    
    def get_Energy(self,dataset):
        if dataset =='train':
            E =  np.array(self.data_train['Energy'])
        elif dataset =='dev':
            E =  np.array(self.data_dev['Energy'])
        elif dataset == 'all':
            E =  np.array(self.data['Energy'])
        else:
            s = "'dataset' only takes the values ['train','dev','all']"
            logger.error(s)
            raise Exception(s)
        return E
    
    def get_dataDict(self,dataset,column):
        if dataset =='train':
            data_dict = self.data_train[column].to_dict()
        elif dataset =='dev':
            data_dict = self.data_dev[column].to_dict()
        elif dataset == 'all':
            data_dict = self.data[column].to_dict()
        else:
            s = "'dataset' only takes the values ['train','dev'','all']"
            logger.error(s)
            raise Exception(s)
        return data_dict
    
    class CostValues:
        def __init__(self):
            self.reg = None
            self.cost = None
            self.total = None
            self.train = None
            self.dev = None
            
            return
        def __repr__(self):
            x = 'Cost : value \n--------------------\n'
            for k,v in self.__dict__.items():
                x+='{} : {:7.8f} \n'.format(k,v)
            x+='--------------------\n'
            return x 
        
    class Model_Info():
        def __init__(self,model,dists,dl,du):
            
            fixed_params = []; isnot_fixed = []
           
            for k, m in model.pinfo.items():
                isnot_fixed.append(m.opt) # its a question variable
                if not  m.opt:
                    fixed_params.append(m.value)
                    
            fixed_params = np.array(fixed_params)
            isnot_fixed = np.array(isnot_fixed)
            self.name = model.name
            self.u_model = model.function 
            self.model_args = model.model_args
            self.fixed_params = fixed_params
            self.isnot_fixed = isnot_fixed
            self.n_notfixed = np.count_nonzero(isnot_fixed)
            self.dists = dists
            self.dl = dl
            self.du = du
            return

        def __repr__(self):
            x = 'Attribute : value \n--------------------\n'
            for k,v in self.__dict__.items():
                x+='{} : {} \n'.format(k,v)
            x+='--------------------\n'
            return x 
    
    def check_excess_models(self,cat,num):
        return num>=getattr(self.setup,'n'+cat)

    def get_parameter_info(self,models):
        params = []; fixed_params = []; isnot_fixed = []; bounds =[]
        reguls =[]
        for name,model in models.items():
            if self.check_excess_models(model.category,model.num):
                continue
            for k, m in model.pinfo.items():
                isnot_fixed.append(m.opt) # its a question variable
                if m.opt:
                    params.append(m.value)
                    bounds.append( (m.low_bound,m.upper_bound) )
                    reguls.append(m.regul)
                else: 
                    fixed_params.append(m.value)
                
        params = np.array(params)
        fixed_params = np.array(fixed_params)
        isnot_fixed = np.array(isnot_fixed) # boolean
        reguls = np.array(reguls)
        return params, bounds, fixed_params, isnot_fixed,reguls
    
    
    def randomize_initial_params(self,params,bounds):
        if self.setup.increased_stochasticity >0 and self.randomize:
            s = self.setup.increased_stochasticity
            ran = [b[1]-b[0] for b in bounds]
            moves = np.random.randint(1,self.setup.max_moves)
            for _ in range(moves):
                change_id = np.random.randint(0,params.shape[0])
                params[change_id] += np.random.normal(0,s*ran[change_id])
                bup = bounds[change_id][1]
                blow = bounds[change_id][0]
                if params[change_id] > bup:
                    params[change_id] = 0.999*bup if bup > 0 else 1.001*bup
                if params[change_id] < blow:
                    params[change_id] = 0.999*blow if blow < 0 else 1.001*blow
        return params
    
    def optimize_params(self,setfrom='init'):
        t0 = perf_counter()

        E = self.get_Energy('train')
        weights =  GeneralFunctions.weighting(self.data_train,
                self.setup.weighting_method,self.setup.bT,self.setup.w)
        #argsopt= np.where (self.data_train['label']=='optimal')[0]
        #weights[argsopt]*=5.0
        self.weights = weights

        models = getattr(self.setup,setfrom+'_models')

        params, bounds, fixed_parameters, isnot_fixed, reguls = self.get_parameter_info(models)        
        models_list_info  = self.get_list_of_model_information(models,'train')


        opt_method = self.setup.optimization_method
        tol = self.setup.tolerance
        maxiter = self.setup.maxiter
        
        costf = getattr(measures(),self.setup.costf)
        grad_costf = getattr(measures(),'grad_' + self.setup.costf)
        
        CostFunc = self.CostFunction
        
        regf = getattr(regularizators(),self.setup.regularization_method)
        grad_regf = getattr(regularizators(),'grad_'+self.setup.regularization_method)
        args = (costf,grad_costf,E,weights,models_list_info, self.setup.reg_par,reguls,regf,grad_regf)
       
        self.models_list_info = models_list_info
        
        try:
            self.randomize
        except AttributeError:
            self.randomize= False
        
        if params.shape[0] >0 and self.setup.optimize :
            params = self.randomize_initial_params(params,bounds)
            t1 = perf_counter()
            
            if opt_method in ['SLSQP','BFGS','L-BFGS-B']:   
                logger.debug('I am performing {}'.format(opt_method))
                t0 = perf_counter()
                res = minimize(CostFunc, params,
                               args =args,
                               jac = self.gradCost,
                               bounds=bounds,tol=tol, 
                               options={'disp':self.setup.opt_disp,
                                        'maxiter':maxiter,
                                        'ftol': self.setup.tolerance},
                               method = opt_method)
                tmt = perf_counter() - t0 
                self.minimization_time = tmt
                self.tpfev = tmt/res.nfev
            elif opt_method =='DE':
                
                workers = 1 # currently only working with 1
                res = differential_evolution(CostFunc, bounds,
                            args =args,
                            maxiter = maxiter,
                            polish = self.setup.polish,
                             workers=workers ,
                            recombination=self.setup.recombination,
                            mutation = self.setup.mutation,
                            disp =self.setup.opt_disp)
            
            elif  opt_method == 'DA':
                #logger.debug('I am performing dual annealing')
                print('I am performing dual annealing ...')
                sys.stdout.flush()
                res = dual_annealing(CostFunc, bounds,
                            args =args ,x0=params,
                             maxiter = maxiter,
                             initial_temp=self.setup.initial_temp,
                             restart_temp_ratio=self.setup.restart_temp_ratio,
                             visit=self.setup.visit,
                             accept = self.setup.accept,)
                
            elif opt_method == 'stochastic_SLSQP':
                print('STOCHASTIC SLSQP Began')
                sys.stdout.flush()
                tmethod = perf_counter()

                if not hasattr(self,'epoch'):
                     self.set_models('init','inittemp')
                     self.total_train_indexes = self.train_indexes.copy()
                     self.epoch = 0
                best_cost = 1e16
                best_epoch = 0

                self.setup.optimization_method = 'SLSQP'
                self.randomize = False
                total_fev = 0
                time_only_min = 0
                while(self.epoch <= self.setup.nepochs):
                    current_total_indexes = list (self.total_train_indexes.copy())
                    itera=0
                    while( len(current_total_indexes) > 0):
                        self.train_indexes = np.random.choice(current_total_indexes,replace=False,
                                                              size=min(self.setup.batchsize, E.shape[0] , len(current_total_indexes) )
                                                              )
                        current_total_indexes = [ i for i in current_total_indexes if i not in self.train_indexes ]

                        self.optimize_params('init')
                        time_only_min += self.minimization_time
                        cost = self.current_costs.dev

                        self.set_models('opt','init')
                        #
                        if cost <= best_cost:
                            best_epoch = self.epoch
                            best_cost = cost
                            self.set_models('opt','best_opt')
                            self.best_costs = copy.deepcopy(self.current_costs)

                        print('epoch = {:d}, i = {:d},  development cost = {:.4e} train cost = {:.4e} '.format(self.epoch,itera,  cost, self.current_costs.train))
                        sys.stdout.flush()
                        total_fev += self.res.nfev
                        itera+=1
                        self.randomize = False

                    self.randomize = True
                    
                    self.train_indexes = self.total_train_indexes.copy()
                    #self.optimize_params('best_opt')
                   #self.set_models('best_opt','init')
                    if self.epoch % 1 == 0:
                        tn = perf_counter() - tmethod
                        print('epoch = {:d} ,  best_epoch = {:d} , best_cost = {:.4e}'.format(self.epoch, best_epoch, best_cost))

                        sys.stdout.flush()

                    self.epoch+=1
                    if self.epoch > best_epoch+50:
                        break
                    ## loop end
                print('time elapsed = {:.3e} sec t/fev = {:.3e} sec/eval eval_time = {:.3e} , overhead = {:.3e} '.format( tn, 
                    time_only_min/total_fev,time_only_min, tn-time_only_min))
                #Final on all data
                self.set_models('best_opt','opt')
                self.set_models('best_opt','init')
                
                self.train_indexes = self.total_train_indexes
                temp = self.setup.increased_stochasticity

                self.setup.increased_stochasticity = 0.0
                
                #self.optimize_params()

                print('Final: best_epoch = {:d} , best_cost = {:.4e}, time needed = {:.3e} sec'.format(best_epoch,best_cost,perf_counter()-tmethod))
                sys.stdout.flush()
                #fixing setup
                self.setup.optimization_method = 'stochastic_SLSQP'
                self.setup.increased_stochasticity = temp
                print('STOCHASTIC SLSQP FINISHED')
                sys.stdout.flush()
                return 
            else:
                s = 'Error: wrong optimization_method name'
                logger.error(s)
                raise Exception(s)
            self.opt_time = perf_counter() - t1
            optimization_performed=True
        else:
            optimization_performed = False
        
        if optimization_performed:
            self.set_results(res,isnot_fixed,fixed_parameters,reguls,regf)
            
             
        else:
            # Filling with dump values to avoid coding errors
            self.set_models('init','opt')
            dump_value = 123321
            self.regCost = dump_value ; self.unregCost = dump_value ; self.results = dump_value
                    
            
        return 
    def set_results(self,res,isnot_fixed,fixed_parameters,reguls,regf):
        
        self.set_models('init','opt',res.x,isnot_fixed,fixed_parameters)
        
        self.UvectorizedContribution_ondata('opt',dataset='all')
        costs = self.CostValues()
        
        costs.reg = self.compute_RegCost(res.x,self.setup.reg_par,reguls,regf)
        costs.cost = res.fun - costs.reg
        costs.total = res.fun
        
        dev_eval = Interfacial_Evaluator(self.data_dev,self.setup,prefix='development')
        
        train_eval = Interfacial_Evaluator(self.data_train,self.setup,prefix='train')
         
        dev_tab = dev_eval.make_evaluation_table([self.setup.costf],['sys_name'])
        
        train_tab = train_eval.make_evaluation_table([self.setup.costf],['sys_name'])
        
        costs.dev = dev_tab.loc['TOTAL',self.setup.costf]
        costs.train = train_tab.loc['TOTAL',self.setup.costf]
        
        self.current_costs = costs
        self.timef = 'time/fev = {:.3e} sec'.format(self.opt_time/res.nfev)
        logger.info(self.timef)
        
        logger.info('Total optimization time = {:.3e} sec'.format(self.opt_time))
        #Get the new params and give to vdw_dataframe
        self.res = res
        return
    
    def set_models(self,setfrom,setto,optx=None,isnotfixed=None,fixed_parameters=None):
        k1 =0 ; k2 =0
        models = getattr(self.setup,setfrom+'_models')
        models_copy = copy.deepcopy(models)
        if optx is not None:
            for n in  models_copy.keys():
                model = models_copy[n] 
                if self.check_excess_models(model.category,model.num):
                    continue
                for k in model.pinfo.keys():
                    j = k1 + k2
                    if isnotfixed[j]:
                        model.pinfo[k].value = optx[k1]
                        k1+=1
                    else:
                        model.pinfo[k].value = fixed_parameters[k2]
                        k2+=1
        name = setto+'_models'
        setattr(self,name,models_copy)
        setattr(self.setup,name,models_copy)
        return 
    
    
                
class regularizators:
    @staticmethod
    def ridge(x):
        return np.sum(x*x)/x.shape[0]
    @staticmethod
    def grad_ridge(x):
        return 2*x/x.shape[0]
    @staticmethod
    def lasso(x):
        return np.sum(abs(x))/x.shape[0]
    @staticmethod
    def grad_lasso(x):
        return np.sign(x)/x.shape[0]
    @staticmethod
    def elasticnet(x):
        return 0.5*(regularizators.ridge(x)+regularizators.lasso(x))
    @staticmethod
    def grad_elasticnet(x):
        return 0.5*(regularizators.grad_ridge(x)+regularizators.grad_lasso(x))
    @staticmethod
    def none(x):
        return 0.0
    @staticmethod
    def grad_none(x):
        return 0.0
class measures:
    
    @staticmethod
    def elasticnet(u1,u2,w=1):
        return 0.5*(measures.MAE(u1,u2,w)+measures.MSE(u1,u2,w))
    
    @staticmethod
    def grad_elasticnet(u1,u2,w=1):
        return 0.5*(measures.grad_MAE(u1,u2,w)+measures.grad_MSE(u1,u2,w))
    
    @staticmethod
    def MAE(u1,u2,w=1):
        u = np.abs(u2-u1)
        return np.sum(u*w)/u1.shape[0]
    
    @staticmethod
    def grad_MAE(u1,u2,w=1):
        u = u2-u1
        return w*np.sign(u)/u1.shape[0]
    
    @staticmethod
    def MSE(u1,u2,w=1):
        u = u2 -u1
        return np.sum(w*u*u)/u1.shape[0]
    
    @staticmethod
    def grad_MSE(u1,u2,w=1):
        u = u2-u1
        return 2*w*u/u1.shape[0]

    @staticmethod
    def BIAS(u1,u2):
        u = u2-u1
        return u.mean()
    
    @staticmethod
    def STD(u1,u2):
        r = (u1 -u2)
        return r.std()
    @staticmethod
    def relBIAS(u1,u2):
        s = np.abs(u1.min())
        u= u2-u1
        return 100*u.mean()/s
    
    @staticmethod
    def MAX(u1,u2):
        return np.abs(u2-u1).max()
    

class Evaluator:    
    def __init__(self,data,setup,selector=dict(),prefix=None):
        
        self.filt  = Data_Manager.data_filter(data,selector)
        self.data = data[self.filt]
        self.Energy = self.data['Energy'].to_numpy()
        self.setup = setup
        #if prefix is not None:
        #    fn = '{:s}_evaluations.log'.format(prefix)
        #else:
        #    fn = 'evaluations.log'
        
        #fname = '{:s}/{:s}'.format(setup.runpath,fn)
        #self.eval_fname = fname
        #self.eval_file = open(fname,'w')
        return
    def __del__(self):
        #self.eval_file.close()
        return


class Interfacial_Evaluator(Evaluator):
    def __init__(self,data,setup,selector=dict(),prefix=None):
        super().__init__(data,setup,selector,prefix)
        self.Uclass = self.data['Uclass'].to_numpy()
        
    def compare(self,fun,col):
        un = np.unique(self.data[col])
        res = dict()
        fun = getattr(measures,fun)
        for u in un:
            f = self.data[col] == u
            u1 = self.Energy[f]
            u2 = self.Uclass[f]
            res[ '='.join((col,u)) ] = fun(u1,u2)
        return res
    
    def get_error(self,fun,colval=dict()):
        f = np.ones(len(self.data),dtype=bool)
        for col,val in colval.items():
            f = np.logical_and(f,self.data[col]==val)
           
        u1 = self.Energy[f]
        u2 = self.Uclass[f]
        try:
            return fun(u1,u2)#
        except:
            return None
    
    def make_evaluation_table(self,funs,cols=None,add_tot=True,save_csv=None):

        if not GeneralFunctions.iterable(funs):
            funs = [funs]
        if cols is not None:
            if not GeneralFunctions.iterable(cols):
                cols = [cols]
            
            
            uncols = [np.unique(self.data[col]) for col in cols]
            sudoindexes = list(itertools.product(*uncols))
            ev = pd.DataFrame(columns=funs+cols)
            
            for ind_iter,index in enumerate(sudoindexes):
                for fn in funs:
                    colval = dict()
                    fun = getattr(measures,fn)
                    for i,col in enumerate(cols):    
                        ev.loc[ind_iter,col] = index[i]
                        colval[col] = index[i]
                        
                    ev.loc[ind_iter,fn] = self.get_error(fun,colval)
        else:
            ev = pd.DataFrame(columns=funs)
        if add_tot:
            for fn in funs:
                fun = getattr(measures,fn)
                ev.loc['TOTAL',fn] = self.get_error(fun)
        if cols is not None:
            ev = ev[cols+funs]
        else:
            ev = ev[funs]

        if save_csv is not None:
            ev.to_csv('{:s}/{:s}'.format(self.setup.runpath,save_csv))
        return ev
                
                
    def print_compare(self,fun,col):
        #try:
           #f = open(self.eval_fname,'a')
        #except IOError:
            #f = self.eval_file
        res = self.compare(fun,col)
        for k,v in res.items():
            pr = '{:s} : {:s} = {:4.3f}'.format(k,fun,v)
            print(pr)
            #f.write(pr+'\n')
        #f.flush()
        #f.close()
        return

    
    def plot_predict_vs_target(self,size=2.35,dpi=500,
                               xlabel=r'$E^{dft}$', ylabel=r'$U^{class}$',
                               col1='Energy', col2='Uclass',title=None,
                               units =r'$kcal/mol$',
                               label_map=None,
                               path=None, fname='pred_vs_target.png',attrs=None,
                               save_fig=True,compare=None,diff=False,scale=0.05):
        GeneralFunctions.make_dir(path)
        x_data = self.data[col1].to_numpy()
        y_data = self.data[col2].to_numpy()
        xlabel += ' (' +units+')'
        ylabel += ' (' +units+')'
        if path is None:
            path = self.setup.runpath
        if compare is not None:
            col = self.data[compare]
            uncol = np.unique(col)
            ncol = int(len(uncol)/3)
            cmap = matplotlib.cm.Spectral
            colors = [cmap(i/len(uncol)) for i in range(len(uncol)) ]  
        else:
            ncol=1
        if attrs is not None:
            uncol = np.array(attrs)
        
        
        if not diff:
            _ = plt.figure(figsize=(size,size),dpi=dpi)
            plt.minorticks_on()
            plt.tick_params(direction='in', which='minor',length=size)
            plt.tick_params(direction='in', which='major',length=2*size)
            xmin = x_data.min()
            xmax = x_data.max()
            air = scale*(xmax-xmin)
            perf_line = [xmin - air , xmax + air]
            plt.xticks(fontsize=3.0*size)
            plt.yticks(fontsize=3.0*size)
            if title is not None:
                plt.title(title,fontsize=3.5*size)
            if compare is None:
                plt.plot(x_data, y_data,ls='None',color='purple',marker='.',markersize=1.3*size,fillstyle='none')
            else:
                for i,c in enumerate(uncol):
                    f = col == c
                    if label_map is not None:
                        lbl = label_map[c]
                    else:
                        lbl = c
                    plt.plot(x_data[f], y_data[f],label=lbl,ls='None',color=colors[i],
                             marker='.',markersize=1.8*size,fillstyle='none')
            plt.plot(perf_line,perf_line, ls='--', color='k',lw=size/2)
            plt.xlabel(xlabel,fontsize=3.5*size)
            plt.ylabel(ylabel,fontsize=3.5*size)
            if label_map is not None:
                 plt.legend(frameon=False,fontsize=3.0*size)
            if fname is not None:
                plt.savefig('{:s}/{:s}'.format(path,fname),bbox_inches='tight')
            #plt.show()
        else:
            for i,c in enumerate(uncol):
                _ = plt.figure(figsize=(size,size),dpi=dpi)
                plt.minorticks_on()
                plt.tick_params(direction='in', which='minor',length=size)
                plt.tick_params(direction='in', which='major',length=2*size)
                perf_line = [x_data.min()*1.05,x_data.max()*1.05]
                plt.plot(perf_line,perf_line, ls='--', color='k',lw=size/2)
                f = col == c
                plt.plot(x_data[f], y_data[f],label=c,ls='None',color=colors[i],
                             marker='.',markersize=1.8*size,fillstyle='none')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.legend(frameon=False,ncol=ncol)
                if fname is not None:
                    plt.savefig('{:s}/{:s}'.format(path,fname),bbox_inches='tight')
                #plt.show()
        
        
        return
    
    def plot_superPW(self,PWparams,model,size=3.3,fname=None,
                       dpi=300,rmin= 1,rmax=5,umax=None,xlabel=None):
        
        figsize = (size,size) 
        _ = plt.figure(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=5)
        plt.tick_params(direction='in', which='major',length=10)
        types = PWparams[list(PWparams.keys())[0]].index
        pw_u = {t:0 for t in types}
        #pw_r = {k:0 for k in PWparams}
        r = np.arange(rmin+1e-9,rmax,0.01)
        for k,params in PWparams.items():
            pars = self.setup.model_parameters(model,params.columns)
            f = globals()['u_'+model]        
            for t, pot in params.iterrows():
                args = np.array([pot[p] for p in pars])
                u = f(r,args)
                pw_u[t]+=u.copy()
                
                if umax is not None:
                    filt = u<=umax
                    uf = u[filt]
                    rf = r[filt]
                else:
                    uf = u
                    rf =r 
                plt.plot(rf,uf,label='{}-{}'.format(k,t),ls='-',lw=0.6)
        for t in types:
            u = pw_u[t]
            if umax is not None:
                filt = u<=umax
                uf = u[filt]
                rf = r[filt]
            else:
                uf = u
                rf =r 
            plt.plot(rf,uf,label='{}-{}'.format('super',t),ls='--',lw=1.0)
        plt.legend(frameon=False,fontsize=1.5*size)
        if xlabel is None:
            plt.xlabel(r'r / $\AA$')
        else:
            plt.xlabel(xlabel)
        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')
        plt.ylabel(r'$U_{'+'{:s}'.format(model)+'}$ / kcal/mol')
        #plt.show()
        return 
    
    
    
    def plot_scan_paths(self,size=2.65,dpi=450,
                   xlabel=r'scanning distance / $\AA$', ylabel=r'$Energy$ / $kcal/mol$',
                   title=None,show_fit=True,
                   path=None, fname=None,markersize=0.7,
                   n1=3,n2=2,maxplots=None, add_u = None,
                   selector=dict(),x_col=None,subsample=None,scan_paths=(0,1e15)):
        #make_dir(path)
        self.data.loc[:,'scanpath'] = ['/'.join(x.split('/')[:-2]) for x in self.data['filename']]
        
        filt = Data_Manager.data_filter(self.data,selector)
        data = self.data[filt]
        cols = ['scanpath','filename','Energy','scan_val','Uclass'] 
        if type(add_u) is dict:
            cols += list(add_u.keys())
        #data = data[cols]
        
        
        unq = np.unique(data['scanpath'])
        nsubs=n1*n2
        if maxplots is None:
            nu = len(unq)
        else:
            nu = min(int(len(unq)/nsubs)+1,maxplots)*nsubs
        figsize=(size,size)
        #cmap = matplotlib.cm.get_cmap('tab20')
        
        if abs(n1-n2)!=1:
            raise Exception('|n1-n2| should equal one')
        for jp in range(0,nu,nsubs):
            fig = plt.figure(figsize=figsize,dpi=dpi)
            plt.xlabel(xlabel, fontsize=2.5*size,labelpad=4*size)
            plt.ylabel(ylabel, fontsize=2.5*size,labelpad=6*size)
            plt.xticks([])
            plt.yticks([])
            nfig = int(jp/nsubs)
            if title is None:
                fig.suptitle('set of paths {}'.format(nfig),fontsize=3*size)
            else:
                fig.suptitle(title,fontsize=3*size)
            gs = fig.add_gridspec(n1, n2, hspace=0, wspace=0)
            ax = gs.subplots(sharex='col', sharey='row')
            lsts = ['-','--','-','--']
            
            colors=  ['#ca0020','#f4a582','#92c5de','#0571b0']
            for j,fp in enumerate(unq[jp:jp+nsubs]):
                
                dp = data[ data['scanpath'] == fp ]
                
                x_data = dp['scan_val'].to_numpy()
                xi = x_data.argsort()   
                x_data = x_data[xi]
                #c = cmap(j%nsubs/nsubs)
                e_data = dp['Energy'].to_numpy()[xi]
                u_data = dp['Uclass'].to_numpy()[xi]
                j1 = j%n1
                j2 = j%n2
                #print(j1,j2,fp)
                ax[j1][j2].plot(x_data, e_data, ls='none', marker='o',color='gray',
                    fillstyle='none',markersize=markersize*size,label=r'$E_{int}$')
                if show_fit:
                    if type(add_u) is not dict:
                        ax[j1][j2].plot(x_data, u_data, lw=0.27*size,color='k',label=r'fit')
                    else:
                        for jmi, (k,v) in enumerate(add_u.items()):
                            u_data= dp[k].to_numpy()[xi]
                            ax[j1,j2].plot(x_data,u_data ,
                                           lw=0.27*size, color=colors[jmi],label=v,
                                           ls=lsts[jmi])
                #ax[j1][j2].legend(frameon=False,fontsize=2.0*size)
                ax[j1][j2].minorticks_on()
                ax[j1][j2].tick_params(direction='in', which='minor',length=size*0.6,labelsize=size*1.5)
                ax[j1][j2].tick_params(direction='in', which='major',length=size*1.2,labelsize=size*1.5)
            lines_labels = [ax.get_legend_handles_labels() for i,ax in enumerate(fig.axes) if i<2]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

            fig.legend(lines, labels, loc='upper center',bbox_to_anchor=(0.5,0.03), ncol=5,frameon=False,fontsize=2.0*size)
        #plt.legend(frameon=False)
            if path is None:
                path= self.setup.runpath
                
            if fname is None:
                plt.savefig('{}/spths{}.png'.format(path,nfig),bbox_inches='tight')
            else:
                plt.savefig('{:s}/{:s}'.format(path,fname),bbox_inches='tight')
            #plt.show()
    
    def plot_eners(self,figsize=(3.3,3.3),dpi=300,
                   xlabel=r'$\AA$', ylabel=r'$kcal/mol$',
                   col1='Energy', col2='Uclass',title=None,
                   length_minor=5, length_major=10,
                   path=None, fname=None,by='sys_name',
                   selector=dict(),x_col=None,subsample=None):
        #make_dir(path)
        data = self.data
        byc = data[by].to_numpy()
        
        if path is None:
            path = self.setup.runpath
        filt = Data_Manager.data_filter(data,selector)
        
        if x_col is not None:
            x_data = data[x_col][filt].to_numpy()
        else:
            x_data = np.array(data.index)
            xlabel = 'index'
        
        xi = x_data.argsort()
        if subsample is not None:
            xi = np.random.choice(xi,replace=False,size=int(subsample*len(xi)))
            xi.sort()
            
        x_data = x_data[xi]
        e_data = data[col1][filt].to_numpy()[xi]
        u_data = data[col2][filt].to_numpy()[xi]
        
        _ = plt.figure(figsize=figsize,dpi=dpi)
        if title is not None:
            plt.title(title)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=length_minor)
        plt.tick_params(direction='in', which='major',length=length_major)
        plt.plot([x_data.min(),x_data.max()],[0,0],lw=0.5,ls='--',color='k')
        
        unqbyc = np.unique(byc)
        n = len(unqbyc)
        cmap = matplotlib.cm.Spectral
        cs = [ cmap((i+0.5)/n) for i in range(n)]
        for j,b in enumerate(unqbyc):
            f = byc == b
            plt.plot(x_data[f], e_data[f],label=b,ls='None',color=cs[j],
                 marker='o',markersize=3,fillstyle='none')
            plt.plot(x_data[f], u_data[f],ls='None',color=cs[j],
                 marker='x',markersize=3,fillstyle='none')
        for i in range(len(x_data)):
            x = [x_data[i],x_data[i]]
            y = [e_data[i],u_data[i]]
            plt.plot(x,y,ls='-',lw=0.2,color='k')
        plt.ylim((1.2*e_data.min(),0.6*abs(e_data.min())))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(frameon=False,fontsize=5,ncol=max(1,int(n/3)))
        if fname is not None:
            plt.savefig('{:s}/{:s}'.format(path,fname),bbox_inches='tight')
        #plt.show()

 ####
##### ##### ##
class mappers():
    elements_mass = {'H' : 1.008,'He' : 4.003, 'Li' : 6.941, 'Be' : 9.012,\
                 'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,\
                 'F' : 18.998, 'Ne' : 20.180, 'Na' : 22.990, 'Mg' : 24.305,\
                 'Al' : 26.982, 'Si' : 28.086, 'P' : 30.974, 'S' : 32.066,\
                 'Cl' : 35.453, 'Ar' : 39.948, 'K' : 39.098, 'Ca' : 40.078,\
                 'Sc' : 44.956, 'Ti' : 47.867, 'V' : 50.942, 'Cr' : 51.996,\
                 'Mn' : 54.938, 'Fe' : 55.845, 'Co' : 58.933, 'Ni' : 58.693,\
                 'Cu' : 63.546, 'Zn' : 65.38, 'Ga' : 69.723, 'Ge' : 72.631,\
                 'As' : 74.922, 'Se' : 78.971, 'Br' : 79.904, 'Kr' : 84.798,\
                 'Rb' : 84.468, 'Sr' : 87.62, 'Y' : 88.906, 'Zr' : 91.224,\
                 'Nb' : 92.906, 'Mo' : 95.95, 'Tc' : 98.907, 'Ru' : 101.07,\
                 'Rh' : 102.906, 'Pd' : 106.42, 'Ag' : 107.868, 'Cd' : 112.414,\
                 'In' : 114.818, 'Sn' : 118.711, 'Sb' : 121.760, 'Te' : 126.7,\
                 'I' : 126.904, 'Xe' : 131.294, 'Cs' : 132.905, 'Ba' : 137.328,\
                 'La' : 138.905, 'Ce' : 140.116, 'Pr' : 140.908, 'Nd' : 144.243,\
                 'Pm' : 144.913, 'Sm' : 150.36, 'Eu' : 151.964, 'Gd' : 157.25,\
                 'Tb' : 158.925, 'Dy': 162.500, 'Ho' : 164.930, 'Er' : 167.259,\
                 'Tm' : 168.934, 'Yb' : 173.055, 'Lu' : 174.967, 'Hf' : 178.49,\
                 'Ta' : 180.948, 'W' : 183.84, 'Re' : 186.207, 'Os' : 190.23,\
                 'Ir' : 192.217, 'Pt' : 195.085, 'Au' : 196.967, 'Hg' : 200.592,\
                 'Tl' : 204.383, 'Pb' : 207.2, 'Bi' : 208.980, 'Po' : 208.982,\
                 'At' : 209.987, 'Rn' : 222.081, 'Fr' : 223.020, 'Ra' : 226.025,\
                 'Ac' : 227.028, 'Th' : 232.038, 'Pa' : 231.036, 'U' : 238.029,\
                 'Np' : 237, 'Pu' : 244, 'Am' : 243, 'Cm' : 247, 'Bk' : 247,\
                 'Ct' : 251, 'Es' : 252, 'Fm' : 257, 'Md' : 258, 'No' : 259,\
                 'Lr' : 262, 'Rf' : 261, 'Db' : 262, 'Sg' : 266, 'Bh' : 264,\
                 'Hs' : 269, 'Mt' : 268, 'Ds' : 271, 'Rg' : 272, 'Cn' : 285,\
                 'Nh' : 284, 'Fl' : 289, 'Mc' : 288, 'Lv' : 292, 'Ts' : 294,\
                 'Og' : 294}
    @property
    def atomic_num(self):
        return {'{:d}'.format(int(i+1)):elem for i,elem in enumerate(self.elements_mass.keys())}
    
    


@jit(nopython=True,fastmath=True,parallel=True)
def numba_isin(x1,x2,f):
    for i in prange(x1.shape[0]):
        for x in x2:
            if x1[i] == x: 
                f[i] = True
    return

