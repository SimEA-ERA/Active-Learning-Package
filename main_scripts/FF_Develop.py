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
import math
from mpmath import erf, mp, exp, sqrt , gamma , power
mp.dps = 32
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
                                'harmonic3':'Default need to fix',
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
    def sample_via_lammps(data,setup,parsed_args, beta_sampling):
        print('MD sampling with Lammps')
        al = al_help()
        cols = data.columns
        candidate_data = pd.DataFrame()
        lammps_main_file = "lammps_working/sample_run.lmscr"

        kB = 0.0019872037514523
        beta_target = 1.0/(kB*parsed_args.Ttarget)
        
        tsample = round(1.0/ (beta_sampling*kB), 2) 
        def update_main_file_temperature(tsample):
            with open(lammps_main_file,'r') as fil:
                lines = fil.readlines()
                fil.closed
            for l,line in enumerate(lines):
                li = line.split()
                if 'variable' in li and 'teff' in li and 'equal' in li:
                    print(f'Updating temperature to {tsample}  K' )
                    lines[l] = f'variable teff equal {tsample}\n'
            with open(lammps_main_file,'w') as fil:
                for line in lines:
                    fil.write(line)
                fil.closed
        #select randomly 0.01% of the index
        unq_sys = np.unique(data['sys_name'])
        #old_bonded_inters = 'dumb variable'
        for sname in unq_sys:
            fs = data['sys_name'] == sname
            udat = data [ fs ]
            us = udat['Uclass'].to_numpy()
            us = us - us.min()
            ps = np.exp(-us*beta_sampling)
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
                command  = '  ;  '.join([c1,c2])

                beta_eff = 1e8
                md_iter = 0
                while md_iter < 10:
                    print(f'md_iter = {md_iter} ,  beta_sampling = {beta_sampling} , tsampling = {tsample} K')
                    os.system(command)

                    new_data = al_help.read_lammps_structs('lammps_working/samples.lammpstrj',inv_types)
                    ne = len(new_data)
                    new_data = al_help.clean_well_separated_nanostructures(new_data, setup)
                    if len(new_data) < 0.01*ne:
                        print('More than 99% of the structures were well separated! rescaling beta by 1.5 to avoid desorption or dissolution')
                        beta_sampling *= 1.5
                        tsample = round(1.0/ (beta_sampling*kB), 5) 
                        update_main_file_temperature(tsample)
                        md_iter += 1
                        sys.stdout.flush()
                        continue

                    al_help.evaluate_potential(new_data, setup,'opt')
                 
                    ut = new_data['Uclass'].to_numpy()
                    shifted_energies = ut - ut.min()
                    tfit, beta_eff, alpha, weights, l_minima, fail = al_help.estimate_Teff_Beff(shifted_energies, nbins = 200 )
                    
                    al_help.plot_candidate_distribution(shifted_energies, (beta_eff, alpha, weights, l_minima), 200,
                            title = f'MD iter {md_iter}' + r': Candidate distribution \n $\beta_{target}$' + '= {:5.4f},'.format(beta_target)+ r' $\beta_{eff}$' + ' = {:5.4f}'.format( beta_eff) + r' $\beta_{sampling}$' + ' = {:5.4f}'.format( beta_sampling),
                            fname=f'{setup.runpath}/CD{md_iter}_{sname}.png')
                    
                    if fail:
                        print(f'md iter = {md_iter}: BETA SCALING FAILED! beta_eff = {beta_eff}   beta_target = {beta_target}    beta_sampling = {beta_sampling}')
                        beta_sampling = al_help.beta_distribution_fit_fail_strategy(shifted_energies , setup, beta_sampling )
                        sys.stdout.flush()
                    if np.abs ((beta_target - beta_eff)/beta_target) < 0.1 :
                        print(f'md iter = {md_iter}: BETA SCALING CONVERGED! beta_eff = {beta_eff}   beta_target = {beta_target}   beta_sampling = {beta_sampling}')
                        break
                    beta_sampling *= np.sqrt(beta_target/beta_eff)
                    
                    tsample = round(1.0/ (beta_sampling*kB), 5) 
                    
                    update_main_file_temperature(tsample)
                    
                    print(f'md_iter = {md_iter} , beta_eff = {beta_eff}')
                    md_iter += 1
                
                os.system('  ;  '.join([c1, c3,c4,c5]) )
                new_data['sys_name'] = sname
                candidate_data = candidate_data.append(new_data, ignore_index=True)
        
        print('Lammps Simulations complete')
        sys.stdout.flush()
        return candidate_data, beta_sampling
    @staticmethod
    def coordinate_simulated_annealing(data, r_setup):
        synames = np.unique(data['sys_name'])
        for sys in sysnames:
            sys_data = data [ sys == data['sys_name'] ]
            c = copy.deepcopy(np.array([ c for c in sys_data['coords'].to_numpy()]))
            init_data = copy.deepcopy(sys_data[['at_type','sys_name','natoms','coords', 'bodies']])
            init_data['coords'] = c

            natoms = init_data['natoms'].to_numpy()[0]
            params = c.flatten()
    @staticmethod
    def beta_distribution_fit_fail_strategy(u , setup, beta_sampling ):
        bs = setup.bS
        u = u - u.min()
        urange = u.max()
        outlier = u.max() - u.mean()
        if outlier > bs/beta_sampling:
            print('Found high energy outliers beta_sampling is doubled!')
            new_beta_sampling = beta_sampling*2
        elif urange < bs/beta_sampling:
            print('Found very small energy range beta_sampling is halfed!')
            new_beta_sampling = beta_sampling/2
        else:
            print('Found no particular reason of fitting failing --> beta_sampling is scaled randomly between 0.66 and 1.34!')
            new_beta_sampling = beta_sampling*np.random.uniform(0.66,1.34)
        return new_beta_sampling

    @staticmethod
    def MC_sample(data, setup, parsed_args, beta_sampling):
        
        max_mc_steps = 100000
        max_candidates_per_system = 40000
        
        kB = 0.0019872037514523
        beta_target = 1.0/(kB*parsed_args.Ttarget)
        
        sigma_init = parsed_args.sigma
        
        c = copy.deepcopy(data['coords'].to_numpy())
        
        init_data = copy.deepcopy(data[['at_type','sys_name','natoms','coords', 'bodies']])
        init_data['coords'] = c
        systems = np.unique(init_data['sys_name'])
        
        asymptotic_steps = 100
        candidate_data = pd.DataFrame()
        
        for sysname in systems:
            
            sys_data = init_data [ init_data['sys_name'] == sysname]
            
            step_data = copy.deepcopy(sys_data)
            
            al_help.evaluate_potential(step_data, setup,'opt')
            Uclass = step_data['Uclass'].to_numpy().copy()
            
            bs = setup.bS

            prop_sel = np.exp( - (Uclass - Uclass.min())/bs )
            prop_sel /= prop_sel.sum()

            all_indexes = np.array(step_data.index)
            idx_chosen = np.random.choice(all_indexes, size= min(len(step_data),100) , replace=False, p = prop_sel)
            step_data = step_data.loc[idx_chosen]

            al_help.evaluate_potential(step_data, setup,'opt')
            Uclass_prev = step_data['Uclass'].to_numpy().copy()
            
            n = len(step_data)
            
            times_scaled_beta = 0
            
            while times_scaled_beta < 10:
                
                print('MC trial = {:d} beta_sampling = {:4.6f} , system = {:s}'.format(times_scaled_beta, beta_sampling, sysname) )
                
                
                step, c_size ,  sigma, avg_accept_ratio = 0, 0 , sigma_init, 0.0
                candidate_data_sys = pd.DataFrame()
                
                while(step <= max_mc_steps and c_size <= max_candidates_per_system):
             
                    all_new_coords = []
                    old_coords = copy.deepcopy(step_data['coords'].to_numpy())
                    for j,dat in step_data.iterrows():
                        new_coords = al_help.petrube_coords(np.array(dat['coords']) ,sigma, 'random_walk', dat['bodies'])
                        all_new_coords.append(new_coords)
                    step_data.loc[step_data.index,'coords'] = all_new_coords
                     
                    al_help.evaluate_potential(step_data, setup,'opt')
                     
                    Uclass_new = step_data['Uclass'].to_numpy()
                     
                    dubt = (Uclass_new  - Uclass_prev )*beta_sampling
                     
                    pe =  np.exp( - dubt ) 
                    accepted_filter = pe > np.random.uniform(0,1,n) 
             
                 
                    not_accepted_filter = np.logical_not(accepted_filter)
                    step_data.loc[ not_accepted_filter, 'coords']  = old_coords[not_accepted_filter]
                     
                    Uclass_prev [ accepted_filter ] = Uclass_new [accepted_filter].copy()
                    Uclass_prev [ not_accepted_filter ] =  Uclass_prev [ not_accepted_filter].copy()
                    
                    accept_ratio = np.count_nonzero(accepted_filter)/n
                    avg_accept_ratio += accept_ratio
             
                    step += 1
                    AR = avg_accept_ratio/step
                    if AR < 0.2:
                         sigma*=0.97
                    elif AR > 0.5:
                         sigma/=0.97
                    sigma  = min( max(sigma,sigma_init*1e-1) , sigma_init*1e1)
                    if step %500 ==0:
                        print( 'MC step {:d},   sigma = {:.4e} A ,  accept_ratio = {:5.4f}  ,  current_accept = {:5.4f} candidate size = {:d}'.format(step,  sigma, AR, accept_ratio, c_size) )
                        sys.stdout.flush()
            
                    if step < asymptotic_steps:
                        continue
                    
                    filtered_step_data = step_data[ accepted_filter ]
                    
                    candidate_data_sys = pd.concat( (candidate_data_sys, filtered_step_data), ignore_index=True)
                    
                    c_size = len(candidate_data_sys)
                    ########


                print('Metropolis Hastings completed! Average acceptance {:5.4f}'.format( AR ) ) 

                u = candidate_data_sys['Uclass'].to_numpy()

                tfit, beta_eff, alpha, weights, l_minima, fail = al_help.estimate_Teff_Beff(u, nbins = 200) 
                
                print('MC trial = {:d} beta_target = {:5.4f} ,  beta_eff = {:5.4f}  ,  beta_sampling = {:5.4f}'.format (times_scaled_beta, beta_target, beta_eff, beta_sampling) )
                
                al_help.plot_candidate_distribution(u - u.min(), (beta_eff, alpha, weights, l_minima), 200,
                            title = f'MC trial {times_scaled_beta}' + r': Candidate distribution $\beta_{target}$' + '= {:5.4f},'.format(beta_target)+ r' $\beta_{eff}$' + ' = {:5.4f}'.format( beta_eff) + r' $\beta_{sampling}$' + ' = {:5.4f}'.format( beta_sampling),
                            fname=f'{setup.runpath}/CD{times_scaled_beta}_{sysname}.png')
                
                if fail:        
                    print(f'MC trial = {times_scaled_beta}: BETA SCALING FAILED! beta_eff = {beta_eff}   beta_target = {beta_target}    beta_sampling = {beta_sampling}\n Following empirical strategy')
                    beta_sampling = al_help.beta_distribution_fit_fail_strategy(u , setup, beta_sampling )
                    times_scaled_beta += 1
                    continue
                if np.abs ((beta_target - beta_eff)/beta_target) < 0.05 :
                    print(f'MC trial = {times_scaled_beta}: BETA SCALING CONVERGED! beta_eff = {beta_eff}   beta_target = {beta_target}   beta_sampling = {beta_sampling}')
                    break

                beta_sampling *= np.sqrt( beta_target/beta_eff)
                times_scaled_beta += 1
            
            candidate_data = candidate_data.append(candidate_data_sys,ignore_index=True)

        print('Metropolis Hastings completed! Average acceptance {:5.4f}'.format( c_size/(n*(step) ) ) )
        
        #raise  Exception('Debuging. Want to stop here')
        return candidate_data , beta_sampling

    @staticmethod
    def plot_candidate_distribution(u, fitting_params, bins, title = '', fname=None):
        
        u_sorted = np.sort(u)
        Pfit = al_help.joint_power_law_Boltzmann_distribution_multy_minima(u_sorted, *fitting_params)
        _ = plt.figure(figsize = (3.3,3.3), dpi=300)
        if title != '':
            plt.title(title, fontsize = 5.5)
        hist,bin_edges = np.histogram(u, bins=bins,density = True)
        plt.hist(u, bins=bins,density = True,label = 'candidates', color='blue')
        beta, alpha, wl, u_min_l = fitting_params
        lstyle =[':','-.','--']*5
        for w, ul, j in zip(wl, u_min_l, range(len(wl)) ):
            pl = w * al_help.joint_power_law_Boltzmann_distribution_multy_minima(u_sorted, beta, alpha, [1.0], [ul])
            plt.plot(u_sorted, pl, ls=lstyle[j], color='orange', label=r'$u_l$ at {:3.2f}'.format(ul), lw =0.75 )
        plt.plot(u_sorted, Pfit, ls ='-',label='fit', color='red')
        plt.legend(frameon=False, fontsize=7)
        if fname is not None:
            plt.savefig(fname, bbox_inches='tight')
        plt.close()
        #plt.show()
        return
    
    @staticmethod
    def joint_Boltzman_Gaussian_distribution(u, beta, cv, mu):
        P = (beta**2 / np.sqrt(2*np.pi*cv) ) * np.exp(-beta*u) * np.exp( - (beta**2/(2*cv) ) * (u -mu)**2 )
        norm = (beta/2) * np.exp(-beta*mu + cv/2) * ( erf( (beta*mu-cv)/np.sqrt(2*cv) ) + 1 )
        return P/norm

    @staticmethod 
    def Irecurr (beta, alpha, n ):
        ab = beta * alpha
        b4a = beta/(4.0*alpha)
        if b4a < 50:
            f0 = exp(b4a) * ( 1 - erf(sqrt(b4a)) ) 
        else:
            f0 = 1/sqrt(np.pi*b4a)
        I0 = 0.5 * sqrt(np.pi/ab) * f0 
        I1 = (1.0 - beta*I0)/(2*ab)
        ir = [I0, I1]
        for i in range(2, n+1):
            ir_i = ( (i-1)*ir[i-2] - beta*ir[i-1] ) / (2*ab)
            ir.append( ir_i )
        return tuple([np.float64(i) for i in ir])
    
    @staticmethod
    def joint_Boltzman_Maxwellian_distribution_multy_minima(u, beta, alpha, w_l=[1.0], min_u_l=[0.0]):
        I0, I1, I2 = al_help.Irecurr(beta, alpha, 2)
        sw = np.sum(w_l)
        P = 0.0
        for w,u_l in zip(w_l, min_u_l):
            P += (w/sw) * al_help.P(u - u_l ,beta, alpha)
        return P/I2
    @staticmethod
    def joint_power_law_Boltzmann_distribution_multy_minima(u, beta, alpha, w_l =[1.0], min_u_l = [0.0]):
        C = np.float64( gamma(alpha + 1.0)/power(beta,(alpha+1)) )
        
        sw = np.sum(w_l)
        P = 0.0
        for w,u_l in zip(w_l, min_u_l):
            normalizing_factor = C*np.exp(-beta*u_l)
            pu = (w/sw) * np.power( (u-u_l), alpha) * np.exp( - beta*u ) / normalizing_factor
            pu [ u <u_l]  = 0.0
            P+=pu

        return P

    @staticmethod
    def P(u, beta, alpha):
        pu = u**2 * np.exp(-beta*u) * np.exp(-alpha*beta*u**2)
        pu[ u <0 ] = 0.0
        return pu

    @staticmethod
    def joint_Boltzman_Maxwellian_distribution(u, beta, alpha):
        I0, I1, I2 = al_help.Irecurr(beta, alpha, 2)

        P = u**2 * np.exp(-beta*u) * np.exp(-alpha*beta*u**2)
        return P/I2

    @staticmethod
    def find_distribution_parameters(u, nminima=0,nbins=200):
        u = u - u.min()
        mu = np.mean(u)
        
        params = [ 1.0, 0.0033 ] # beta, alpha
        bounds = [ [0.02,40], [0.003,8.004]] 
        if nminima > 0:
            for _ in range(nminima +1):
                # adding the weights
                params.append(0.5)
                bounds.append([0,1.0])
            umax = u.max()
            for j in range(nminima):
                # adding initializations about the minima
                params.append( np.random.uniform(0,1))
                bounds.append([0,umax])

        params = np.array(params)
        
        kB = 0.00198720375145233
        
        def get_params( params, n_l):
            beta, alpha = params[0], params[1]
            if n_l>0:
                n_w = n_l + 1
                n_w_e = n_w + 2
                w_l =  params[2:n_w_e]
                min_u_l = np.array([0.0, *list( params[ n_w_e : n_w_e + n_l  ] )] )
            else:
                w_l = [1.0]
                min_u_l = [ 0.0 ]
            return beta, alpha, w_l, min_u_l
        
        def reg_cost(params, n_l):
            c2 = 0.0 
            c1 = 0.0
            
            if n_l >0:
                w_l = params[2:3+n_l]
                w = w_l/w_l.sum()
                nw = w.shape[0]
                for i in range(nw):
                    #c1 += w[i]**2
                    for j in range(i+1,nw):
                        c2 += w[i]*w[j]
                        for k  in range(j+1,nw):
                            c1 += w[i]*w[j]*w[k]
                c2 /= math.perm(nw,2)
                if nw>2:
                    c1 /= math.perm(nw,3)
            return 0.2*(c2 + c1)

        def weights_to_one(params, n_l):
            w = params[2:3+n_l]
            return w.sum() -1

        def cost_BG(params, dens ,bc, n_l):
            c = cost_distribution_fit(params, dens, bc, n_l) 
            creg = reg_cost(params,n_l)
            return c + creg
        
        def cost_distribution_fit(params, dens, bc, n_l):
            beta, alpha, w_l, min_u_l = get_params( params, n_l)
            ps = al_help.joint_power_law_Boltzmann_distribution_multy_minima(bc,beta, alpha, w_l, min_u_l)
            return 100*np.sum( np.abs(ps - dens ) ) /ps.shape[0]
        
        dens, bin_edges = np.histogram(u, bins=nbins, density=True)
        
        bc = bin_edges[0 : -1] - 0.5*(bin_edges[1]-bin_edges[0])
        args = (dens , bc, nminima)
        
        #res = dual_annealing(cost_BG, bounds, args = args,  initial_temp=15000, maxiter=3500, restart_temp_ratio=2e-04)
        
        best_cost = 1e16
        t0 = perf_counter()
        for ntry in range(50):
            res = minimize(cost_BG, params,args = args, bounds=bounds,tol=1e-4,  method = 'SLSQP', constraints = {'type':'eq','fun':weights_to_one, 'args': (nminima,) } )
            if res.fun < best_cost:
                best_cost = res.fun
                best_trial = ntry
                best_params = res.x.copy()
            params = np.array([ np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(params.shape[0]) ] )
        tf = perf_counter() - t0
        print('SLSQP best trial {:d} , finished in {:.3e} sec costf = {:5.4f}'.format(best_trial,tf, best_cost))
        beta, alpha, w_l, min_u_l = get_params( best_params, nminima)
        
        cfit = cost_distribution_fit(best_params , dens, bc, nminima)
        reg = reg_cost(best_params, nminima)
        cv_eff = u.var()*kB*beta**2
        print ('bins = {:d}, beta = {:5.4f}, alpha = {:5.4f}  costf = {:8.7f} reg_cost = {:8.7f} cv_histogram = {:5.4f} kcal/mol/K'.format(nbins, beta, alpha, cfit, reg,  cv_eff))
        w_l = np.array(w_l)/np.sum(w_l)
        
        al_help.plot_candidate_distribution(u, (beta, alpha, w_l, min_u_l), nbins,
                title = f'nl  {nminima}:' + r'$\beta_{eff}$ =' + '{:5.4f}'.format(beta),
                fname=f'Results/nl{nminima}.png')
        return (beta, alpha, w_l, min_u_l), cfit

    @staticmethod
    def estimate_Teff_Beff(u, nbins = 200):
        u = u -u.min()
        dens, bin_edges = np.histogram(u, bins=nbins, density=True)
        std = np.sqrt(dens/u.shape[0]).sum()/nbins*100
        print('Standard error of the data = {:8.6f}'.format(std))
        rel_err = 1e16
        old_err = 1e16
        n_l = 0
        def print_minima(weights, l_minima):
            for wl, ul in zip(weights, l_minima):
                print('found minima with weight {:5.4f} at {:5.4f}'.format(wl, ul) )
            sys.stdout.flush()
            return

        while( n_l<10 ):
            p,  new_err  = al_help.find_distribution_parameters(u, nminima=n_l, nbins = nbins)
            beta, alpha, weights, l_minima = p
            rel_err = (old_err - new_err)/old_err
            fail = old_err > std*100
            print('number of local minima {:d} , rel error = {:5.4f}'.format(n_l, rel_err) )
            
            print_minima (weights, l_minima) 
            if rel_err <0.1:
                break
            if new_err < 1*std:
                pold = p ; old_err = new_err
                break
            
            n_l += 1
            old_err = new_err
            pold = p
        
        beta, alpha, weights, l_minima = pold
        
        print(' .... Solution ....'*5)
        print('beta = {:5.4f} alpha = {:5.4f}'.format(beta,alpha) )
        
        print_minima (weights, l_minima) 
        
        kB = 0.00198720375145233
        #I0, I1, I2, I3, I4 = al_help.Irecurr(beta, alpha, 4)
        #for j,i in enumerate([ I0, I1, I2, I3, I4]):
        #    print(f'I{j} --> {i}')
        cv = (alpha+1) *kB

        print('Effective cv = {:7.6f} kcal/mol/K'.format(cv))
        print('Fitting error = {:3.2f} std'.format(old_err/std) )
        print(' .................'*5)
        sys.stdout.flush()

        Teff = 1.0/(beta*kB)
        return Teff, beta, alpha, weights, l_minima,  fail
    
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
        return 

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
    def evaluate_potential(data,setup, which='init'):
        al_help.make_interactions(data,setup)
        
        
        #train_indexes, dev_indexes = Data_Manager(data,setup).train_development_split()
        indexes = data.index
        setup.optimize = False

        optimizer = FF_Optimizer(data,indexes, indexes, setup)
        #print('dataevaluations')
        #results
        #optimizer.optimize_params()
        optimizer.set_UFclass_ondata(which = which,dataset='all')
        

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
        al_help.evaluate_potential(data,setup,'init')
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
                s = 'pair_coeff * * local/density frhos.ld \n'
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
        
        optimizer = al_help.evaluate_potential(data,setup,'init')
        
        optimizer.set_models('init', 'opt')
        
        optimizer.set_results()
        
        predicted_costs = optimizer.current_costs

        dataeval = Interfacial_Evaluator(data,setup,prefix='all')
        #print(data)
        #funs = ['MAE','relMAE','BIAS','STD','MSE']
        funs = ['MAE','MSE']
        
        cols = ['sys_name']
        df = cols + ['MAE','MSE']
        if setup.costf not in funs: funs.append(setup.costf)
        
        trev = dataeval.make_evaluation_table(funs,cols,save_csv='predict.csv')
        print('prediction errors \n----------\ns',trev[df])
        
        E = data['Energy'].to_numpy()
        U = data['Uclass'].to_numpy()
        dataeval.plot_predict_vs_target(E, U, path = setup.runpath,title='prediction dataset',
                                          fname='predict.png',size=2.35,compare='sys_name')
        
        forces_true = [] ; forces_class = []
        for ftrue, fclass in zip(data['Forces'].values, data['Fclass'].values):
            for ft,fc in zip(ftrue,fclass):
                for x,y in zip(ft,fc):
                    forces_true.append(x)
                    forces_class.append(y)
        
        dataeval.plot_predict_vs_target(np.array(forces_true),np.array(forces_class),
                                        path = setup.runpath,title='Prediction dataset Forces',
                                          fname='predictForces.png',size=2.35,
                                          xlabel=r'$F^{dft}$ (kcal/mol/$\AA$)',ylabel=r'$F^{class}$ (kcal/mol/$\AA$)')
        
        dataeval.plot_eners(subsample=1,fname='predicteners.png')
       
        return  predicted_costs

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
        
               
        optimizer = FF_Optimizer(data,train_indexes, dev_indexes,setup)
        
        method = setup.training_method
        
        min_per_system = np.min( [ np.count_nonzero( data['sys_name'] == name ) for name in np.unique(data['sys_name']) ] )
        ndata = len(data)
        if ndata < 100 or min_per_system < 10:
            method = 'scan_lambda_force'
            print(f'Setting method to scan_lambda_force: ndata = {ndata}, min_per_system = {min_per_system}')
        if not setup.optimize:
          optimizer.set_models('init','opt')
          optimizer.set_results()
        elif method =='scan_lambda_force':
            optimizer.pareto_via_scan()
        elif method == 'scan_force_error':
            optimizer.pareto_via_constrain()
        elif method=='fixed_lambda':
            if not ( 0 < setup.lambda_force < 1.0):
                raise ValueError(f'lambda_force {setup.lambda_force} must be between 0 and 1 ')
            optimizer.optimize_params()
        else:
            raise Exception(f'method "{method}"')
        optimizer.report()
        #print('evaluations')
        #results
        
        if not setup.optimize:
            optimizer.set_models('init','opt')
        optimizer.set_UFclass_ondata(which='opt',dataset='all')
        
        setup.write_running_output()
        
        funs = ['MAE','MSE','MAX','BIAS','STD']

        #df = cols + ['relMAE','relMSE','relBIAS','MAE','MSE']
        if setup.costf not in funs: funs.append(setup.costf)
        

        train_eval = Interfacial_Evaluator(data.loc[train_indexes],setup,prefix='train')
        dev_eval = Interfacial_Evaluator(data.loc[dev_indexes],setup,prefix='development')
        
        dtrain = data.loc[train_indexes]
        E = dtrain['Energy'].to_numpy()
        U = dtrain['Uclass'].to_numpy()
        
        
        train_eval.plot_predict_vs_target(E,U, path = setup.runpath,title='train dataset',
                                          fname='train.png',size=2.35,compare='sys_name')
        
        forces_true = [] ; forces_class = []
        for ftrue, fclass in zip(dtrain['Forces'].values, dtrain['Fclass'].values):
            for ft,fc in zip(ftrue,fclass):
                for x,y in zip(ft,fc):
                    forces_true.append(x)
                    forces_class.append(y)
        
        train_eval.plot_predict_vs_target(np.array(forces_true),np.array(forces_class),
                                        path = setup.runpath,title='Training dataset Forces',
                                          fname='trainingForce.png',size=2.35,
                                          xlabel=r'$F^{dft}$',ylabel=r'$F^{class}$')
        
        ddev = data.loc[dev_indexes]
        Edev = ddev['Energy'].to_numpy()
        Udev = ddev['Uclass'].to_numpy()
        
        dev_eval.plot_predict_vs_target(Edev,Udev,path = setup.runpath,title='development dataset',
                                          fname='development.png',size=2.35,compare='sys_name')
        
        forces_true = [] ; forces_class = []
        for ftrue, fclass in zip(ddev['Forces'].values, ddev['Fclass'].values):
            for ft,fc in zip(ftrue,fclass):
                for x,y in zip(ft,fc):
                    forces_true.append(x)
                    forces_class.append(y)
                    
        dev_eval.plot_predict_vs_target(np.array(forces_true),np.array(forces_class),
                                        path = setup.runpath,title='development dataset Forces',
                                          fname='developmentForce.png',size=2.35,
                                          xlabel=r'$F^{dft}$ (kcal/mol/$\AA$)',ylabel=r'$F^{class}$ (kcal/mol/$\AA$)')
        
        train_eval.plot_eners(subsample=1,path=setup.runpath,fname='eners.png')

        fe_ty = []
        for model in setup.init_models.values():
            fe = model.feature
            ty = model.type
            c = (fe, ty)
            if c in fe_ty:
                continue
            else:
                fe_ty.append(c)
                dataMan.plot_discriptor_distribution(ty,fe)

        dataMan.save_selected_data(setup.runpath+'/frames.xyz', data,labels=['sys_name','Energy'])
        
        setup.plot_models(which='opt')
        
        tf =perf_counter() - t0
        print('Time consumed for solving the model --> {:4.3e} min'.format(tf/60))
        return  data, optimizer.current_costs , optimizer.setup, optimizer

    @staticmethod
    def make_random_petrubations(data,nper=10,sigma=0.05):
        candidate_data= pd.DataFrame()
        for j,dat in data.iterrows():
            for k in range(nper):
                d = dat.copy()
                new_coords = al_help.petrube_coords( np.array(dat['coords']) ,sigma,'atoms',dat['bodies'])
                d['coords'] = new_coords
                candidate_data = pd.concat([candidate_data, d.to_frame().T], ignore_index=True)
        return candidate_data
    
    @staticmethod
    def petrube_coords(coords,sigma, method, bodies = {} ):
        
        c = coords.copy()
        if method == 'atoms':
            c += np.random.normal(0,sigma,c.shape)
        elif method =='rigid':
            for j,body in bodies.items():
                b = body
                c[b] = al_help.rottrans_randomly(c[b],sigma)
        elif method =='random_walk':
            idx = np.random.randint(0,c.shape[0])
            r_move =  np.random.normal(0 , sigma,3) 
            c[idx] += r_move
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
    def random_selection(data , setup,candidate_data,batchsize, method='random'):
        al_help.make_interactions(candidate_data,setup) 
        candidate_data = al_help.clean_well_separated_nanostructures(candidate_data,setup)
        Ucls = candidate_data['Uclass'].to_numpy()
        n = len(candidate_data)
        indx = np.arange(0,n,1,dtype=int)
        if n<=batchsize:
            batchsize=min(n,batchsize)
        col = data['sys_name'].to_numpy()
        colvals = np.unique(col)
        Props = [1/np.count_nonzero(col == c) for c in colvals]
        Props = np.array(Props)/np.sum(Props)

        names = np.random.choice(  colvals,size=batchsize,replace=True,p = Props  )
        nums = {name: np.count_nonzero(names==name) for name in colvals} 
        selected_data = pd.DataFrame()
        ix = candidate_data.index
        for name,num in nums.items():
            
            fsystem = candidate_data['sys_name'] == name 
            ix_f = ix [fsystem]
            if method == 'random':
                psel = None
            elif method =='control_energy':
                u_f = Ucls [ fsystem]
                psel = np.exp (- (u_f - u_f.min())/setup.bS)
                psel /= psel.sum()
            elif method == 'histogram_uncertainty':
                psel = al_help.find_histogran_uncertainty(candidate_data [fsystem] )

            nx = len(ix_f)
            ix_sel = np.random.choice (ix_f, size=min(num, nx),replace=False, p = psel)

            selected_data = selected_data.append( candidate_data.loc[ix_sel] , ignore_index=True)
        return selected_data


    @staticmethod
    def disimilarity_selection(data,setup,candidate_data,batchsize,method='sys_name'):
        al_help.make_interactions(candidate_data,setup) 
        candidate_data = al_help.clean_well_separated_nanostructures(candidate_data,setup)
        dis = []
        for k1,d1 in candidate_data.iterrows():
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
        candidate_data['disimilarity'] = dis
        fcheck = np.isnan(candidate_data['disimilarity'])
        for j,d in candidate_data[fcheck].iterrows():
            print('Warning: Removing nan values', j,d['sys_name'],d['disimilarity'])
            sys.stdout.flush()
        candidate_data = candidate_data[~fcheck]
        print('evaluating in selection step with nld = {:d}'.format(setup.nLD))
        candidate_data['Energy']=np.zeros(len(candidate_data))
        al_help.evaluate_potential(candidate_data,setup,'opt')

        candidate_data['Prob. select'] = candidate_data['disimilarity']/dis.sum()
        
        Ucls = candidate_data['Uclass'].to_numpy()
        n = len(candidate_data)
        indx = np.arange(0,n,1,dtype=int)
        if n<=batchsize:
            batchsize=min(n,batchsize)
            method = None

        if method is None:
            chosen = np.random.choice(indx,size=batchsize,replace=False,p=candidate_data['Prob. select'])
        else:
            col = candidate_data[method].to_numpy()
            colvals = np.unique(col)
            Props = [1/np.count_nonzero(col == c) for c in colvals]
            Props = np.array(Props)/np.sum(Props)

            names = np.random.choice(  colvals,size=batchsize,replace=True,p = Props  )
            nums = {name: np.count_nonzero(names==name) for name in colvals} 
            
            chosen = []
            psel = candidate_data['Prob. select'].to_numpy()
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
        candidate_data['select'] = select

        return  candidate_data [ candidate_data ['select'] ] 

    @staticmethod
    def write_gjf(path,fname,atom_types,coords,
                  multiplicity=1):
        file = '{:s}/{:s}'.format(path,fname)
        #lines = ['%nprocshared=16\n%mem=16000MB\n#p wb97xd/def2TZVP scf=xqc scfcyc=999\n\nTemplate file\n\n']    
        lines = ['%nprocshared=4\n%mem=16000MB\n#p wb97xd/def2TZVP scf(xqc) scfcyc=999 force\n\nTemplate file\n\n']    
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
        "#SBATCH --ntasks-per-node=4  ",
        "#SBATCH --partition=milan  ",
        "#SBATCH --time=2:00:00  ",
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
    def log_to_xyz(input_path,output_path,read_forces=True):
        GeneralFunctions.make_dir(output_path)
        
        for fname in [x for x in os.listdir(input_path) if x[-4:] =='.log'] :
            try:
                data = Data_Manager.read_Gaussian_output('{:s}/{:s}'.format(input_path,fname),
                        read_forces=read_forces)
            except Exception as ve:
                print('warning: DFT null data --> {}'.format(ve))
                continue
         
            labels = [c for c in data.columns if c not in ['coords','natoms','at_type','filename', 'Forces'] ]
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
            exrd = data.loc[ existing_indexes ]
        return nexrd, exrd
    
    
    @staticmethod
    def write_errors(model_costs, num, prefix='' ):
        errfile = prefix + 'COSTS.csv'
        if os.path.exists(errfile):
            with open(errfile,'r') as f:
                nl = len(f.readlines())
                if nl ==0:
                    write_header=True
                else:
                    write_header=False
            f.closed
        else:
            write_header=True
          
        head_line = ','.join(['AL_iteration'] +  list(model_costs.__dict__.keys())  )
        values_line = ','.join([ str(num) ] + ['{:4.6e}'.format(x) for x in  list(model_costs.__dict__.values()) ]  )

        with open(errfile,'a') as f:
            if write_header:
                f.write(f'{head_line}\n')
        
            f.write(f'{values_line}\n')
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
    def clean_well_separated_nanostructures(data, setup):
        t0 = perf_counter()
        rc = setup.rclean
        keep_index = []
    
        for j, df in data.iterrows():
            coords = np.array(df['coords'])  # Ensure it's a NumPy array
            natoms = df['natoms']
            cluster_ids = {0}
            cluster_size_old = len(cluster_ids)
            cluster_size = 0
            
            while cluster_size_old != cluster_size:
                cluster_size_old = len(cluster_ids)
                new_ids = set(cluster_ids)  # Copy the set to avoid modification during iteration
                for i in cluster_ids:  # Iterate over a copy
                    rref = coords - coords[i]
                    distances = np.sqrt(np.sum(rref * rref, axis=1))
                    neibs = np.where(distances < rc)[0]
                    new_ids |= set(neibs)
                cluster_ids = new_ids  # Update the set after the loop
                cluster_size = len(cluster_ids)

            if cluster_size == natoms:
                keep_index.append(j)
        
        n = len(data)
        ncleaned = n - len(keep_index)
        print (f'Cleaned {ncleaned}/{n} candidate data due to separated nanostructure (r_clean = {rc})' + ' time needed --> {:.3e} sec'.format(perf_counter()-t0))
        sys.stdout.flush()
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
        data = data.loc[ix] 
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
        r = r1 - r2
        d = np.sqrt(np.dot(r,r))
        return d
    
    @jit(nopython=True,fastmath=True)
    def calc_unitvec(r1,r2):
        r = r1 - r2
        d = np.sqrt(np.dot(r,r))
        return r/d
    
    @jit(nopython=True,fastmath=True)
    def calc_angle_pa_pc(ri,rj,rk):
        vij = ri - rj 
        vkj = rk - rj
        
        a = np.dot(vij,vkj)
        b = np.sqrt( np.dot(vij,vij) ) 
        c = np.sqrt( np.dot(vkj,vkj) ) 
        
        
        fi = vkj/(b*c) - a*vij/(c*b**3)
        fk = vij/(b*c) - a*vkj/(b*c**3)
        
        cth = a/(b*c)
        
        dth_dcth = -1/max(np.sin(np.arccos(cth)),1.49e-8)
        
        fi *= dth_dcth
        fk *= dth_dcth
        return fi, fk
    
    @jit(nopython=True,fastmath=True)
    def calc_angle(r1,r2,r3):
        d1 = r1 -r2 ; d2 = r3 - r2
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
    
    def find_gradient(self):
        
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
    
    def find_derivative_gradient(self):
        
        r0, k1, k2, k3 = self.params
        r = self.r
        r_r0 = r - r0
        r_r0m2 = r_r0*r_r0
        r_r0m3 = r_r0m2*r_r0
        
        fg = np.empty((4,r.shape[0]), dtype=np.float64)
        
        
        fg[0] = - (2*k1 + 6*k2*r_r0 + 12*k3*r_r0m2)
        fg[1] = 2*r_r0 
        fg[2] = 3*r_r0m2 
        fg[3] = 4*r_r0m3 
        
        self.derivative_gradient = fg
        
        return fg


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
    
    def find_gradient(self):
        
        r0, k = self.params
        r = self.r
        
        g = np.empty((2,r.shape[0]), dtype=np.float64)
        r_r0 = r - r0
        g[0] = -2*k*(r_r0)
        g[1] = r_r0 * r_r0
        
        self.params_gradient = g
        return g

    def find_derivative_gradient(self):
        
        r0, k = self.params
        r = self.r
        
        fg = np.empty((2,r.shape[0]), dtype=np.float64)
        r_r0 = r - r0
        fg[0] = -2*k
        fg[1] = 2*r_r0 
        
        self.derivative_gradient = fg
        return fg


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
       
    def find_gradient(self):
        sigma, epsilon = self.params
        r = self.r
        
        so_r6 = (sigma/r)**6
        so_r12 = so_r6*so_r6
        
        g = np.zeros((2,r.shape[0]))
        
        g[0] = 4 * epsilon * (12*so_r12  - 6*so_r6 )/sigma 
        g[1] = 4 * (so_r12 - so_r6) 
        
        self.params_gradient = g
        return g

    def find_derivative_gradient(self):
        sigma, epsilon = self.params
        r = self.r
        
        so_r6 = (sigma/r)**6
        so_r12 = so_r6*so_r6
        
        fg = np.zeros((2,r.shape[0]))
        
        fg[0] = 144 * epsilon * (so_r6  - 4*so_r12 )/sigma/r 
        fg[1] = 24 * (so_r6 -2*so_r12)/r 
        
        self.derivative_gradient = fg
        return fg
    
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
       
    def find_gradient(self):
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
    
    def find_derivative_gradient(self):
        r = self.r
        x = self.params
        re, De, alpha = x
        nr = r.shape[0]
        n = self.params.shape[0]
        
        fg = np.zeros((n,nr))
        
        r_re = r-re
        t1 = -alpha*r_re
        e1 = np.exp(t1)
        e2 = np.exp(2*t1)
        rr = (e2  - e1 )
        r2r = (2*e2-e1)
        fg[0] = -2 *  alpha * alpha * De  * r2r  #d^2udrdre
        fg[1] = -2 * alpha * rr # d^2udrDe
        fg[2] = 2 * De * ( alpha*r_re*r2r - rr  ) # d^22udrdalpha
        
        self.derivative_gradient = fg
        return fg
    
    
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
       
    def find_gradient(self):
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
    
    def find_derivative_gradient(self):
        r = self.r
        x = self.params
        re, De, alpha = x
        nr = r.shape[0]
        n = self.params.shape[0]
        
        fg = np.zeros((n,nr))
        
        r_re = r-re
        t1 = -alpha*r_re
        e1 = np.exp(t1)
        e2 = np.exp(2*t1)
        rr = (e2  - e1 )
        r2r = (2*e2-e1)
        fg[0] = -2 *  alpha * alpha * De  * r2r  #d^2udrdre
        fg[1] = -2 * alpha * rr # d^2udrDe
        fg[2] = 2 * De * ( alpha*r_re*r2r - rr  ) # d^22udrdalpha
        
        self.derivative_gradient = fg
        return fg

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
    
    def find_gradient(self):
        
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
    
    def find_derivative_gradient(self):
        
        taus = self.taus
        M = self.M
        y = self.ycontrol
        n = self.npoints
        
        if not hasattr(self,'dC'):
            dC = self.find_dC_vectorized()
        else:
            dC = self.dC 
        if not hasattr(self,'dydt'):
           _ = self.find_dydx()
        
        taus_power = self.taus_power
        coeff_tj = np.zeros((n,))
        
        for i in range(n):
            ry = y[i]
            for j in range(i,n):
                mij = M[i,j]
                coeff_tj[j] += ry * mij * j * (j-1)
        
        g0_1 = np.dot(taus_power[0:-2].T,coeff_tj[2:])   
        
        self.d2ydt2 = g0_1
        
        
        nt = taus.shape[0]
        fg = np.zeros((self.params.shape[0],nt))
        L = self.L
        
        fg[0] = ( -1/(L*L) )*(self.dydt + self.taus*g0_1) # dydt*dtdL
        
        fg[1] = (dC[0] + dC[1] + dC[2])/L
        fg[-1] = (dC[-1] + dC[-2] + dC[-3])/L
        
        fg[2:-1] = dC[3:-3]/L
        
        self.params_gradient = fg
        
        return fg
    
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
   
    
    def find_dC_vectorized(self):
        
        M = self.M
        if not hasattr(self,'taus_power'):
            self.find_taus_power()
        taus_power = self.taus_power
        
        j = np.arange(1, self.npoints)  # Create an array of j indices (1-based)
        # Create a mask to determine valid (i, j) pairs where j >= i
        
        C =  np.dot(M[:,j]*j,taus_power[j-1]) 
        self.dC = C

        return C 
   
    def find_dC_serial(self):
        n = self.npoints
        M = self.M
        if not hasattr(self,'taus_power'):
            self.find_taus_power()
            
        taus_power = self.taus_power
        nt = self.taus.shape[0]
        C = np.zeros((n, nt))
        
        for i in range(n):
            for j in range(i,n):
                C[i] += j*M[i,j]*taus_power[j-1]
        self.dC = C
        return C 
   
    def find_dydyc_serial(self):
        n = self.npoints
        M = self.M
        taus = self.taus
        nt = taus.shape[0]
        
        if not hasattr(self,'taus_power'):
            self.find_taus_power()
            
        taus_power = self.taus_power
        
        
        C = np.zeros((n, nt))
        
        
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
        
        bs = [ copy.deepcopy(self.b) for _ in range(Nt)]
        
        t0 = perf_counter()
        for b,_ in zip(bs,range(Nt)):
            _ = b.find_gradient()
        tf = 1000*(perf_counter() - t0 ) - t_overhead
        
        times['grads'] = tf/Nt 
        
        bs = [ copy.deepcopy(self.b) for _ in range(Nt)]
        
        t0 = perf_counter()
        for b,_ in zip(bs,range(Nt)):
            _ = b.find_dydx()
        tf = 1000*(perf_counter() - t0 ) - t_overhead
        
        times['dydx'] = tf/Nt 
        
        bs = [ copy.deepcopy(self.b) for _ in range(Nt)]
        
        t0 = perf_counter()
        for b,_ in zip(bs,range(Nt)):
            _ = b.find_derivative_gradient()
        tf = 1000*(perf_counter() - t0 ) - t_overhead
        
        times['grads_dydx'] = tf/Nt 
        
        if verbose:
            line = "".join(["{:s}  --> {:4.3e} ms\n".format(k,v) for k,v in times.items() ]) 
            print(" Npoints: {:d}\n ------\n {:s} ".format(u.shape[0],line))
        return times
    
    def vectorization_scalability(self,Nt=30,verbose=False,plot=True):
        x0 = 1e-8
        if hasattr(self.b,"L"):
            L = self.b.L
        else:
            L=10.0
        xe =L+x0
        func = [] ; grads = [] ; dydx = [] ; grads_dydx = []
        Npoints = []
        for dx in [1.0,1e-1,1e-2,1e-3,1e-4,1e-5]:
            xvals = np.arange(x0,xe,dx)
            Npoints.append(xvals.shape[0])
            
            self.b = self.f(xvals, self.params,*self.fargs,**self.fkwargs)
            
            times = self.time_cost(Nt,verbose)
            
            func.append(times['func'])
            grads.append(times['grads'])
            dydx.append(times['dydx'])
            grads_dydx.append(times['grads_dydx'])
        func = np.array(func)
        grads = np.array(grads)
        dydx = np.array(dydx)
        grads_dydx = np.array(grads_dydx)
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
            plt.plot(Npoints, grads_dydx/Npoints,marker='*', 
                     label='grads_dydx',ls='-.')
            plt.legend(fontsize=6,frameon=False)
            plt.show()
        return 
    
    def derivative_gradient_check(self, epsilon=1e-4,tol=1,plot=False,verbose=False):
        tol=tol*epsilon
        b = self.b
        params = self.params.copy()
        vals = self.vals
        filt = self.filt
        
        passed = True
        
        g = b.find_derivative_gradient()
        for i in range(params.shape[0]):
            # Create copies of params and perturb only the i-th parameter for 4th order central difference
            p1p, p1m = params.copy(), params.copy()
            p1p[i] += epsilon
            p1m[i] -= epsilon
            p2p, p2m = params.copy(), params.copy()
            p2p[i] += 2 * epsilon
            p2m[i] -= 2 * epsilon
    
            # Compute function values at perturbed points
            du1p = self.f(vals, p1p, *self.fargs, **self.fkwargs).find_dydx()
            du1m = self.f(vals, p1m, *self.fargs, **self.fkwargs).find_dydx()
            du2p = self.f(vals, p2p, *self.fargs, **self.fkwargs).find_dydx()
            du2m = self.f(vals, p2m, *self.fargs, **self.fkwargs).find_dydx()
            
            # 4th-order central difference approximation
            g_num = (-du2p + 8 * du1p - 8 * du1m + du2m) / (12 * epsilon)
    
            # Compute the max difference between numerical and analytical gradients
            diff = np.abs(g_num[filt] - g[i][filt]).max()
            if verbose:
                print("parameter {:d} --> max difference {:4.3e}".format(i, diff))
            
            if plot:
                _ = plt.figure(figsize=(3.3,3.3),dpi=250)
                plt.title(f'Derivative Gradient of x_{i}')
                plt.plot(vals[filt], g_num[filt], label='numerical')
                plt.plot(vals[filt], g[i][filt], label='analytical',ls='--')
                plt.legend(fontsize=6,frameon=False)
                plt.show()
            if diff > tol:
                
                print('Derivative Gradient Test not passed')
                passed = False
        if passed:
            print('Derivative Gradient check ok!')
        return
    
    
    def gradient_check(self, epsilon=1e-4,tol=1,plot=False,verbose=False):
        tol=tol*epsilon
        b = self.b
        params = self.params.copy()
        vals = self.vals
        filt = self.filt
        passed = True
        
        g = b.find_gradient()
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
                passed =False
                print('Gradient Test not passed')
        if passed:
            print('Gradient check ok!')
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
        'storing_path':'Results',
        'run': '0',
        'runpath_attributes':['run'],
        
        'max_ener':1.0,
        'rclean':6.0,
        'max_force':0.003,
        'clean_perc':0.8,
        'bC':30.0,
        'bS':15.0,
        
        'optimization_method':'SLSQP',
        'opt_disp':True,
        'optimize':True,
        'costf':'MSE',
        
        'training_method':'scan_force_error',
         'random_initializations': 2,
         'npareto':15,
        'lambda_force':0.5,
        
        'normalize_data':True,
        
        'regularization_method': 'ridge',
        'reg_par': 1e-6,
        
        'maxiter':300,
        'max_moves':10,
        'increased_stochasticity':0.2,
        'SLSQP_batchsize':100000,
        'tolerance':1e-5,
        
        'train_perc':0.8,
        'sampling_method':'random',
        'seed':1291412,

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
        'w':1.0,
        'bT':15.0,

        'nLD':1,
        'nPW':2,
        'nBO':2,
        'nAN':2,

        'rho_r0' : 0.1,
        'rho_rc': 5.5,
        
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
                raise Exception('InputError: Uknown input variable "{:s}"'.format(attrname))
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
                    #print(model.name, model.parameters)
                    b = fu(r,model.parameters,*model.model_args)
                    u = b.u_vectorized()
                    if 'PW1' in model.name:
                        pass
                        #print(u.min(),u.max())
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
            elif ti is float: s+='{:7.8f} '.format(v)
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
            add_empty_line = ['runpath_attributes','bS','costf','lambda_force','normalize_data','reg_par',
                              'tolerance','seed','recombination','visit','bT','nAN','rho_rc'] 
            
            for i,(k,v) in enumerate(self.defaults.items()):
                var = getattr(self,k)
                if k in self.executes:
                    var = str(var)
                write(file,k,var)
                if k in add_empty_line: 
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
            du = np.zeros(r.shape[0],dtype=float)
            for model in models_of_ty:
                fobj = model.function(r,model.parameters,*model.model_args)
                u += fobj.u_vectorized()
                du += - fobj.find_dydx()
            
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
            du = np.zeros(r.shape[0],dtype=float)
            for model in models_of_ty:
                fobj = model.function(r,model.parameters,*model.model_args)
                u += fobj.u_vectorized()
                du += - fobj.find_dydx()

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

            cobj = model.function(r,model.parameters,*model.model_args)
            u = cobj.u_vectorized()
            du = - cobj.find_dydx()
            
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
            #num_pars = model.number_of_parameters

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
            
            cobj = model.function(rhov,model.parameters,*model.model_args)
            Frho = cobj.u_vectorized()
            #du = cobj.find_dydx()
            
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
        #self.calc_neibs_lists()
        #self.calc_rhats()
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
                        
                        r  = np.empty(npairs, dtype=float)
                        partial_ri = np.empty( (npairs,3), dtype=float)
                        i_index = np.empty(npairs,dtype=int)
                        j_index = np.empty(npairs,dtype=int)
                                    
                        for ip,p in enumerate(pairs):
                            i, j = p
                            
                            r1 = np.array( ac[i] ) 
                            r2 = np.array( ac[j] )
                            
                            i_index[ip] = i
                            j_index[ip] = j
                            r[ip] =  VectorGeometry.calc_dist(r1,r2)
                            partial_ri[ip] =  VectorGeometry.calc_unitvec(r1,r2)
                       
                        temp = {'values':r, 'partial_ri':partial_ri,
                                'i_index':i_index,'j_index':j_index}
                        
                    elif intertype=='angles':
                        npairs = len(pairs) 
                        
                        angles  = np.empty(npairs, dtype=float)
                        pa = np.empty( (npairs,3), dtype=float)
                        pc = np.empty( (npairs,3), dtype=float)
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
                            
                            pa[ip], pc[ip] = VectorGeometry.calc_angle_pa_pc(r1,r2,r3)
                            angles[ip] =  VectorGeometry.calc_angle(r1,r2,r3)
                        
                        temp = {'values':angles, 'pa':pa, 'pc': pc,
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
                            
                        temp = {'values':dihedrals, 'ra':ra, 'rd': rd,
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
                        temp = {'values':rhos,'n_central':n_rhos , 
                                'v_ij':v_ij,'to_pair_index':to_pair_index,
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
            plt.hist(self.data[col][f], bins=200, density=True, color='magenta')
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
                    fa = data['Forces']
                    fa_ex=True
                except:
                    fa_ex=False
                try:
                    comment = data['comment']
                except:
                    comment = ''
                f.write('{:d} \n{:s}\n'.format(na,comment))
                if fa_ex == False:
                    for j in range(na):
                        f.write('{:3s} \t {:8.8f} \t {:8.8f} \t {:8.8f}  \n'.format(at[j],ac[j][0],ac[j][1],ac[j][2]) )
                else:
                    for j in range(na):
                        f.write('{:3s} \t {:8.8f} \t {:8.8f} \t {:8.8f}  \t {:8.8f} \t {:8.8f} \t {:8.8f}\n'.format(at[j],ac[j][0],ac[j][1],ac[j][2], fa[j][0],fa[j][1],fa[j][2]) )
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
        forces=[]
        for line in lines[2:na+2]:
            l = line.split()
            at_types.append(l[0])
            coords.append(np.array(l[1:4], dtype=float))
            forces.append(np.array(l[4:7], dtype=float))

        newattr['coords'] = [coords]
        newattr['at_type'] = [at_types]
        newattr['Forces'] = [forces]
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
            
            if key_in_line(['Input','orientation:'],lin): 
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
                    forces.append(np.array(li[2:5],dtype=float))
                forces = np.array(forces)
                 #hartrees/bohr to kcal/mol/A
                if units =='kcal/mol': forces*=627.5096080305927/0.529177
                fdat.append(np.array(forces))
            
            data_dict['Forces'] = fdat
        
        data_dict['Energy'] = eners
        
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


    def plot_discriptor_distribution(self,ty,inter_type='vdw',bins=100,ret = False):
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
        plt.hist(dists,bins=bins,density=True,color='magenta')
        #plt.show()
        plt.savefig(f'{self.setup.runpath}/discriptor_distribution_{inter_type}_{ty}.png',bbox_inches='tight') 
        if ret:
            return dists
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


        
    
class FF_Optimizer(Optimizer):
    
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
        
        fid = self.get_dataDict(dataset,'forces_info')
        
        natoms_dict =self.get_dataDict(dataset,'natoms')
        
        models_list = []

        for name,model in models.items():
            
            if self.check_excess_models(model.category,model.num):
                continue
            
            model_attributes = self.serialize_values(fid, natoms_dict, model)
            
            minfo = self.Model_Info(model, model_attributes)

            models_list.append(minfo)
        
        return  models_list
    
    def set_UFclass_ondata(self,which='opt',dataset='all'):
        
        index = self.get_indexes(dataset)
        ndata = len(index)
        models = getattr(self.setup,which+'_models')
        
        
        params, bounds, fixed_params, isnot_fixed,reguls = self.get_parameter_info(models)        
        models_list_info =  self.get_list_of_model_information(models,dataset)
        
        Uclass = self.computeUclass(params, ndata, models_list_info)
        
        natoms_per_point = np.array( list( self.get_dataDict(dataset,'natoms').values() ) ) 
        n_forces = np.sum( natoms_per_point )
        
        Fclass_array = self.computeForceClass(params, n_forces, models_list_info)
        
        Fclass = {m: np.zeros( (natoms,3),dtype=float) for m, natoms in enumerate(natoms_per_point) }
        for m  in range(ndata):
            for model_info in models_list_info:
                nat_low = model_info.nat_low
                nat_up = model_info.nat_up
                Fclass[m] = Fclass_array[nat_low[m]:nat_up[m]].copy() 
        
        
        index = self.get_indexes(dataset)
        self.data.loc[index,'Uclass'] = Uclass
        self.data.loc[index,'Fclass'] = pd.Series(Fclass.values(), index=index)
        return 

    @staticmethod
    def UperModelContribution(u_model,dists,dl,du,model_pars,*model_args):
        #compute UperModelContribution
        #a = (dists,model_pars,*model_args)
        Up = np.empty((dl.shape[0]), dtype=np.float64)
        pobj = u_model(dists,model_pars,*model_args)
        U = pobj.u_vectorized()
        #U = u_model(*a)
        for i in prange(Up.size):
            ui = U[dl[i] : du[i]].sum() #if du[i] -dl[i]>0 else 0
            Up[i] = ui 
        
        return Up
   
    @staticmethod
    def UperModelContribution_grad(u_model,dists,dl,du,model_pars,*model_args):
        #compute UperModelContribution
        #a = (dists,model_pars,*model_args)
        n_p = model_pars.shape[0]
        data_size = dl.shape[0]
        gu = np.empty((n_p,data_size), dtype=np.float64)
        
        pobj = u_model(dists,model_pars,*model_args)
        g = pobj.find_gradient() # shape = (n_p, dists.shape[0]) # dists.shape[0] == all the values serialized

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
            model_pars = FF_Optimizer.array_model_parameters(objparams,
                                                    minf.fixed_params,
                                                    minf.isnot_fixed,
                                                    )
            #print('overhead {:4.6f} sec'.format(perf_counter()-t0))
            #compute Uclass
            Utemp = FF_Optimizer.UperModelContribution( minf.u_model,
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
            model_pars = FF_Optimizer.array_model_parameters(objparams,
                                                    minf.fixed_params,
                                                    minf.isnot_fixed,
                                                    )

            gu = FF_Optimizer.UperModelContribution_grad( 
                    minf.u_model,
                    minf.dists, minf.dl, minf.du,
                    model_pars, *minf.model_args)

            Uclass_grad[npars_old: npars_new] = gu[minf.isnot_fixed]
           
            npars_old = npars_new
        return Uclass_grad
    
    @staticmethod
    def computeForceClass(params, n_forces, models_list_info):

        Forces_tot =  np.zeros( ( n_forces ,3), dtype=np.float64) 
        npars_old = 0
        for model_info in models_list_info:
            Forces =  np.zeros( (  n_forces ,3), dtype=np.float64)   
            #t0 = perf_counter()
           
            npars_new = npars_old  + model_info.n_notfixed
            
            objparams = params[ npars_old : npars_new ]
            model_pars = FF_Optimizer.array_model_parameters(objparams,
                                                    model_info.fixed_params,
                                                    model_info.isnot_fixed,
                                                    )
            
            FF_Optimizer.ForcesPerModel(Forces, model_pars, model_info)
            
            
            npars_old = npars_new
            Forces_tot += Forces

        return Forces_tot
    
    @staticmethod
    def computeGradForceClass(params, n_forces, models_list_info):
        
        gradForces_tot =  np.zeros( (params.shape[0], n_forces ,3), dtype=np.float64) 
        npars_old = 0
        for model_info in models_list_info:
            gradForces =  np.zeros( ( model_info.n_pars,  n_forces ,3),
                               dtype=np.float64)   
            #t0 = perf_counter()
           
            npars_new = npars_old  + model_info.n_notfixed
            
            objparams = params[ npars_old : npars_new ]
            model_pars = FF_Optimizer.array_model_parameters(objparams,
                                                    model_info.fixed_params,
                                                    model_info.isnot_fixed,
                                                    )
            
            FF_Optimizer.gradForcesPerModel(gradForces, model_pars, model_info)
            
            gradForces_tot[npars_old: npars_new] = gradForces[model_info.isnot_fixed]
            
            npars_old = npars_new
            
        return gradForces_tot
    
    @staticmethod
    def gradForcesPerModel(gradForces, model_pars, model_info ):
        #compute UperModelContribution
        dists = model_info.dists
        if dists.shape[0] == 0: 
            return
        n_pars = model_info.n_pars
        
        compute_obj = model_info.u_model(dists,model_pars,*model_info.model_args)
        
        i_index = model_info.i_indexes
        j_index = model_info.j_indexes
        #ntotal = number of forces
        fg = - compute_obj.find_derivative_gradient() #shape = (npars, ntotal)
        nf = fg.shape[1]
        if model_info.category == 'PW' or model_info.category == 'BO':
            for n in range(n_pars):
                pw_ij = fg[n].reshape( (nf,1) )*model_info.partial_ri
                FF_Optimizer.numba_add_ij(gradForces[n], pw_ij, i_index, j_index)
            
        elif model_info.category == 'LD':
            for n in range(n_pars):
                pw_ij = fg[n].reshape( (nf,1) )[ model_info.to_ij ]*model_info.v_ij
                FF_Optimizer.numba_add_ij(gradForces[n], pw_ij, i_index, j_index)
        elif model_info.category =='AN':
            k_index = model_info.k_indexes
            for n in range(n_pars):
                fa = model_info.pa*fg[n].reshape( (nf,1) )
                fc = model_info.pc*fg[n].reshape( (nf,1) )
                FF_Optimizer.numba_add_angle(gradForces[n], fa, fc, i_index, j_index, k_index)
        return
    
    
    @staticmethod
    def ForcesPerModel(Forces, model_pars, model_info ):
        #compute UperModelContribution
        dists = model_info.dists
        if dists.shape[0] == 0:
            return
        compute_obj = model_info.u_model(dists,model_pars,*model_info.model_args)
        
        i_index = model_info.i_indexes
        j_index = model_info.j_indexes
       
        dudx_vectorized = - compute_obj.find_dydx()
        dudx_vectorized = dudx_vectorized.reshape((dudx_vectorized.shape[0],1))
        
        if model_info.category == 'PW' or model_info.category == 'BO':
            pw_i = dudx_vectorized*model_info.partial_ri
            FF_Optimizer.numba_add_ij(Forces, pw_i, i_index, j_index)
            
        elif model_info.category == 'LD':
            pw_i = dudx_vectorized[ model_info.to_ij ]*model_info.v_ij
            FF_Optimizer.numba_add_ij(Forces,pw_i, i_index,j_index)
        
        elif model_info.category =='AN':
            k_index = model_info.k_indexes
            fa = model_info.pa*dudx_vectorized
            fc = model_info.pc*dudx_vectorized
            FF_Optimizer.numba_add_angle(Forces, fa, fc, i_index, j_index, k_index)
            
        return
    
    @jit(nopython=True,fastmath=True)
    def numba_add_angle(forces, fa,fc, i_indices, j_indices,k_indices):
        for m in prange(len(i_indices)):
            i, j, k = i_indices[m] , j_indices[m], k_indices[m]
            forces[i] += fa[m]
            forces[k] += fc[m]
            forces[j] -= (fa[m]+fc[m])
        return
    
    @jit(nopython=True,fastmath=True)
    def numba_add_ij(forces, pairwise_forces, i_indices, j_indices):
        for k in prange(len(i_indices)):
            i, j = i_indices[k], j_indices[k]
            forces[i] += pairwise_forces[k]
            forces[j] -= pairwise_forces[k]
        return 
    
    def test_gradForceClass(self, which='opt', epsilon=1e-4, 
                        verbose=False,order=2):
        """
        Compute and compare the analytical and numerical gradients of the Forces
        using second order or forth order finite difference methods.
    
        This method calculates the gradients of the Forces analytically and  numerically.
. 
    
        Parameters:
        ----------
        which : str, optional
            Specifies which model to use for analytical gradient computation. 
            Default is 'opt', which refers to the optimized model.
            
        epsilon : float, optional
            The step size for finite differences when calculating numerical gradients. 
            Default is 1e-4.

        verbose : bool, optional
            If True, prints detailed information about the computation and comparisons. 
            Default is False.
        order : int, optional
            order of differentiation
            Default is 4
        
        """
        
        
        dataset='all'

        models = getattr(self.setup, which + '_models')
        params, bounds, fixed_params, isnot_fixed, reguls = self.get_parameter_info(models)
        models_list_info = self.get_list_of_model_information(models, dataset)
        
        n_p = params.shape[0]
        natoms_per_point = self.data['natoms'].to_numpy()
        ntot  =  np.sum(natoms_per_point)
        for _ in range(2):
            t0 = perf_counter()
            gradForces_tot = self.computeGradForceClass(params,ntot, models_list_info)
            tf = perf_counter() - t0
        
        grads_numerical = np.empty_like(gradForces_tot)
        
        print('Time to compute Analytical Force Gradients = {:4.3e}  ms, {:4.3e} ms/datapoint '.format(tf*1000,tf*1000/len(self.data)))
        for i in range(n_p):
            p1 = params.copy()
            p1[i] += epsilon
            fp1 = self.computeForceClass(p1, ntot , models_list_info)
            
            m1 = params.copy()
            m1[i] -= epsilon
            fm1 = self.computeForceClass(m1, ntot, models_list_info)
            if order == 2:
                # Second-order central difference

                grads_numerical[i] = (fp1 - fm1) / (2 * epsilon)
            
            elif order == 4:
                p2 = params.copy()
                p2[i] += 2*epsilon
                fp2 = self.computeForceClass(p2, ntot,  models_list_info)
                
                m2 = params.copy()
                m2[i] -= 2 * epsilon
                fm2 = self.computeForceClass(m2, ntot,  models_list_info)
                
                grads_numerical[i] = (-fp2 + 8 * fp1 - 8 * fm1 + fm2) / (12 * epsilon)
            
            else:
                raise ValueError("Order must be 2 or 4.")
            if verbose:
                for dr in range(3):
                    max_diff = np.abs(grads_numerical[i,:,dr] - gradForces_tot[i,:,dr]).max()
                    a = np.abs(grads_numerical[i,:,dr] - gradForces_tot[i,:,dr]).argmax()
                    print('Gradient {:d}: direction {:d} max diff = {:4.3e} on {}'.format(i,dr, max_diff,a))
        
        max_diff = np.abs(grads_numerical - gradForces_tot).max()
        a = np.abs(grads_numerical - gradForces_tot).argmax()
        indices = np.unravel_index(a, gradForces_tot.shape)
        mean_diff = np.abs(grads_numerical - gradForces_tot).mean()
        print('max diff = {:4.3e}  at {} mean diff = {:4.3e}'.format(max_diff, indices, mean_diff))
        
        return gradForces_tot
   
    def test_ForceClass(self, which='opt', epsilon=1e-4,  seed = 2024,
                        verbose=False,random_tries=10,
                        check_only_analytical_forces=False,order=4):
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
        order : int, optional
            order of differentiation
            Default is 4
        
        """
        dataset='all'
        
        ndata = len(self.get_Energy(dataset))
        models = getattr(self.setup, which + '_models')
        params, bounds, fixed_params, isnot_fixed, reguls = self.get_parameter_info(models)
        models_list_info = self.get_list_of_model_information(models, dataset)
        for _ in range(2):
            t0 = perf_counter()
            _ = self.computeUclass(params, ndata, models_list_info)
            tf = perf_counter() - t0
        print('Time to compute analytical classical energy {:4.3e} ms'.format(tf*1000))
        
        # Analytical gradient calculation
        natoms_per_point = self.data['natoms'].to_numpy()
        for _ in range(2):
            t0 = perf_counter()
            Forces_tot = self.computeForceClass(params, np.sum( natoms_per_point), models_list_info)
        tf = perf_counter() - t0
        Forces_analytical = {m: np.zeros( (natoms,3),dtype=float) for m, natoms in enumerate(natoms_per_point) }
        #Forces_analytical = Forces_tot
        for m  in range(len(Forces_analytical)):
            for model_info in models_list_info:
                nat_low = model_info.nat_low
                nat_up = model_info.nat_up
                Forces_analytical[m] = Forces_tot[nat_low[m]:nat_up[m]] 
            
            
        print('Time to compute Analytical Forces = {:4.3e}  ms, {:4.3e} ms/datapoint '.format(tf*1000,tf*1000/len(self.data)))
        if check_only_analytical_forces:
            for m, fa in Forces_analytical.items():
                max_force = np.abs(fa).max()
                min_force = np.abs(fa).min()
                mean_force = np.abs(fa).mean()
                if verbose:
                    print('point {:d} max_force = {:4.3e} mean_force = {:4.3e} min_force = {:4.3e}'.format(m,max_force,mean_force,min_force))
            return Forces_analytical
        # Numerical gradient calculation
        
        Forces_numerical =  {m: np.zeros( (natoms,3),dtype=float) for m, natoms in enumerate(natoms_per_point) }
        
        
        
        self.data.drop(columns=['forces_info', 'interactions' ], inplace=True)
        
        
        coords_copy = copy.deepcopy(self.data['coords'].to_numpy())
        if verbose:
            print(f'Calculating the Forces on a random atom, seed = {seed} ...')
        
        all_diffs = []
        where_max_diff = [] 
        np.random.seed(seed)
        seeds = np.random.randint(0,random_tries*1000,size=random_tries)
        for random_try in range(random_tries):
            np.random.seed(seeds[random_try])
            atoms_to_modify = [np.random.randint(0,natoms) for m, natoms  in enumerate(natoms_per_point)]
            differences = []
            for dir_index in range(3): 
                
                # up1
                for m,idx in enumerate(self.data.index):
                    atom_index = atoms_to_modify[m]
                    self.data['coords'][idx][atom_index][dir_index] += epsilon
                    
                al_help.make_interactions(self.data, self.setup)
                models_list_info = self.get_list_of_model_information(models, dataset)    
                up1 = self.computeUclass(params, ndata, models_list_info)
                
                self.data.drop(columns=['forces_info', 'interactions','coords' ], inplace=True)
                self.data['coords'] = copy.deepcopy(coords_copy)
                
                #um1
                for m,idx in enumerate(self.data.index):
                    atom_index = atoms_to_modify[m]
                    self.data['coords'][idx][atom_index][dir_index] -= epsilon
                
                al_help.make_interactions(self.data, self.setup)
                models_list_info = self.get_list_of_model_information(models, dataset)    
               
                um1 = self.computeUclass(params, ndata, models_list_info)
                
                self.data.drop(columns=['forces_info', 'interactions','coords' ], inplace=True)
                self.data['coords'] = copy.deepcopy(coords_copy)
                
                
                if order == 4:
                    #up2
                    for m,idx in enumerate(self.data.index):
                        atom_index = atoms_to_modify[m]
                        self.data['coords'][idx][atom_index][dir_index] += 2*epsilon
                        
                    al_help.make_interactions(self.data, self.setup)
                    models_list_info = self.get_list_of_model_information(models, dataset)    
                    up2 = self.computeUclass(params, ndata, models_list_info)
                    
                    self.data.drop(columns=['forces_info', 'interactions','coords' ], inplace=True)
                    self.data['coords'] = copy.deepcopy(coords_copy)
                    
                    for m,idx in enumerate(self.data.index):
                        atom_index = atoms_to_modify[m]
                        self.data['coords'][idx][atom_index][dir_index] -= 2*epsilon
                        
                    al_help.make_interactions(self.data, self.setup)
                    models_list_info = self.get_list_of_model_information(models, dataset)    
                    um2 = self.computeUclass(params, ndata, models_list_info)
                    
                    self.data.drop(columns=['forces_info', 'interactions','coords' ], inplace=True)
                    self.data['coords'] = copy.deepcopy(coords_copy)

                if order==4:
                    for m in Forces_numerical.keys():
                        atom_index = atoms_to_modify[m]
                        Forces_numerical[m][atom_index, dir_index] = - (-up2[m] + 8 * up1[m] - 8 * um1[m] + um2[m]) / (12 * epsilon)
                else:
                    for m in Forces_numerical.keys():
                        atom_index = atoms_to_modify[m]
                        Forces_numerical[m][atom_index, dir_index] = - (up1[m] - um1[m]) / ( 2*epsilon)
                #if verbose:
                #    print(f'Numerical Forces Calculated. Comparing direction {dir_index}...')
                
                for m in Forces_numerical.keys():
                    
                    fn = Forces_numerical[m]
                    fa = Forces_analytical[m]
                    
                    atom_index = atoms_to_modify[m]
                    
                    fn_ad = fn[atom_index , dir_index]
                    fa_ad = fa[atom_index , dir_index]
                    diff = np.abs(fn_ad - fa_ad)
                    differences.append(diff)
                    where_max_diff.append((m,atom_index,dir_index))
                    if verbose and diff.max()>1e-3:
                        print('data_point = {:d}, atom = {:d}, dir = {:d}, Fnum = {:.4e} , Fana = {:.4e} --> diff = {:.4e}'.format( m, atom_index, dir_index, fn_ad, fa_ad, diff))
            dmax = np.max(differences)
            dmean = np.mean(differences)
            print('random try {:d} --> max diff = {:4.3e}, mean diff = {:4.3e}'.format(random_try,dmax,dmean))
            all_diffs.extend(differences)
        a = where_max_diff[np.argmax(all_diffs)]
        print('Max diff: {:4.3e} at {}'.format(np.max(all_diffs),a))
        return  Forces_analytical
    
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
        for _ in range(2):
            t0 = perf_counter()
            grads_analytical = self.gradUclass(params, ndata, models_list_info)
            tf = perf_counter() -t0
        print('time for grad of classical energy = {:4.3e} ms '.format(tf*1000))
        # Numerical gradient calculation
        n_p = params.shape[0]
        grads_numerical = np.empty((n_p, ndata), dtype=np.float64)
        
        for i in range(n_p):
            p1 = params.copy()
            p1[i] += epsilon
            up1 = self.computeUclass(p1, ndata, models_list_info)
            
            m1 = params.copy()
            m1[i] -= epsilon
            um1 = self.computeUclass(m1, ndata, models_list_info)
            if order == 2:
                # Second-order central difference

                grads_numerical[i] = (up1 - um1) / (2 * epsilon)
            
            elif order == 4:
                p2 = params.copy()
                p2[i] += 2*epsilon
                up2 = self.computeUclass(p2, ndata, models_list_info)
                
                m2 = params.copy()
                m2[i] -= 2 * epsilon
                um2 = self.computeUclass(m2, ndata, models_list_info)
                
                grads_numerical[i] = (-up2 + 8 * up1 - 8 * um1 + um2) / (12 * epsilon)
            
            else:
                raise ValueError("Order must be 2 or 4.")
        diff = np.abs(grads_analytical - grads_numerical).max()
        ave = np.abs(grads_analytical - grads_numerical).mean()
        print( "Maximum Difference in numerical and analytical gradient {:4.3e} , mean diff = {:4.3e}".format(diff,ave))
        return grads_analytical, grads_numerical

    def test_CostGrads(self,params, args, order = 4, epsilon= 1e-5,tol=10 ):
        print('Testing Cost Gradient ... ')
        n_p = params.shape[0]
        gnum = np.empty_like(params)
        gana = self.gradCost(params,*args)
        for i in range(n_p):
            p1 = params.copy()
            p1[i] += epsilon
            up1 = self.CostFunction(p1, *args)
            
            m1 = params.copy()
            m1[i] -= epsilon
            um1 = self.CostFunction(m1, *args)
            if order == 2:
                # Second-order central difference

                gnum[i] = (up1 - um1) / (2 * epsilon)
            
            elif order == 4:
                p2 = params.copy()
                p2[i] += 2*epsilon
                up2 = self.CostFunction(p2, *args)
                
                m2 = params.copy()
                m2[i] -= 2 * epsilon
                um2 = self.CostFunction(m2, *args)
                
                gnum[i] = (-up2 + 8 * up1 - 8 * um1 + um2) / (12 * epsilon)
            
            else:
                raise ValueError("Order must be 2 or 4.")
            diff = np.abs(gnum[i] - gana[i])
            print('parameter {:d} diff = {:4.3e} '.format(i, diff ) )
            if diff > tol*epsilon:
                print(f'Warning: Parameter {i} might have inaccuracies. It might be also due to numerical errors on the gradient estimation')
        return
    @staticmethod
    def CostFunction(params,
                     Energy, Forces, lambda_force, models_list_info,
                     reg, reguls,
                     measure,reg_measure,
                     mu_e =0.0, std_e=1.0,
                     mu_f = 0.0, std_f=1.0):   
        
        cE = CostFunctions.Energy(params, Energy, models_list_info, measure, mu_e,std_e)
        cR = reg*CostFunctions.Regularization(params,reguls,reg_measure) 
        cF = CostFunctions.Forces(params, Forces, models_list_info, measure, mu_f,std_f) 
        cost = (1.0-lambda_force)*cE + lambda_force*cF + cR
        return cost
    
    @staticmethod
    def gradCost(params,
                Energy, Forces, lambda_force, models_list_info,
                reg,reguls,
                measure,reg_measure,
                mu_e =0.0, std_e=1.0,
                mu_f = 0.0, std_f=1.0):   
        # serialize the params
        gradE = CostFunctions.gradEnergy(params, Energy, models_list_info, measure, mu_e,std_e)
        gradR = reg*CostFunctions.gradRegularization(params,reguls,reg_measure) 
        gradF = CostFunctions.gradForces(params, Forces, models_list_info, measure, mu_f,std_f) 
        grads =  (1.0-lambda_force)*gradE + lambda_force*gradF + gradR
        
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
    

    
    def serialize_values(self,values_dict, natoms_dict, model):
        
        ndata = len(values_dict)
        dl = np.empty((ndata,), dtype=int)
        du = np.empty((ndata,), dtype=int)
        nat_up = np.empty((ndata,), dtype=int)
        nat_low = np.empty((ndata,), dtype=int)
        
        dists = [] 
        nd = 0
        na = 0
        for m,(idx,val) in enumerate(values_dict.items()):
            if model.type  not in val[model.feature]:
                d = np.empty(0,dtype=float)
            else:
                d = val[model.feature][model.type]['values']
                # val_to_ij
            dl[m] = nd
            nd += d.shape[0]
            du[m] = nd 
            dists.extend(d)
            
            nat_low[m] = na
            na += natoms_dict[idx]
            nat_up[m] = na
            
        dists = np.array(dists)
        model_attributes = {'dists':dists, 'dl':dl, 'du':du,
                                       'nat_low': nat_low, 'nat_up':nat_up
                                       }
        
        if model.category == 'BO' or model.category =='PW':
            
            partial_ri  = [] ;  i_indexes = [] ; j_indexes = []
            na = 0
            
            for m,(idx,val) in enumerate(values_dict.items()):
                
                if model.type  not in val[model.feature]:
                    rh = np.empty(0,dtype=float)
                    i_ix = np.empty(0,dtype=int)
                    j_ix = np.empty(0,dtype=int)
                else:
                    rh = val[model.feature][model.type]['partial_ri']
                    i_ix = val[model.feature][model.type]['i_index']
                    j_ix = val[model.feature][model.type]['j_index']
                
            
                partial_ri.extend(rh)
                i_indexes.extend(i_ix + na)
                j_indexes.extend(j_ix + na)
                na += natoms_dict[idx]
            
            model_attributes.update({'partial_ri':np.array(partial_ri),
                                           'i_indexes':np.array(i_indexes),
                                           'j_indexes':np.array(j_indexes)})
            
        elif model.category == 'LD':
            
            v_ij  = [] ;  i_indexes = [] ; j_indexes = [] ; to_ij = []
            na = 0
            nc =0
            for m,(idx,val) in enumerate(values_dict.items()):
                
                if model.type  not in val[model.feature]:
                    vh = np.empty(0,dtype=float)
                    i_ix = np.empty(0,dtype=int)
                    j_ix = np.empty(0,dtype=int)
                    to_ij_x = np.empty(0,dtype=int)
                    n_central = 0
                else:
                    vh = val[model.feature][model.type]['v_ij']
                    i_ix = val[model.feature][model.type]['i_index'] 
                    j_ix = val[model.feature][model.type]['j_index'] 
                    to_ij_x = val[model.feature][model.type]['to_pair_index']
                    n_central = val[model.feature][model.type]['n_central']

                v_ij.extend( vh )
                i_indexes.extend( i_ix  + na )
                j_indexes.extend( j_ix + na )
                to_ij.extend( to_ij_x  + nc )
                na += natoms_dict[ idx ]
                nc += n_central
            model_attributes.update({'v_ij':np.array(v_ij),
                                           'i_indexes':np.array(i_indexes),
                                           'j_indexes':np.array(j_indexes),
                                           'to_ij':np.array(to_ij)})
        elif model.category == 'AN':
            pa = [] ; pc = [];  i_indexes = [] ; j_indexes = [] ; k_indexes = []
            na = 0
            
            for m,(idx,val) in enumerate(values_dict.items()):
                
                if model.type  not in val[model.feature]:
                    pa_ix = np.empty(0,dtype=float)
                    pc_ix = np.empty(0,dtype=float)
                    i_ix = np.empty(0,dtype=int)
                    j_ix = np.empty(0,dtype=int)
                    k_ix = np.empty(0,dtype=int)
                else:
                    pa_ix = val[model.feature][model.type]['pa']
                    pc_ix = val[model.feature][model.type]['pc']
                    i_ix = val[model.feature][model.type]['i_index']
                    j_ix = val[model.feature][model.type]['j_index']
                    k_ix = val[model.feature][model.type]['k_index']
                    
            
                pa.extend( pa_ix )
                pc.extend( pc_ix )
                
                i_indexes.extend( i_ix + na )
                j_indexes.extend( j_ix + na )
                k_indexes.extend( k_ix + na )
                
                na += natoms_dict[ idx ]
            
            model_attributes.update({'pa':np.array(pa),'pc':np.array(pc),
                                           'i_indexes':np.array(i_indexes),
                                           'j_indexes':np.array(j_indexes),
                                           'k_indexes':np.array(k_indexes)
                                    })
        return model_attributes
    
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
    
    class TimeValues:
        def __init__(self,tot,nfev,overhead,npoints):
            self.total_time = tot
            self.time_per_evaluation = (tot-overhead)/nfev
            self.evaluation_time = (tot - overhead)
            self.overhead = overhead
            self.time_per_point = (tot-overhead)/npoints
            self.time_per_point_per_evaluation = (tot-overhead)/nfev/npoints
            
            return
        
        @staticmethod
        def format_time(seconds):
            # Convert to milliseconds
            milliseconds = (seconds - int(seconds)) * 1000.0
        
            # Convert to hours and remaining seconds
            hours = int(seconds // 3600)
            remaining_seconds = seconds % 3600
        
            # Separate minutes and seconds
            mins = int(remaining_seconds // 60)
            secs = int(remaining_seconds % 60) 
            
            return f"{hours} h : {mins:02} min :{secs:02} sec  + " + "{:4.3e}".format(milliseconds)

        def __repr__(self):
            x = 'Time Cost : value \n--------------------\n'
            for k,v in self.__dict__.items():
                x+='{} : {} ms\n'.format(k,self.format_time(v)  )
            x+='--------------------\n'
            return x 
    
    class CostValues:
        def __init__(self): 
            return

        def __repr__(self):
            x = 'Cost : value \n--------------------\n'
            for k,v in self.__dict__.items():
                x+='{} : {:7.8f} \n'.format(k, v )
            x+='--------------------\n'
            return x 

    class Model_Info():
        def __init__(self,model, model_attributes):
            
            fixed_params = []; isnot_fixed = []
           
            for k, m in model.pinfo.items():
                isnot_fixed.append(m.opt) # its a question variable
                if not  m.opt:
                    fixed_params.append(m.value)
                    
            fixed_params = np.array(fixed_params)
            isnot_fixed = np.array(isnot_fixed)
            self.name = model.name
            self.category = model.category
            self.u_model = model.function 
            self.n_pars = len(model.pinfo)
            self.model_args = model.model_args
            self.fixed_params = fixed_params
            self.isnot_fixed = isnot_fixed
            self.n_notfixed = np.count_nonzero(isnot_fixed)
            #self.dists = dists
            #self.dl = dl
            #self.du = du
            for k,attr in model_attributes.items():
                setattr(self,k,attr)
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
        params = params.copy()
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
    
    def get_true_Forces(self,dataset):
        fv = self.get_dataDict(dataset,'Forces').values()
        Forces = []
        for f in fv:
            for fatom in f:
                Forces.append(fatom)
        Forces = np.array(Forces)
        return Forces
    
    def get_params_n_args(self,setfrom,dataset):
        
        E = self.get_Energy(dataset)
        
        Forces = self.get_true_Forces(dataset)
        
        
        weights =  GeneralFunctions.weighting(self.data_train,
                self.setup.weighting_method,self.setup.bT,self.setup.w)
        #argsopt= np.where (self.data_train['label']=='optimal')[0]
        #weights[argsopt]*=5.0
        self.weights = weights

        models = getattr(self.setup,setfrom+'_models')

        params, bounds, fixed_parameters, isnot_fixed, reguls = self.get_parameter_info(models) 
        
        models_list_info  = self.get_list_of_model_information(models, dataset)
        
        self.models_list_info = models_list_info
        
        args = (E,Forces,self.setup.lambda_force,
                models_list_info, 
                self.setup.reg_par, reguls,
                self.setup.costf,
                self.setup.regularization_method)
        
        return params, bounds, args, fixed_parameters, isnot_fixed

    def pareto_via_scan(self,setfrom='init'):
        
        nrandom, npareto = self.setup.random_initializations, self.setup.npareto
        tol = self.setup.tolerance

        params, bounds,  args, fixed_parameters, isnot_fixed = self.get_params_n_args(setfrom,'train')
        
        (Energy,Forces, lambda_force,models_list_info, 
                 reg_par, reguls,
                 measure,
                 measure_reg ) = args
        
        normalize_data = self.setup.normalize_data
        if normalize_data:
            mu_e, std_e, mu_f, std_f = self.get_normalized_data('train')
            args = (*args, mu_e, std_e, mu_f, std_f)
        if params.shape[0] >0 and self.setup.optimize :
            self.randomize = True
            best_params, success, best_iter = params,  False,0
            
            lfs = np.arange(0,0.999999, 1.0/npareto)
            energy_costs = []
            force_costs = []
            best_se = 1e17
            
            for iteration, lf in enumerate(lfs):
                args = list(args)
                args[2] = lf
                args = tuple(args)
              
                best_params, best_fun, success = params, 1e17 , False
                
                for rand_sol in range(nrandom+1):
                    if rand_sol !=0:
                        params = self.randomize_initial_params(best_params.copy(),bounds)
                    res = minimize(self.CostFunction, params,
                               args = args,
                               jac = self.gradCost,
                               bounds=bounds,tol=tol, 
                               options={'disp':self.setup.opt_disp,
                                        'maxiter':self.setup.maxiter,
                                        'ftol': self.setup.tolerance},
                               method = 'SLSQP')
                    if res.success:
                        success = True
                        if res.fun < best_fun:
                            best_fun ,best_params  = res.fun, res.x.copy()
                    else:
                        break
                params = best_params
                        
                if not success:
                    break
                self.current_res = res
                self.set_models('init','opt',params,isnot_fixed,fixed_parameters)
                self.set_results()
                se = self.current_costs.selection_metric
                ce, cf = self.current_costs.selection_energy, self.current_costs.selection_forces
                energy_costs.append( ce )
                force_costs.append( cf )
                if se < best_se:
                    best_se , best_iter = se, iteration
                    self.set_models('opt','best_opt')
                    params = res.x
                print('Iteration {:d}, Energy Cost = {:4.5f} Force Cost = {:4.5f}'.format(
                    iteration, ce, cf))
                print('Iteration {:d},  best iter = {:d}, best_metric = {:4.5f}'.format(
                    iteration,best_iter,best_se))

                sys.stdout.flush()
            self.set_models('best_opt','opt')
            self.set_results()        
            _ = plt.figure(figsize=(3.3,3.3),dpi=300)
            plt.xlabel('Energy costs')
            plt.ylabel('Force costs')
            
            plt.plot(energy_costs,force_costs,marker='s',ls='none',color='blue')
            plt.plot(energy_costs[0], force_costs[0], marker='*', ls='none',color='k')
            plt.plot(energy_costs[best_iter],force_costs[best_iter],marker='o',ls='none',color='red')
            plt.savefig(f'{self.setup.runpath}/pareto_scan.png', bbox_inches='tight')
            plt.show()
            return     

    def pareto_via_constrain(self,setfrom='init'):
        
        nrandom, npareto = self.setup.random_initializations, self.setup.npareto
        tol = self.setup.tolerance

        params, bounds,  args, fixed_parameters, isnot_fixed = self.get_params_n_args(setfrom,'train')
        
        (Energy,Forces, lambda_force,models_list_info, 
                 reg_par, reguls,
                 measure,
                 measure_reg ) = args
        
        args_energy = (Energy, models_list_info, reg_par, reguls, measure, measure_reg)
        #args_forces = (Forces, models_list_info, reg_par, reguls, measure, measure_reg)
        
        normalize_data = self.setup.normalize_data
        if normalize_data:
            mu_e, std_e, mu_f, std_f = self.get_normalized_data('train')

            args_energy = (*args_energy, mu_e, std_e ) 
            
        if params.shape[0] >0 and self.setup.optimize :
            self.randomize = True
            best_params, best_fun, success = params, 1e17, False
            for rand_sol in range(nrandom+1):
                if rand_sol !=0:
                    params = self.randomize_initial_params(params.copy(),bounds)
                t0 = perf_counter()
                res = minimize(self.costEnergy, params,
                                   args = args_energy,
                                   jac = self.gradEnergy,
                                   bounds=bounds,tol=tol, 
                                   options={'disp':self.setup.opt_disp,
                                            'maxiter':self.setup.maxiter,
                                            'ftol': self.setup.tolerance},
                                   method = 'SLSQP')
                tmt = perf_counter() - t0 
                self.minimization_time = tmt
                self.tpfev = tmt/res.nfev
                if res.fun < best_fun:
                    best_fun ,best_params  = res.fun, res.x

            self.current_res = res
            
            self.set_models('init','opt',best_params,isnot_fixed,fixed_parameters)
            self.set_models('opt','best_opt')
            self.set_results()
            best_se = self.current_costs.selection_metric
            
            params = best_params
            max_force  = self.current_costs.selection_forces
            
            fvals = np.arange(max_force,0,-max_force/npareto)
            ce_init, cf_init = self.current_costs.selection_energy, self.current_costs.selection_forces
            best_iter, best_ce , best_cf = 0, ce_init, cf_init
            
            energy_costs = [ ce_init ] 
            force_costs =  [ cf_init ]
            
            for iteration, fv in enumerate(fvals):
                args_c_forces = (Forces, models_list_info, measure, fv)
                if normalize_data:
                    args_c_forces = (*args_c_forces, mu_f, std_f)
                best_params, best_fun, success = params, 1e17, False
                
                for rand_sol in range(nrandom+1):
                    if rand_sol !=0:
                        params = self.randomize_initial_params(params.copy(),bounds)
                    res = minimize(self.costEnergy, params,
                               args = args_energy,
                               jac = self.gradEnergy,
                               bounds=bounds,tol=tol, 
                               constraints = {'type':'eq',
                                               'fun':self.constrainForces,
                                               'jac':self.gradconstrainForces,
                                               'args':args_c_forces},
                               options={'disp':self.setup.opt_disp,
                                        'maxiter':self.setup.maxiter,
                                        'ftol': self.setup.tolerance},
                               method = 'SLSQP')
                    if res.success:
                        success = True
                        if res.fun < best_fun:
                            best_fun ,best_params  = res.fun, res.x
                    else:
                        break
                params = best_params
                        
                if not success:
                    break
                self.current_res = res
                self.set_models('init','opt',params,isnot_fixed,fixed_parameters)
                self.set_results()
                se = self.current_costs.selection_metric
                ce, cf = self.current_costs.selection_energy, self.current_costs.selection_forces
                energy_costs.append( ce )
                force_costs.append( cf )
                print('Iteration {:d}, Energy Cost = {:4.5f} Force Cost = {:4.5f}'.format(
                    iteration, ce, cf))
                print('Iteration {:d}, force desired error {:4.5f} best iter = {:d}, best_metric = {:4.5f}'.format(
                    iteration,fv,best_iter,best_se))
                sys.stdout.flush()
                if se < best_se:
                    best_se , best_ce, best_cf, best_iter = se, ce, cf, iteration
                    self.set_models('opt','best_opt')
                    params = res.x
            self.set_models('best_opt','opt')
            self.set_results()        
            _ = plt.figure(figsize=(3.3,3.3),dpi=300)
            plt.xlabel('Energy costs')
            plt.ylabel('Force costs')
            
            plt.plot(energy_costs,force_costs,marker='s',ls='none',color='blue')
            plt.plot([ce_init], [cf_init], marker='*', ls='none',color='k')
            plt.plot([best_ce],[best_cf],marker='o',ls='none',color='red')
            plt.savefig(f'{self.setup.runpath}/pareto_constrain.png', bbox_inches='tight')
            plt.show()
            return 
    @staticmethod
    def costEnergy(params,Energy, models_list_info, 
                   reg_par, reguls, measure, measure_reg, mu_e=0, std_e=0):
        cE = CostFunctions.Energy(params, Energy, models_list_info, measure, mu_e,std_e)
        cR = reg_par*CostFunctions.Regularization(params,reguls,measure_reg) 
        return cE + cR
    
    @staticmethod
    def constrainForces(params,Forces, models_list_info, 
                   measure, value,  mu_f=0, std_f=0 ):   
        cF = CostFunctions.Forces(params, Forces, models_list_info, measure, mu_f,std_f)
        return cF - value 
    @staticmethod
    def gradconstrainForces(params,Forces, models_list_info, 
                    measure, value, mu_f=0, std_f=0):   
        gF = CostFunctions.gradForces(params, Forces, models_list_info, measure, mu_f,std_f)
        return gF 
    @staticmethod
    def costForces(params,Forces, models_list_info, 
                   reg_par, reguls, measure, measure_reg, mu_f=0, std_f=0):   
        cR = reg_par*CostFunctions.Regularization(params,reguls,measure_reg) 
        cF = CostFunctions.Forces(params, Forces, models_list_info, measure, mu_f,std_f)
        return cF + cR 
    
    @staticmethod
    def gradEnergy(params,Energy, models_list_info, 
                   reg_par, reguls, measure, measure_reg, mu_e=0, std_e=0):
        gE = CostFunctions.gradEnergy(params, Energy, models_list_info, measure, mu_e,std_e)
        gR = reg_par*CostFunctions.gradRegularization(params,reguls,measure_reg) 
        return gE + gR
    
    @staticmethod
    def gradForces(params,Forces, models_list_info, 
                   reg_par, reguls, measure, measure_reg, mu_f=0, std_f=0):   
        gR = reg_par*CostFunctions.gradRegularization(params,reguls,measure_reg) 
        gF = CostFunctions.gradForces(params, Forces, models_list_info, measure, mu_f,std_f)
        return gF + gR 
    
    def optimize_params(self,setfrom='init'):
        t0 = perf_counter()
        
        CostFunc = self.CostFunction
        opt_method = self.setup.optimization_method
        tol = self.setup.tolerance
        maxiter = self.setup.maxiter
        
        params, bounds,  args, fixed_parameters, isnot_fixed = self.get_params_n_args(setfrom,'train')
        
        n_train = args[0].shape[0]

        normalize_data = self.setup.normalize_data
        if normalize_data:
            mu_e, std_e, mu_f, std_f = self.get_normalized_data('train')

            args = (*args, mu_e, std_e, mu_f,std_f ) 
        try:
            self.randomize
        
        except AttributeError:
            self.randomize= False
        
        if params.shape[0] >0 and self.setup.optimize :
            params = self.randomize_initial_params(params.copy(),bounds)
            t1 = perf_counter()
            
            if opt_method in ['SLSQP','BFGS','L-BFGS-B']:   
                logger.debug('I am performing {}'.format(opt_method))
                t0 = perf_counter()
                res = minimize(self.CostFunction, params,
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
                while(self.epoch <= self.setup.random_initializations):
                    current_total_indexes = list (self.total_train_indexes.copy())
                    itera=0
                    while( len(current_total_indexes) > 0):
                        self.train_indexes = np.random.choice(current_total_indexes,replace=False,
                                                              size=min(self.setup.SLSQP_batchsize, n_train , len(current_total_indexes) )
                                                              )
                        current_total_indexes = [ i for i in current_total_indexes if i not in self.train_indexes ]

                        self.optimize_params('init')

                        time_only_min += self.minimization_time

                        self.set_models('opt','init')
                        norm = 'norm_' if normalize_data else ''
                        dev_cost = getattr(self.current_costs, 'selection_metric')
                        #
                        if dev_cost <= best_cost:
                            best_epoch = self.epoch
                            best_cost = dev_cost
                            self.set_models('opt','best_opt')
                            self.best_costs = copy.deepcopy(self.current_costs)
                        train_cost = getattr(self.current_costs, norm + self.setup.costf +'_train_cost')  
                        print('epoch = {:d}, i = {:d},  development cost = {:.4e} train cost = {:.4e} '.format(self.epoch,itera,  dev_cost, train_cost ))
                        sys.stdout.flush()
                        total_fev += self.current_res.nfev
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
                
                self.timecosts = self.TimeValues(tn,total_fev,tn-time_only_min, args[0].shape[0]) 
                #Final on all data
                self.set_models('best_opt','opt')
                self.set_models('best_opt','init')
                
                self.set_results()

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
            self.current_res = res
            self.set_models('init','opt',res.x,isnot_fixed,fixed_parameters)
            self.set_results()
            
             
        else:
            # Filling with dump values to avoid coding errors
            self.set_models('init','opt')
            self.set_results()

        return 
    
    def get_normalized_data(self,dataset):
        
        Energy = self.get_Energy('train')
        Forces = self.get_true_Forces('train')
        sysname_train = self.data_train['sys_name'].to_numpy()
        
        if dataset == 'train':
            sysname_dataset = sysname_train
        else:
            sysname_dataset = np.array( list(self.get_dataDict(dataset, 'sys_name').values()) )
       
        unsys = np.unique(sysname_dataset)
        
        if len(unsys) == 1:
            mu_e = Energy.mean()
            std_e = Energy.std()
            if std_e ==0 : 
                std_e = 1.0 ;
        else:
            #assert sysn.shape[0] == Energy.shape[0], 'array of sys names and Energies is different'
            shape = sysname_dataset.shape 
            mu_e = np.empty(shape, dtype=float)
            std_e =np.empty(shape, dtype=float)
            for us in unsys:
                itr = np.where( sysname_train == us )[0] 
                ids = np.where( sysname_dataset == us )[0]
                mu_e[ids] = Energy[itr].mean()
                std_e[ids] = Energy[itr].std()
            std_e[ std_e == 0 ] = 1.0
        
        mu_f = Forces.mean()
        std_f = Forces.std()
 
        if std_f == 0: 
            std_f = 1.0 ;
        
        return mu_e, std_e , mu_f ,std_f
    
    def set_results(self):
        
        params, bounds, args, fixed_parameters, isnot_fixed = self.get_params_n_args('opt','train')
        
        self.set_UFclass_ondata('opt',dataset='all')
        
        costs = self.CostValues() 
        
        (Energy,Forces, lambda_force,models_list_info, 
                 reg_par, reguls,
                 measure,
                 measure_reg ) = args
        
        
        normalize = self.setup.normalize_data
        if normalize:
            train_mu_e, train_std_e, train_mu_f, train_std_f = self.get_normalized_data('train')
        
        for meas in np.unique(['MAE','MSE',measure]):
            for dataname in ['train','dev', 'all']:
                for norm in ['','norm_']:
                    params, bounds, args, fixed_parameters, isnot_fixed = self.get_params_n_args('opt',dataname)
                
                    (Energy,Forces, lambda_force,models_list_info, 
                     reg_par, reguls,
                     measure,
                     measure_reg ) = args
                    if normalize and norm =='norm_':
                        mu_e, std_e, mu_f, std_f = self.get_normalized_data(dataname)
                    else:
                        mu_e, std_e, mu_f, std_f = 0.0, 1.0, 0.0, 1.0
                    
                    creg = CostFunctions.Regularization(params,reguls, measure_reg)
                    ce = CostFunctions.Energy(params, Energy, models_list_info, meas, mu_e, std_e)
                    cf = CostFunctions.Forces(params, Forces, models_list_info, meas, mu_f, std_f)
                    
                    prefix = norm + meas+'_'+dataname+'_' 
                    setattr(costs,prefix + 'energy',  ce)
                    setattr(costs,prefix + 'forces', cf) 
                    
                    if dataname=='train':
                        if (normalize and norm=='norm_') or (norm=='' and normalize==False):
                            setattr(costs,prefix + 'cost_with_reg', (1.0-lambda_force)*ce + lambda_force*cf + reg_par*creg) 
                                
                        if norm == '': 
                            setattr(costs,prefix + 'reg', creg) 
                            setattr(costs,prefix + 'reg_scaled', reg_par*creg) 
                    setattr(costs,prefix + 'cost', ce + lambda_force*cf) 
                    if dataname == 'dev' and meas == measure and norm =='norm_':
                        costs.selection_metric = (ce*ce + cf*cf)**0.5
                        costs.selection_energy = ce
                        costs.selection_forces = cf
        self.current_costs = costs
        #Get the new params and give to vdw_dataframe
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
    
    def report(self):
        print(self.current_costs)
        try:
            print(self.timecosts)
        except AttributeError:
            print('time costs info not available')
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
        return 0.5*(regularizators.grad_ridge(x) + regularizators.grad_lasso(x))
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
        u = np.abs(u1-u2)
        return np.sum(u*w)/u2.size
    
    @staticmethod
    def grad_MAE(u1,u2,w=1):
        u = u1-u2
        return w*np.sign(u)/u2.size
    
    @staticmethod
    def MSE(u1,u2,w=1):
        u = u1 -u2
        return np.sum(w*u*u)/u2.size
    
    @staticmethod
    def grad_MSE(u1,u2,w=1):
        u = u1-u2
        return 2*w*u/u2.size

    @staticmethod
    def BIAS(u1,u2):
        u = u1-u2
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

    
    def plot_predict_vs_target(self,x_data,y_data,size=2.35,dpi=500,
                               xlabel=r'$E^{dft}$ (kcal/mol)', ylabel=r'$U^{class}$ (kcal/mol)',
                               title=None,
                               label_map=None,
                               path=None, fname='pred_vs_target.png',attrs=None,
                               save_fig=True,compare=None,scale=0.05):
        GeneralFunctions.make_dir(path)
        
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
            plt.plot(x_data.flatten(), y_data.flatten(),ls='None',color='purple',marker='.',markersize=1.3*size,fillstyle='none')
        else:
            for i,c in enumerate(uncol):
                f = col == c
                if label_map is not None:
                    lbl = label_map[c]
                else:
                    lbl = c
                plt.plot(x_data[f].flatten(), y_data[f].flatten(),label=lbl,ls='None',color=colors[i],
                         marker='.',markersize=1.8*size,fillstyle='none')
        plt.plot(perf_line,perf_line, ls='--', color='k',lw=size/2)
        plt.xlabel(xlabel,fontsize=3.5*size)
        plt.ylabel(ylabel,fontsize=3.5*size)
        if label_map is not None:
             plt.legend(frameon=False,fontsize=3.0*size)
        if fname is not None:
            plt.savefig('{:s}/{:s}'.format(path,fname),bbox_inches='tight')
        #plt.show()
        if compare is not None:
            for i,c in enumerate(uncol):
                _ = plt.figure(figsize=(size,size),dpi=dpi)
                f = col == c
                plt.minorticks_on()
                plt.tick_params(direction='in', which='minor',length=size)
                plt.tick_params(direction='in', which='major',length=2*size)
                xmin = x_data[f].min()
                xmax = x_data[f].max()
                air = scale*(xmax-xmin)
                perf_line = [xmin - air , xmax + air]
                plt.plot(perf_line,perf_line, ls='--', color='k',lw=size/2)
                
                plt.plot(x_data[f], y_data[f],label=c,ls='None',color=colors[i],
                             marker='.',markersize=1.8*size,fillstyle='none')
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.legend(frameon=False,ncol=ncol)
                if fname is not None:
                    pre,po = fname.split('.')
                    fn = f'{pre}_{c}.{po}'
                    plt.savefig('{:s}/{:s}'.format(path,fn),bbox_inches='tight')
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
    

class CostFunctions():   
    def Energy(params, Energy, models_list_info, measure, mu=0.0,std=1.0):
        ne = Energy.shape[0]
        
        Uclass = FF_Optimizer.computeUclass(params,ne,models_list_info)
        
        func = getattr(measures,measure)
        ce = func(  (Uclass-mu)/std, (Energy-mu)/std )
        
        return ce
    
    def gradEnergy(params, Energy, models_list_info, measure, mu=0.0, std=1.0):
        ne = Energy.shape[0]
        
        Uclass = FF_Optimizer.computeUclass(params,ne,models_list_info)
        gradU = FF_Optimizer.gradUclass(params,ne,models_list_info)
        
        func = getattr(measures,'grad_'+measure)
        grad = np.sum( func( (Uclass-mu)/std, (Energy-mu)/std ) * gradU/std, axis = 1)
        
        return grad 
    
    def Forces(params,Forces_True, models_list_info, measure, mu=0.0,std=1.0):
        n_forces = Forces_True.shape[0]
        
        Forces = FF_Optimizer.computeForceClass(params,n_forces,models_list_info)
        
        func = getattr(measures,measure)
        cf = func( (Forces-mu)/std, (Forces_True-mu)/std )
        return cf 
    
    def gradForces(params,Forces_True, models_list_info, measure, mu=0.0, std=1.0):
        n_forces = Forces_True.shape[0]
        
        Forces = FF_Optimizer.computeForceClass(params,n_forces,models_list_info)
        gradF = FF_Optimizer.computeGradForceClass(params,n_forces,models_list_info)
        
        func = getattr(measures,'grad_'+measure)
        
        grad = np.sum( func( (Forces -mu)/std, (Forces_True-mu)/std ) * gradF/std, axis = (1,2) )
        return grad
    
    def Regularization(params,reguls,reg_measure):
        cr = getattr(regularizators,reg_measure)(params*reguls)
        return cr
    
    def gradRegularization(params,reguls,reg_measure):
        gr = getattr(regularizators,'grad_'+reg_measure)(params*reguls)
        return gr
    
@jit(nopython=True,fastmath=True,parallel=True)
def numba_isin(x1,x2,f):
    for i in prange(x1.shape[0]):
        for x in x2:
            if x1[i] == x: 
                f[i] = True
    return


