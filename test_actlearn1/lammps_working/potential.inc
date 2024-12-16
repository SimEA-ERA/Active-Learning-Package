# generated from FF_develop results
# created within AL scheme step by Nikolaos Patsalidis  
group Ag type 1


set type 1 charge 0.0





pair_style hybrid/overlay       morse 20.0 table linear 50000 local/density
pair_coeff 1 1 morse 1.0697959950000000e+01  2.4740122000000001e-01  1.1200000000000001e+00 
pair_coeff 1 1  table tablePW.tab Ag-Ag 
pair_coeff * * local/density frhos1.ld 
