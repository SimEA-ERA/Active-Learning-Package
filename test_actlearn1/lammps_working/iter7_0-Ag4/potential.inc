# generated from FF_develop results
# created within AL scheme step by Nikolaos Patsalidis  
group Ag type 1


set type 1 charge 0.0





pair_style hybrid/overlay       morse 20.0 table linear 50000 local/density
pair_coeff 1 1 morse 2.9107630180000001e+01  1.1846702000000000e+00  2.6763038799999999e+00 
pair_coeff 1 1  table tablePW.tab Ag-Ag 
pair_coeff * * local/density frhos1.ld 



 
