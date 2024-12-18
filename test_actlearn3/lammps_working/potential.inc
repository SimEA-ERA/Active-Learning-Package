# generated from FF_develop results
# created within AL scheme step by Nikolaos Patsalidis  
group Ag type 1


set type 1 charge 0.0





pair_style hybrid/overlay       morse 20.0 table linear 50000 local/density
pair_coeff 1 1 morse 6.3053730000000003e+00  2.7373120200000001e+00  2.3051547600000002e+00 
pair_coeff 1 1  table tablePW.tab Ag-Ag 
pair_coeff * * local/density frhos1.ld 
