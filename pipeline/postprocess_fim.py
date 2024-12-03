import numpy as np
import glob
import matplotlib.pyplot as plt
src_numb = 0

cov = [np.load(el) for el in glob.glob(f"./science_obj/source_{src_numb}_*/*cov.npy")]
params = [np.load(el) for el in glob.glob(f"./science_obj/source_{src_numb}_*/*params.npy")]
# combine covariance matrices
relative_precision = np.asarray([np.sqrt(np.diag(cov[ii]))/np.delete(params[ii], [5, 12]) for ii in range(len(cov))])
mean_rel_prec = np.mean(relative_precision,axis=0)
std_rel_prec = np.std(relative_precision, axis=0)
param_names = np.array(['M','mu','a','p0','e0','xI0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0'])
param_names = np.delete(param_names, [5, 12])
print("parameter, mean relative precision, std relative precision")
for ii in range(len(mean_rel_prec)):
    print(f"{param_names[ii]}: {mean_rel_prec[ii]} +/- {std_rel_prec[ii]}")