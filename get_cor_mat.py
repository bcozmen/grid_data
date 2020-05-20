import glob
import scipy.io
import numpy as np


def matrix_correlation(M1, M2):
    if M1 is None or M2 is None:
        return 0
    return np.corrcoef(M1.reshape((1,M1.size)), M2.reshape( (1,M2.size) ) )[0,1]
    
def fc(BOLD):
    simFC = np.corrcoef(BOLD)
    simFC = np.nan_to_num(simFC) # remove NaNs
    return simFC

def filter_subcortical(a, axis = "both"):
    """
    Filter out subcortical areas out of aal2
    Hippocampus: 41 - 44
    Amygdala: 45-46
    Basal Ganglia: 75-80
    Thalamus: 81-82
    Cerebellum: 94-120
    """
    subcortical_index = np.array(list(range(40, 46)) + list(range(74, 82)))
    if axis == "both":
        a = np.delete(a, subcortical_index, axis=0)
        a = np.delete(a, subcortical_index, axis=1)
    else:
        a = np.delete(a, subcortical_index, axis=axis)
    return a


def load_matrices(directory = "../../Greifswald_fMRI_DTI", filt_subcort = False, return_index= -1):

    CmatFilenames = glob.glob(directory + "/NAP_*/SC/Normalized/DTI_CM.mat")
    DmatFilenames = glob.glob(directory + "/NAP_*/SC/Normalized/DTI_LEN.mat")
    FmatFilenames = glob.glob(directory + "/NAP_*/FC/FC.mat")
    TmatFilenames = glob.glob(directory + "/NAP_*/FC/TC.mat")

    CmatFilenames.sort()
    DmatFilenames.sort()
    FmatFilenames.sort()
    TmatFilenames.sort()


    Cmats = []
    Dmats = []
    Fmats = []
    Bolds = []
    for cm in CmatFilenames:
        this_cm = scipy.io.loadmat(cm)['sc']
        if filt_subcort:
            this_cm = filter_subcortical(this_cm)
        Cmats.append(this_cm)
    
    Cmat = np.zeros(Cmats[0].shape)
    
    for cm in Cmats:
        Cmat += cm
    Cmat /= len(Cmats)    
        
        
    for dm in DmatFilenames:
        this_dm = scipy.io.loadmat(dm)['len']
        if filt_subcort:
            this_dm = filter_subcortical(this_dm)
        Dmats.append(this_dm)
    
    Dmat = np.zeros(Dmats[0].shape)
    
    for dm in Dmats:
        Dmat += dm
    Dmat /= len(Dmats)

    if not filt_subcort:
        for fm,bold in zip(FmatFilenames,TmatFilenames):
            Fmats.append(scipy.io.loadmat(fm)['fc'])
            Bolds.append(scipy.io.loadmat(bold)['tc'])
    else:
        for tc in TmatFilenames:
            this_tc = scipy.io.loadmat(tc)['tc']
            this_tc = filter_subcortical(this_tc, axis=0)
            Bolds.append(this_tc)
            this_fc = fc(this_tc)
            Fmats.append(this_fc)

    if return_index == 10:
        return Cmat, Dmat, Fmats, Bolds
    elif return_index == -2:
        return Cmats, Dmats, Fmats, Bolds
    else:
        return Cmats[return_index], Dmats[return_index], Fmats, Bolds


_, _, empFCs, _ = load_matrices(return_index = 0)


fnames = glob.glob("*/")

corr_mat = np.ones(shape=(11,10,41,41,5,6))*-2


for file in fnames:
	
	bnames = glob.glob(file+"*")

	for bname in bnames:
		try :
			simBOLD = np.load(bname)
		except:
			continue
		
		if np.any(np.equal(simBOLD,None)):	
			continue
		
		simFC = fc(simBOLD)


		
		bname = (bname.split('/')[1]).replace(".npy","").split("_")
		this_sc = int(bname[0])
		I = int(bname[1])
		K = int(bname[2])
		c = int(bname[3])
		sigma = int(bname[4])

		for indx, empFC in enumerate(empFCs):
			corr_mat[this_sc, indx, I, K, c, sigma] = matrix_correlation(simFC, empFC)	

np.save("corr_mat.npy", corr_mat)
