"""
Spectacle_python : An implementation of a spectral algorithm for Hidden Markov Models. The input file consists of sample (cross) moments computed from chromatin mark datasets (e.g. from ENCODE). The output file consists of Hidden Markov Model parameters (initial state distribution, emission matrix and transition matrix). This is a module inside the main Spectacle program which is written in Java and based on the ChromHMM software. Command line instructions:

python Spectacle_python.py file_directory fileID num_states num_marks (min_occurrence_observation)

* Example 

java -jar Spectacle.jar LearnModel -nobed -nobrowser -noenrich -f "hg19_inputfilelist.txt" -i spectral -lambda 1 -p 4 -computesamplemomentonly SAMPLEDATA_HG19 OUTPUTSAMPLE_HG19 20 hg19

python Spectacle_python.py OUTPUTSAMPLE_HG19 spectral 20 8	 

java -jar Spectacle.jar MakeSegmentation -f "hg19_inputfilelist.txt" -i Spectacle -comb "OUTPUTSAMPLE_HG19/model_comb_20_spectral.txt" SAMPLEDATA_HG19 OUTPUTSAMPLE_HG19

"""

import re
import sys
import scipy as sp
import scipy.sparse as ss 
import scipy.sparse.linalg as ssl 
import numpy as np
import numpy.linalg as nl

chromosomes = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrX']



def getSampleMomentsFromFile(nobs,sample_moment_file,cutoff=1):
	""" 
	Read in sample moments from the input file.
 	We remove observations whose occurrence in the genome is less than a cutoff.
	If cutoff==-1, we include all possible observations.
	"""
	f = open(sample_moment_file)
	line = f.readline()
	line = f.readline()
	nsegment = int(line.rstrip())
	line = f.readline()
	marks = []
	for line in f:
		if (line=='single_vector\n'):	break
		marks.append(line.rstrip())	
	M = 2**nobs

	Single = ss.dok_matrix((M,1))
	for line in f:
		if (line=='initial_single_vector\n'):	break
		tokens = re.split("[\t\n]",line)
		i = int(tokens[0])
		Single[i,0] = float(tokens[1])

	initSingle0 = ss.dok_matrix((M,1))
	observation = []
	for line in f:
		if (line=='pair21_matrix\n'):	break
		tokens = re.split("[\t\n]",line)
		i = int(tokens[0])
		if (round(Single[i,0]*nsegment,0)<cutoff):	continue
		initSingle0[i,0] = float(tokens[1])
		observation.append(i)

	if cutoff==-1:
		observation = range(0,M)

	M2 = len(observation)
#	print 'Number of non-zero combinations: '+str(M2)
	print 'Number of >='+str(cutoff)+' combinations: '+str(M2)

	transObs = {}
	initSingle = ss.dok_matrix((M2,1))
	for i in range(0,M2):
		transObs[observation[i]] = i	
		initSingle[i,0] = initSingle0[observation[i],0]

	Pair21 = ss.dok_matrix((M2,M2))
	for line in f:
		if (line=='pair31_matrix\n'):	break
		tokens = re.split("[\t\n]",line)
		if ((round(Single[int(tokens[0]),0]*nsegment,0)<cutoff)|(round(Single[int(tokens[1]),0]*nsegment,0)<cutoff)):	continue
		ID2 = transObs[int(tokens[0])]
		ID1 = transObs[int(tokens[1])]
		Pair21[ID2,ID1] = float(tokens[2])

	Pair31 = ss.dok_matrix((M2,M2))
	for line in f:
		if (line=='triple31_matrix\n'):	break
		tokens = re.split("[\t\n]",line)
		if ((round(Single[int(tokens[0]),0]*nsegment,0)<cutoff)|(round(Single[int(tokens[1]),0]*nsegment,0)<cutoff)):	continue
		ID3 = transObs[int(tokens[0])]
		ID1 = transObs[int(tokens[1])]
		Pair31[ID3,ID1] = float(tokens[2])	

	Triple = {}
	t = 0
	for line in f:
		tokens = re.split("[\t\n]",line)
		if ((round(Single[int(tokens[0]),0]*nsegment,0)<cutoff)|(round(Single[int(tokens[1]),0]*nsegment,0)<cutoff)|(round(Single[int(tokens[2]),0]*nsegment,0)<cutoff)):	continue
		ID2 = transObs[int(tokens[0])]
		if (ID2 not in Triple):	Triple[ID2] = ss.dok_matrix((M2,M2))
		ID3 = transObs[int(tokens[1])]
		ID1 = transObs[int(tokens[2])]
		Triple[ID2][ID3,ID1] = float(tokens[3])
		t += 1
	f.close()
#	print 'Number of pairs21: '+str(Pair21.getnnz())
#	print 'Number of pairs31: '+str(Pair31.getnnz())
#	print 'Number of triples: '+str(t)
	print 'read sample moment data'
	return (transObs,M2,marks,initSingle,Single,Pair21,Pair31,Triple)
			

def SpectralLearning(M2,K,initSingle,Single,Pair21,Pair31,Triple,transObs,marks,emission_file):
	""" The main spectral learning algorithm """
	transObs2 = {}
	for i in transObs.keys():
		transObs2[transObs[i]] = i

#	'Computes SVD of observation pairs matrix'
	(U2,S2,V_T2) = ssl.svds(Pair31,K)
	U_T2 = np.matrix(U2.transpose())
	V_T2 = np.matrix(V_T2)
	U_T = np.matrix(np.zeros((K,M2)))
	S = np.zeros((K))
	V = np.matrix(np.zeros((M2,K)))
	for i in range(0,K):
		U_T[i,:] = U_T2[K-i-1,:]*np.sign(U_T2[K-i-1,0])
		S[i] = S2[K-i-1]
		V[:,i] = (V_T2[K-i-1,:]*np.sign(V_T2[K-i-1,0])).transpose()

# Pair_inv3 = (U^T * Pair31)^+
	Pair_inv3 = ss.dok_matrix((M2,K))
	for j in range(0,K):
		for i in range(0,M2):
			Pair_inv3[i,j] = V[i,j]/S[j]

#	'Finds the major observation by using U'
	mIdx = {}
	exist = {}
	for i in range(0,K):
		mV = 0
		mI = 0
		for j in range(0,M2):	
			if (j in exist):	continue
			v = U_T[i,j]**2
			if (v>mV):
				mV = v
				mI = j
		mIdx[i] = mI
		exist[mI] = True
#		print i,transObs2[mIdx[i]]

# 'Computes eigenvectors'
	eigVec = np.matrix(np.zeros((K,K)))
	for i in range(0,K):
		Cx = U_T * Triple[mIdx[i]] * Pair_inv3
#		[w,v] = eigs(Cx,1,maxiter=miter)
		[Ws,Vs] = nl.eig(Cx)
		v = Vs[:,0]
		eigVec[:,i] = np.matrix(np.real(v)*np.sign(np.real(v[0,0])))
	eigVec_inv = nl.inv(eigVec)

#	'Computes emission matrix'
	O = ss.dok_matrix((M2,K))
	O1 = eigVec_inv * U_T
	O2 = Pair_inv3 * eigVec
	for i in range(0,M2):
		if (i not in Triple):	continue
		Ox = O1 * Triple[i] * O2
		for j in range(0,K):	O[i,j] = abs(Ox[j,j])
	for j in range(0,K):	O[:,j] /= O[:,j].sum()

# 'Write emission matrix to the output file'
	f = open(emission_file,'w+')
	f.write(str(K)+'\t'+str(transObs.__len__())+'\t(Emission order)')
	for mark in marks:	f.write('\t'+mark)
	f.write('\n')
	for j in range(0,K):
		for i in sorted(transObs.keys()):
			f.write(str(j+1)+'\t'+str(i)+'\t'+str(O[transObs[i],j])+'\n')
	f.close()
#	U_T_O = U_T * O

#	'Computes Inverse of Emission Matrix'
#	O_inv = matrix(pinv(O.todense())) 
#	Pi = O_inv * initSingle
	A = ssl.inv((O.transpose()*O).tocsc())

# 'Computes initial state distribution vector'
	Pi = A * (O.transpose()*initSingle)
	s = abs(Pi).sum()
	for i in range(0,K):	
		Pi[i,0] = abs(Pi[i,0])/s	

#	'Computes transition matrix'
#	T = abs(O_inv*Pair21*O_inv.transpose()*inv(diagMatrix(Pi).todense()))
#	T1 = O_inv*Pair31
#	T2 = pinv(O_inv*Pair21)
#	T = abs(T1 * T2)
	T3 = O.transpose()*Pair31
	T2 = O.transpose()*Pair21
	T2_inv = ssl.inv((T2*T2.transpose()).tocsc())
	A_inv = ssl.inv(A)
	T = abs(A*(T3*T2.transpose())*T2_inv*A_inv)
	for j in range(0,K):	T[:,j] /= T[:,j].sum() 
	return (T,O,Pi)



def diagMatrix(V):
	n = max(np.shape(V)[0],np.shape(V)[1])
	n2 = min(np.shape(V)[0],np.shape(V)[1])
	if (n2!=1):	
		print 'Error: it is not a vector'
		return V
	if (np.shape(V)[1]==n):	V = V.transpose()
	M = ss.dok_matrix((n,n))
	for i in range(0,n):	M[i,i] = V[i,0]
	return M


def diagSum(M):
	n = min(np.shape(M)[0],np.shape(M)[1])
	c = 0
	for i in range(0,n):	c += M[i,i]
	return c


def writeModel(T,O,Pi,nobs,marks,transObs,model_file):
	(M2,K) = np.shape(O)
	f = open(model_file,'w+')
	f.write(str(K)+'\t'+str(nobs)+'\tU\t0\t0\n')
	for i in range(0,K):
		f.write('probinit\t'+str(i+1)+'\t'+str(Pi[i,0])+'\n')
	for j in range(0,K):
		for i in range(0,K):
			f.write('transitionprobs\t'+str(j+1)+'\t'+str(i+1)+'\t'+str(T[i,j])+'\n')
	for j in range(0,K):
		for i in sorted(transObs.keys()):
			f.write('emissionprobs\t'+str(j+1)+'\t'+str(i)+'\t'+str(O[transObs[i],j])+'\n')
	f.close()



if __name__ == "__main__":
	nargv = sys.argv.__len__()
	if ((nargv!=5)&(nargv!=6)):
		print 'Error: wrong number of arguments'
		print 'python Spectacle_python.py file_directory fileID num_states num_marks (min_occurrence_observation)'
		exit(1)
	filedir = sys.argv[1]
	outputfileID = sys.argv[2]
	K = int(sys.argv[3])
	nobs = int(sys.argv[4])
	if (nargv==6):	cutoff = int(sys.argv[5])
	else:	cutoff = 1
	sample_moment_file = filedir+'/sample_moments_'+outputfileID+'_'+str(nobs)+'.txt'
	(transObs,M2,marks,initSingle,Single,Pair21,Pair31,Triple) = getSampleMomentsFromFile(nobs,sample_moment_file,cutoff)
	if (cutoff>1):	outputfileID2 = outputfileID+'_moo'+str(cutoff)
	else:	outputfileID2 = outputfileID
	emission_file = filedir+'/emissions_comb_'+str(K)+'_'+outputfileID2+'.txt'
	(T,O,Pi) = SpectralLearning(M2,K,initSingle,Single,Pair21,Pair31,Triple,transObs,marks,emission_file)
	model_file = filedir+'/model_comb_'+str(K)+'_'+outputfileID2+'.txt'
	writeModel(T,O,Pi,nobs,marks,transObs,model_file)






