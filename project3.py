import numpy as np
from matplotlib import pyplot as plt
import sklearn as skl


from coding import single_response_frequency
pi = np.pi

#Generate tuning curve
N = 3
n_stim = 300
stimuli = np.linspace(0, 2*pi, n_stim)
n_trials = 200

responses = np.empty([N, n_trials, n_stim])

for n in range(N):
    for i in range(n_trials):
        responses[n, i] = single_response_frequency(stimuli, neuronid=n)

avg_responses = responses.mean(axis=1)

f1 = plt.figure(figsize=(10,5))
plt.plot(stimuli, avg_responses[::2].T);
plt.xlabel("stimuli")
plt.ylabel("average response")
f1.show()

'''
Effect of correlations

The response of a neuron to a stimulus is noisy, like everything else in a biological system.

When the stimulus is the same on both neurons, it can enforce similar responses (if the tuning curves are similar), or very different responses (if they're very different). This gives rise to signal-driven correlations.

Noise-driven correlations can be present too, when the fluctuations in the activity of a neuron are dependent on the ones of the others. Why would they be? Because neurons belong to a network, and are correlated to the network's background activity.
'''

st1 = 2.  # orientation of the two stimuli
st2 = st1 + 1.
n0, n1 = 0, 1  # choose two neurons
npoints = 100

#f2 = plt.figure()
stims1 = st1*np.ones(npoints)
resp0 = single_response_frequency(stims1, n0)
resp1 = single_response_frequency(stims1, n1)

#plt.scatter(resp0, resp1, alpha=0.4,
#            label="Stimulus orientation {} rad".format(st1))

stims2 = st2*np.ones(npoints)
resp0 = single_response_frequency(stims2, n0)
resp1 = single_response_frequency(stims2, n1)
#plt.scatter(resp0, resp1, alpha=0.4,
 #           label="Stimulus orientation {} rad".format(st2))

#plt.ylim([0, 100])
#plt.xlim([0, 100])
#plt.xlabel("Response of neuron {}".format(n0))
#plt.ylabel("Response of neuron {}".format(n1))
#plt.legend();
#f2.show()

from scipy.stats.stats import pearsonr

noise_intensity = 1
angle = pi/4

#generate noise
np.random.seed(42)

#noise2 = np.random.poisson(1, npoints)
#print(noise1)
#var_noise = pearsonr(noise1, noise2)
#print var_noise
#noise = np.random.poisson(1,100)
#print(noise)

noise_array = [1, 10, 20, 30 ,40, 50]
noise1 = np.random.normal(0, noise_intensity, size=(1, npoints))[0]
noise2 = np.random.normal(0, noise_intensity, size=(1, npoints))[0]
#classifier

# define perceptron
from sklearn.linear_model import Perceptron
from sklearn import svm
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
accList = []
perceptron_accuracy = []

for i in noise_array:
	
	f3 = plt.figure()
	resp0S1 = single_response_frequency(stims1, n0) + noise1*i
	resp1S1 = single_response_frequency(stims1, n1) + noise2*i
	plt.scatter(resp0S1, resp1S1, alpha=0.4,
            label="Stimulus orientation {} rad".format(st1))

	resp0S2 = single_response_frequency(stims2, n0) + noise1*i
	resp1S2 = single_response_frequency(stims2, n1) + noise2*i
	plt.scatter(resp0S2, resp1S2, alpha=0.4,
            label="Stimulus orientation {} rad".format(st2))

	plt.ylim([0, 150])
	plt.xlim([0, 150])
	plt.xlabel("Response of neuron {}".format(n0))
	plt.ylabel("Response of neuron {}".format(n1))
	plt.legend();
	fig_name = "Noise amplitude = " + str(i) + ", Variance = " + str(0.1) 	
	plt.title(fig_name)
	f3.show()

	
	ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    
	xTrain = []
	label = []
	for e in range(npoints):
    		xTrain.append([resp0S1[e], resp1S1[e]])
    		label.append(0)
    
	for e in range(npoints):
    		xTrain.append([resp0S2[e], resp1S2[e]])
    		label.append(1)
    
	yLabel = np.array(label)
	ppn.fit(xTrain, yLabel)
	SVMM = svm.SVC(kernel='linear', C=1).fit(xTrain, yLabel)


	# testing the model
	#print("Accuracy of Perceptron" , ppn.score(xTrain, yLabel))
	#print("Accuracy of SVM" , SVMM.score(xTrain, yLabel))
	# testing measures

	predicted = cross_val_predict(SVMM,xTrain , yLabel, cv=10)
	#print("accuracy_score", metrics.accuracy_score(yLabel, predicted))
	#print("f1_score", metrics.f1_score(yLabel, predicted))
	accList.append(SVMM.score(xTrain, yLabel))
	perceptron_accuracy.append(ppn.score(xTrain, yLabel))	
print accList
print perceptron_accuracy

'''
#for variance sensitivity analysis
noise_varr= [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4]
for i in noise_varr:
	noisevar1 = np.random.normal(0, i, size=(1, npoints))[0]
	noisevar2 = np.random.normal(0, i, size=(1, npoints))[0]
	f3 = plt.figure()
	resp0S1 = single_response_frequency(stims1, n0) + noisevar1*20
	resp1S1 = single_response_frequency(stims1, n1) + noisevar2*20
	plt.scatter(resp0S1, resp1S1, alpha=0.4,
            label="Stimulus orientation {} rad".format(st1))

	resp0S2 = single_response_frequency(stims2, n0) + noisevar1*20
	resp1S2 = single_response_frequency(stims2, n1) + noisevar2*20
	plt.scatter(resp0S2, resp1S2, alpha=0.4,
            label="Stimulus orientation {} rad".format(st2))

	plt.ylim([0, 150])
	plt.xlim([0, 150])
	plt.xlabel("Response of neuron {}".format(n0))
	plt.ylabel("Response of neuron {}".format(n1))
	plt.legend();
	fig_name = "Noise amplitude = 20" + ", Variance = " + str(i) 	
	plt.title(fig_name)
	f3.show()

	
	ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    
	xTrain = []
	label = []
	for e in range(npoints):
    		xTrain.append([resp0S1[e], resp1S1[e]])
    		label.append(0)
    
	for e in range(npoints):
    		xTrain.append([resp0S2[e], resp1S2[e]])
    		label.append(1)
    
	yLabel = np.array(label)
	ppn.fit(xTrain, yLabel)
	SVMM = svm.SVC(kernel='linear', C=1).fit(xTrain, yLabel)


	# testing the model
	#print("Accuracy of Perceptron" , ppn.score(xTrain, yLabel))
	#print("Accuracy of SVM" , SVMM.score(xTrain, yLabel))
	# testing measures

	predicted = cross_val_predict(SVMM,xTrain , yLabel, cv=10)
	#print("accuracy_score", metrics.accuracy_score(yLabel, predicted))
	#print("f1_score", metrics.f1_score(yLabel, predicted))
	accList.append(SVMM.score(xTrain, yLabel))	
print accList
'''
'''
#for different correlations given the same amplitude of noise
alpha_corr= [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1]
noisecorr1 = np.random.normal(0,1.3,100)*15
noisecorr2 = np.random.normal(0,1.3,100)*15

for i in range(len(alpha_corr)):
	noisecorrf = alpha_corr[i] * noisecorr1 + noisecorr2
	f4 = plt.figure()
	resp0S1 = single_response_frequency(stims1, n0) + noisecorr1
	resp1S1 = single_response_frequency(stims1, n1) + noisecorrf
	plt.scatter(resp0S1, resp1S1, alpha=0.4,
            label="Stimulus orientation {} rad".format(st1))

	resp0S2 = single_response_frequency(stims2, n0) + noisecorr1
	resp1S2 = single_response_frequency(stims2, n1) + noisecorrf
	plt.scatter(resp0S2, resp1S2, alpha=0.4,
            label="Stimulus orientation {} rad".format(st2))

	plt.ylim([0, 150])
	plt.xlim([0, 150])
	plt.xlabel("Response of neuron {}".format(n0))
	plt.ylabel("Response of neuron {}".format(n1))
	plt.legend();
	fig_name = "corr =" + str(alpha_corr[i])
	plt.title(fig_name)
	f4.show()

	
	ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    
	xTrain = []
	label = []
	for e in range(npoints):
    		xTrain.append([resp0S1[e], resp1S1[e]])
    		label.append(0)
    
	for e in range(npoints):
    		xTrain.append([resp0S2[e], resp1S2[e]])
    		label.append(1)
    
	yLabel = np.array(label)
	ppn.fit(xTrain, yLabel)
	SVMM = svm.SVC(kernel='linear', C=1).fit(xTrain, yLabel)


	# testing the model
	#print("Accuracy of Perceptron" , ppn.score(xTrain, yLabel))
	#print("Accuracy of SVM" , SVMM.score(xTrain, yLabel))
	# testing measures

	predicted = cross_val_predict(SVMM,xTrain , yLabel, cv=10)
	accList.append(SVMM.score(xTrain, yLabel))	
print accList
'''

'''
#correlated noise
#noise1 = np.random.normal(0, noise_intensity, size=(1, npoints))[0]
noisecorr1 = np.random.normal(0,1.3,100)
noisecorr2 = np.random.normal(0,1.3,100)
noise2 = 0.8 * noisecorr1 + noisecorr2
corr_noise = pearsonr(noisecorr1, noise2)[0]
#print(pearsonr(noisecorr1, noise2)
noise_array = [1, 10, 20, 30, 40, 50]
accList = []
for i in noise_array:
	
	f3 = plt.figure()
	resp0S1 = single_response_frequency(stims1, n0) + noisecorr1*i
	resp1S1 = single_response_frequency(stims1, n1) + noise2*i
	plt.scatter(resp0S1, resp1S1, alpha=0.4,
            label="Stimulus orientation {} rad".format(st1))

	resp0S2 = single_response_frequency(stims2, n0) + noisecorr1*i
	resp1S2 = single_response_frequency(stims2, n1) + noise2*i
	plt.scatter(resp0S2, resp1S2, alpha=0.4,
            label="Stimulus orientation {} rad".format(st2))

	plt.ylim([0, 150])
	plt.xlim([0, 150])
	plt.xlabel("Response of neuron {}".format(n0))
	plt.ylabel("Response of neuron {}".format(n1))
	plt.legend();
	fig_name = "Noise amplitude = " + str(i) + ", Correlation = " + str(corr_noise)  	
	plt.title(fig_name)
	f3.show()

	
	ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    
	xTrain = []
	label = []
	for e in range(npoints):
    		xTrain.append([resp0S1[e], resp1S1[e]])
    		label.append(0)
    
	for e in range(npoints):
    		xTrain.append([resp0S2[e], resp1S2[e]])
    		label.append(1)
    
	yLabel = np.array(label)
	ppn.fit(xTrain, yLabel)
	SVMM = svm.SVC(kernel='linear', C=1).fit(xTrain, yLabel)


	# testing the model
	#print("Accuracy of Perceptron" , ppn.score(xTrain, yLabel))
	#print("Accuracy of SVM" , SVMM.score(xTrain, yLabel))
	# testing measures

	predicted = cross_val_predict(SVMM,xTrain , yLabel, cv=10)
	#print("accuracy_score", metrics.accuracy_score(yLabel, predicted))
	#print("f1_score", metrics.f1_score(yLabel, predicted))
	accList.append(SVMM.score(xTrain, yLabel))	
print accList
'''
'''
noise_varr= [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4]
for i in noise_varr:
	noisevar1 = np.random.normal(0, i, size=(1, npoints))[0]
	noisevar2 = np.random.normal(0, i, size=(1, npoints))[0]
	noisevar3 = 0.8 * noisevar1 + noisevar2
	corr_noise = pearsonr(noisevar1, noisevar3)[0]
	corr_noise = round(corr_noise, 2)
	f3 = plt.figure()
	resp0S1 = single_response_frequency(stims1, n0) + noisevar1*20
	resp1S1 = single_response_frequency(stims1, n1) + noisevar3*20
	plt.scatter(resp0S1, resp1S1, alpha=0.4,
            label="Stimulus orientation {} rad".format(st1))

	resp0S2 = single_response_frequency(stims2, n0) + noisevar1*20
	resp1S2 = single_response_frequency(stims2, n1) + noisevar3*20
	plt.scatter(resp0S2, resp1S2, alpha=0.4,
            label="Stimulus orientation {} rad".format(st2))

	plt.ylim([0, 150])
	plt.xlim([0, 150])
	plt.xlabel("Response of neuron {}".format(n0))
	plt.ylabel("Response of neuron {}".format(n1))
	plt.legend();
	fig_name = "Noise amplitude = 20" + ", Variance = " + str(i) + ", Correlation = " + str(corr_noise)	
	plt.title(fig_name)
	f3.show()

	
	ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    
	xTrain = []
	label = []
	for e in range(npoints):
    		xTrain.append([resp0S1[e], resp1S1[e]])
    		label.append(0)
    
	for e in range(npoints):
    		xTrain.append([resp0S2[e], resp1S2[e]])
    		label.append(1)
    
	yLabel = np.array(label)
	ppn.fit(xTrain, yLabel)
	SVMM = svm.SVC(kernel='linear', C=1).fit(xTrain, yLabel)


	# testing the model
	#print("Accuracy of Perceptron" , ppn.score(xTrain, yLabel))
	#print("Accuracy of SVM" , SVMM.score(xTrain, yLabel))
	# testing measures

	predicted = cross_val_predict(SVMM,xTrain , yLabel, cv=10)
	#print("accuracy_score", metrics.accuracy_score(yLabel, predicted))
	#print("f1_score", metrics.f1_score(yLabel, predicted))
	accList.append(SVMM.score(xTrain, yLabel))	
print accList
'''

raw_input()

