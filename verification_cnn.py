from numpy import loadtxt
import numpy
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from numpy import savetxt
import tensorflow
from numpy import mean

MY_CONST = 20.
MY_CONST_NEG = -20.

def NormalizeData(data):
	return (data + (MY_CONST)) / (MY_CONST - (MY_CONST_NEG))

# load the test data
#X = loadtxt('d:\\eeg_speller\\table_of_flashes_1_to_45_612_atharva_1g_4o.csv', delimiter=',')
#X = loadtxt('d:\\eeg_speller\\table_of_flashes_1_to_45_612_swati_3g.csv', delimiter=',')
X = loadtxt('d:\\eeg_speller\\table_of_flashes_1_to_45_612_ritu_6o.csv', delimiter=',')



mean_of_test = mean(X[:, 0:612])
print(mean_of_test)

# make the data zero mean
input = X[:, 0:612] - mean_of_test

too_high_input = input > MY_CONST
input[too_high_input] = MY_CONST
too_low_input = input < MY_CONST_NEG
input[too_low_input] = MY_CONST_NEG

# normalize the data between 0 and 1
input = numpy.apply_along_axis(NormalizeData, 1, input)
#input = NormalizeData(input)
savetxt('d:\\input-swati-online.csv', input, delimiter=',')


# transform the data in the format which the model wants
input = input.reshape(len(input), 4, 153)
input = input.transpose(0, 2, 1)

# get the expected outcome 
y_real = X[:, -1]

# load the model
model = load_model('D:\\eeg_speller\\model_conv1d.h5')

# get the "predicted class" outcome
y_pred = model.predict_proba(input) 
print(y_pred.shape)

#----------------------------------

y_corr = numpy.zeros((len(y_pred), 2))

for i in range(len(y_corr)):
	y_corr[i][1] = (y_pred[i][1] * 0.66)/((y_pred[i][1] * 0.66) + ((1 - y_pred[i][1]) * 1.32))
	y_corr[i][0] = ((1 - y_pred[i][1]) * 1.32)/((y_pred[i][1] * 0.66) + ((1 - y_pred[i][1]) * 1.32)) 

print(y_corr.shape)
print(y_corr)
#----------------------------------


y_max = numpy.argmax(y_pred, axis=1)

# calculate the confusion matrix
matrix = confusion_matrix(y_real, y_max)
print(matrix)
