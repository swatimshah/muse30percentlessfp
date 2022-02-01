import numpy
from imblearn.over_sampling import SMOTE
from numpy import savetxt
from numpy import loadtxt
from matplotlib import pyplot
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,AveragePooling1D
from numpy import mean
from tensorflow.random import set_seed
import tensorflow
import keras

def NormalizeData(data):
	return (data + abs(numpy.amin(data))) / (numpy.amax(data) - numpy.amin(data))


# setting the seed
seed(1)
set_seed(1)

# load array
X_train_whole = loadtxt('d:\\eeg_speller\\combined_50_612_1_to_45.csv', delimiter=',')

# augment data
choice_0 = X_train_whole[:, -1] == 0.
X_total_1 = numpy.append(X_train_whole, X_train_whole[choice_0, :], axis=0)

choice_1 = X_train_whole[:, -1] == 1.
X_total_2 = numpy.append(X_total_1, X_train_whole[choice_1, :], axis=0)

X_total = X_total_2
print(X_total.shape)

# data balancing
#sm = SMOTE(random_state = 2)
#X_train_res, Y_train_res = sm.fit_resample(X_total[:, 0:612], X_total[:, -1].ravel())
#print("After OverSampling, counts of label '1': {}".format(sum(Y_train_res == 1)))
#print("After OverSampling, counts of label '0': {}".format(sum(Y_train_res == 0)))

# split the data between training and validation
tensorflow.compat.v1.reset_default_graph()
X_train, X_test, Y_train, Y_test = train_test_split(X_total[:, 0:612], X_total[:, -1], random_state=1, test_size=0.3, shuffle = True)
print(X_train.shape)
print(X_test.shape)


#=======================================
 
# Data Pre-processing

input = numpy.zeros((0, 612))
testinput = numpy.zeros((0, 612))

MY_CONST = 20.
MY_CONST_NEG = -20.

mean_of_train = mean(X_train[:, 0:612])
print(mean_of_train)

# make the training data zero mean
input = X_train[:, 0:612] - mean_of_train

too_high_input = input > MY_CONST
input[too_high_input] = MY_CONST
too_low_input = input < MY_CONST_NEG
input[too_low_input] = MY_CONST_NEG

# normalize the training data
input = numpy.apply_along_axis(NormalizeData, 1, input)
#input = NormalizeData(input)

input_output = numpy.append(input, Y_train.reshape(len(Y_train), 1), axis=1) 
savetxt('d:\\input_output.csv', input_output, delimiter=',')

mean_of_test = mean(X_test[:, 0:612])
print(mean_of_test)

# make the test data zero mean
testinput = X_test[:, 0:612] - mean_of_test

too_high_testinput = testinput > MY_CONST
testinput[too_high_testinput] = MY_CONST
too_low_testinput = testinput < MY_CONST_NEG
testinput[too_low_testinput] = MY_CONST_NEG

# normalize the test data
testinput = numpy.apply_along_axis(NormalizeData, 1, testinput)
#testinput = NormalizeData(testinput)
savetxt('d:\\testinput.csv', testinput, delimiter=',')


#=====================================

# Model configuration

print(len(input))
print(len(testinput))

input = input.reshape(len(input), 4, 153)
input = input.transpose(0, 2, 1)
print (input.shape)

testinput = testinput.reshape(len(testinput), 4, 153)
testinput = testinput.transpose(0, 2, 1)
print (testinput.shape)

initialBias = numpy.log([2/4])

# Create the model
model=Sequential()
model.add(Conv1D(filters=20, kernel_size=10, padding='valid', activation='relu', strides=2, input_shape=(153, 4)))
model.add(Conv1D(filters=20, kernel_size=10, padding='valid', activation='relu', strides=2))
model.add(AveragePooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Conv1D(filters=40, kernel_size=4, padding='valid', activation='relu', strides=1))
model.add(GlobalAveragePooling1D())
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax', bias_initializer=keras.initializers.Constant(initialBias)))

model.summary()

# Compile the model
adam = Adam(learning_rate=0.003)       
model.compile(loss=sparse_categorical_crossentropy, optimizer=adam, metrics=['accuracy'])


hist = model.fit(input, Y_train, batch_size=64, epochs=500, verbose=1, validation_data=(testinput, Y_test), steps_per_epoch=None)		


# evaluate the model
Y_hat_classes = model.predict_classes(testinput)
matrix = confusion_matrix(Y_test, Y_hat_classes)
print(matrix)


# plot training and validation history
pyplot.plot(hist.history['loss'], label='tr_loss')
pyplot.plot(hist.history['val_loss'], label='val_loss')
pyplot.plot(hist.history['accuracy'], label='tr_accuracy')
pyplot.plot(hist.history['val_accuracy'], label='val_accuracy')
pyplot.legend()

pyplot.show()

#==================================

model.save("D:\\eeg_speller\\model_conv1d.h5")

#==================================

#Removed dropout and reduced momentum and reduced learning rate