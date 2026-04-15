#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
import numpy as np 
import pandas as pd 
import glob
from random import randint
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
   try:
      tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
   except RuntimeError as e:
      # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
      print(e)



def convert_to_float(data):
    # Check if the input is a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to numpy array
        print("Convert DataFrame to numpy array")
        data = data.to_numpy()
    
    # Ensure the data is now a numpy array
    if isinstance(data, np.ndarray):
        # Check the data type of the elements
        print("Check the data type of the elements")
        if data.dtype.type is np.str_ or data.dtype.type is np.object_:
            # Try to convert the strings to floats
            print("Try to convert the strings to floats")
            try:
                data = data.astype(np.float64)
            except ValueError:
                raise ValueError("Data contains non-numeric strings that cannot be converted to float.")
        else:
            print("numpy array float")
    else:
        # If the data is not a numpy array, it might be a single string value or a list of strings
        if isinstance(data, (str, list)):
            # If it's a list of strings, convert each element to float
            if isinstance(data, list):
                print("If it's a list of strings, convert each element to float")
                try:
                    data = np.array([float(item) for item in data], dtype=np.float64)
                except ValueError:
                    raise ValueError("Data contains non-numeric strings that cannot be converted to float.")
            else:
                # If it's a single string, try to convert it to a float
                print("If it's a single string, try to convert it to a float")
                try:
                    data = float(data)
                except ValueError:
                    raise ValueError("Data is a non-numeric string that cannot be converted to float.")
        else:
            raise TypeError("Input data must be a pandas DataFrame, numpy array, or a string/list of numbers.")
    
    return data
	
# 절대값의 표준 편차로 정규화
def normalize_with_magnitude(I_data, Q_data):
    magnitude = np.sqrt(I_data**2 + Q_data**2)
    std_magnitude = np.std(magnitude)
    I_data_normalized = I_data / std_magnitude
    Q_data_normalized = Q_data / std_magnitude
    return I_data_normalized, Q_data_normalized
	
# past data preperatioa
file1_I	 = 'input_iq_I.csv'
file1_Q	 = 'input_iq_Q.csv'
#allfile1 = 'input_0_QAM_16_rician_4.csv'
def f(x):
    return np.complex(x)
f2 = np.vectorize(f)

N = 32

#all_files = glob.glob("data*_2/input_1_QAM_16_rician_*_o.csv")
#all_files_1 = glob.glob("data11/input_1_QAM_16_rician_*_o.csv")
#all_files_sort = sorted(all_files)
#all_files_sort_1 = sorted(all_files_1)

#inputData =[];
#inputData_1 =[];
#for filename in all_files_sort:
#    df = pd.read_csv(filename, index_col=None, header=None)
#    inputData.append(df)

#inputData = pd.concat(inputData, axis=0, ignore_index=True)

#for filename_1 in  all_files_sort_1:
#    df_1 = pd.read_csv(filename_1, index_col=None, header=None)
#    inputData_1.append(df_1)

#inputData_1 = pd.concat(inputData_1, axis=0, ignore_index=True)
inputData_I = pd.read_csv(file1_I,header=None)
inputData_Q = pd.read_csv(file1_Q,header=None)

#frame = pd.concat(inputData, axis=0, ignore_index=True);

#inputData = pd.DataFrame(inputData)

#inputData.columns=['input']


print("Complete import data")
inputData_I = inputData_I.to_numpy()
#inputData_I = inputData_I.transpose()
inputData_Q = inputData_Q.to_numpy()
#inputData_Q = inputData_Q.transpose()
len_i = len(inputData_I)
len_q = len(inputData_Q)
print("----------------------------")
print(inputData_I.shape, len_i)
print(inputData_Q.shape, len_q)
print("----------------------------")
inputData_I_train = inputData_I[0:int(0.8*len_i)];
inputData_I_test = inputData_I[int(0.8*len_i):len_i];
inputData_Q_train = inputData_Q[0:int(0.8*len_q)];
inputData_Q_test = inputData_Q[int(0.8*len_q):len_q];

#inputData = inputData.to_numpy()
#inputData = inputData.transpose()
#inputData = f2(inputData)

#inputData_1 = inputData_1.to_numpy()
#inputData_1 = inputData_1.transpose()
#inputData_1 = f2(inputData_1)

#inputData_train = inputData.reshape(len(all_files_sort),512000);
#inputData_test = inputData_1.reshape(len(all_files_sort_1),512000);
#for idx in range(0,len(all_files_sort)):
#    test1 = inputData_train[idx]/np.std((inputData_train[idx]))
#    inputData_train[idx] = test1;
    #input("Train ends. Enter")
#for idx in range(0,len(all_files_sort_1)):
#    test2 = inputData_test[idx]/np.std((inputData_test[idx]))
#    inputData_test[idx] = test2;
    #input("Test ends. Enter")

# DataFrame을 numpy array로 변환 후 복소수로 변환
# intputData_train_complex = inputData_train.values.astype(complex)
# intputData_test_complex = inputData_test.values.astype(complex)
inputData_I_train_float = convert_to_float(inputData_I_train)
inputData_Q_train_float = convert_to_float(inputData_Q_train)
inputData_I_test_float = convert_to_float(inputData_I_test)
inputData_Q_test_float = convert_to_float(inputData_Q_test)

# inputData_train_abs = np.abs(intputData_train_complex)
# inputData_train = intputData_train_complex / np.std(inputData_train_abs)
# inputData_test_abs = np.abs(intputData_test_complex)
# inputData_test = intputData_test_complex / np.std(inputData_test_abs)
inputData_I_train, inputData_Q_train = normalize_with_magnitude(inputData_I_train_float, inputData_Q_train_float)
inputData_I_test, inputData_Q_test = normalize_with_magnitude(inputData_I_test_float, inputData_Q_test_float)
#input("Train ends. Enter")
#inputData_train = inputData_train.transpose().reshape(32*500*len(all_files_sort),32);
#inputData_test = inputData_test.transpose().reshape(32*500*len(all_files_sort_1),32);
#inputData_train = inputData_train.transpose().reshape(-1, 64)
#inputData_test = inputData_test.transpose().reshape(-1, 64)
#iD_train_r = inputData_train.real
#iD_train_i = inputData_train.imag
iD_train_r = inputData_I_train.transpose().reshape(-1, N)
iD_train_i = inputData_Q_train.transpose().reshape(-1, N)


#print(iD_train_r[0:10])
#print(iD_train_i[0:10])

#iD_test_r = inputData_test.real
#iD_test_i = inputData_test.imag
iD_test_r = inputData_I_test.transpose().reshape(-1, N)
iD_test_i = inputData_Q_test.transpose().reshape(-1, N)

# distinguish feature, targeta

#all_files1 = glob.glob("data*_2/input_1_QAM_16_rician_*_Non.csv")
#all_files1_1 = glob.glob("data11/input_1_QAM_16_rician_*_Non.csv")

#all_files1_sort = sorted(all_files1)
#all_files1_sort_1 = sorted(all_files1_1)

outputData =[];
outputData_1 =[];

#for filename in all_files1_sort:
#    df = pd.read_csv(filename, index_col=None, header=None)
#    outputData.append(df)

#outputData = pd.concat(outputData, axis=0, ignore_index=True)

#for filename_1 in all_files1_sort_1:
#    df = pd.read_csv(filename_1, index_col=None, header=None)
#    outputData_1.append(df)

#outputData_1 = pd.concat(outputData_1, axis=0, ignore_index=True)




#inputData = pd.read_csv(file1,header=None)



file2_I = 'input_I_r.csv'
file2_Q = 'input_Q_r.csv'
outputData_I = pd.read_csv(file2_I,header=None)
outputData_Q = pd.read_csv(file2_Q,header=None)
print(outputData_I.shape)
print(outputData_Q.shape)

#outputData=outputData
#outputData_1=outputData_1
outputData_I = outputData_I.to_numpy()
#outputData_I = outputData_I.transpose()
outputData_Q = outputData_Q.to_numpy()
#outputData_Q = outputData_Q.transpose()

len_o_I = len(outputData_I)
len_o_Q = len(outputData_Q)
outputData_train_I = outputData_I[0:int(0.8*len_o_I)];
outputData_train_Q = outputData_Q[0:int(0.8*len_o_Q)];
outputData_test_I = outputData_I[int(0.8*len_o_I):len_o_I];
outputData_test_Q = outputData_Q[int(0.8*len_o_Q):len_o_Q];

#outputData = outputData.to_numpy()
#outputData = outputData.transpose()
#outputData = f2(outputData)


#outputData_1 = outputData_1.to_numpy()
#outputData_1 = outputData_1.transpose()
#outputData_1 = f2(outputData_1)

#outputData_train = outputData.reshape(len(all_files1_sort),512000);
#outputData_test = outputData_1.reshape(len(all_files1_sort_1),512000);


#for idx in range(0,len(all_files1_sort)):
#    test1 = outputData_train[idx]/np.std((outputData_train[idx]));
#    outputData_train[idx] = test1;
#for idx in range(0,len(all_files1_sort_1)):
#    test2 = outputData_test[idx]/np.std((outputData_test[idx]));
#    outputData_test[idx] = test2;
# DataFrame을 numpy array로 변환 후 복소수로 변환
#outputData_train_complex = outputData_train.values.astype(complex)
#outputData_test_complex = outputData_test.values.astype(complex)
target_I_train_float = convert_to_float(outputData_train_I)
target_Q_train_float = convert_to_float(outputData_train_Q)
target_I_test_float = convert_to_float(outputData_test_I)
target_Q_test_float = convert_to_float(outputData_test_Q)
#outputData_train_abs = np.abs(outputData_train_complex)
#outputData_train = outputData_train_complex / np.std(outputData_train_abs)
#outputData_test_abs = np.abs(outputData_test_complex)
#outputData_test = outputData_test_complex / np.std(outputData_test_abs)
target_I_train, target_Q_train = normalize_with_magnitude(target_I_train_float, target_Q_train_float)
target_I_test, target_Q_test = normalize_with_magnitude(target_I_test_float, target_Q_test_float)

#input("Train ends. Enter")
#outputData_train = outputData_train.transpose().reshape(32*500*len(all_files1_sort),32);
#outputData_test = outputData_test.transpose().reshape(32*500*len(all_files1_sort_1),32);
#outputData_train = outputData_train.transpose().reshape(-1, 64)

#oD_train_r = outputData_train.real
#oD_train_i = outputData_train.imag
oD_train_r = target_I_train.transpose().reshape(-1, N)
oD_train_i = target_Q_train.transpose().reshape(-1, N)

#oD_test_r = outputData_test.real
#oD_test_i = outputData_test.imag
oD_test_r = target_I_test.transpose().reshape(-1, N)
oD_test_i = target_Q_test.transpose().reshape(-1, N)

iterations = 500


#print(len(oD_train_r[0]))
#print(len(iD_train_r[0]))



print(len(oD_train_r))
print(len(iD_train_r))

# First, let's define a RNN Cell, as a layer subclass.


#print(outputData_train.shape)

iD_test_r=np.expand_dims(iD_test_r, axis=-1)
#iD_test_r=np.expand_dims(iD_test_r, axis=-1)
#iD_test_i=np.expand_dims(iD_test_i, axis=-1)
iD_test_i=np.expand_dims(iD_test_i, axis=-1)

iD_train_r=np.expand_dims(iD_train_r, axis=-1)
iD_train_i=np.expand_dims(iD_train_i, axis=-1)
oD_train_r=np.expand_dims(oD_train_r, axis=-1)
oD_train_i=np.expand_dims(oD_train_i, axis=-1)

#iD_train_r=np.expand_dims(iD_train_r, axis=-1)
#iD_train_i=np.expand_dims(iD_train_i, axis=-1)
#oD_train_r=np.expand_dims(oD_train_r, axis=-1)
#oD_train_i=np.expand_dims(oD_train_i, axis=-1)

#outputData_test=np.expand_dims(outputData_test, axis=-1)
#print(outputData_test.shape)


oD_test_r=np.expand_dims(oD_test_r, axis=-1)
#oD_test_i=np.expand_dims(oD_test_i, axis=-1)
#oD_test_r=np.expand_dims(oD_test_r, axis=-1)
oD_test_i=np.expand_dims(oD_test_i, axis=-1)




#LSTM

model2_r = keras.Sequential()
model2_i = keras.Sequential()
#model2_r.add(layers.LSTM(units=64, return_sequences=True,
#                           input_shape=(oD_train_r.shape[0],1)))
#model2_i.add(layers.LSTM(units=64, return_sequences=True,
#                           input_shape=(oD_train_i.shape[0],1)))

#model2_r.add(layers.Dropout(0.2))
#model2_i.add(layers.Dropout(0.2))
#model2_r.add(layers.LSTM(20, return_sequences=False))
#model2_i.add(layers.LSTM(20, return_sequences=False))
#model2_r.add(layers.Dropout(0.2))
#model2_i.add(layers.Dropout(0.2))

#model2_r.add(layers.Dense(1))
#model2_i.add(layers.Dense(1))
#model2_r.add(layers.Dropout(0.2))
#model2_i.add(layers.Dropout(0.2))

#model2_r.add(layers.Bidirectional(layers.LSTM(units=70, return_sequences=True, dropout=0.2,
#                           input_shape=(oD_train_r.shape[1],1))))
#model2_i.add(layers.Bidirectional(layers.LSTM(units=70, return_sequences=True,dropout=0.2,
#                           input_shape=(oD_train_i.shape[1],1))))

#model2_r.add(layers.Bidirectional(layers.LSTM(90,dropout=0.2)))
#model2_i.add(layers.Bidirectional(layers.LSTM(90,dropout=0.2)))

#model2_r.add(layers.Bidirectional(layers.LSTM(units=70, return_sequences=True, input_shape=(oD_train_r.shape[1],1), dropout=0.4, recurrent_dropout=0.4,
#    kernel_regularizer=keras.regularizers.l2(0.001))))
#model2_i.add(layers.Bidirectional(layers.LSTM(units=70, return_sequences=True, input_shape=(oD_train_r.shape[1],1), dropout=0.4, recurrent_dropout=0.4,
#    kernel_regularizer=keras.regularizers.l2(0.001))))
#model2_r.add(layers.BatchNormalization())
#model2_r.add(layers.Bidirectional(layers.LSTM(units=68, return_sequences=True, input_shape=(oD_train_r.shape[1],1))))
model2_r.add(layers.Bidirectional(layers.LSTM(units=90, return_sequences=True, input_shape=(oD_train_r.shape[1],1))))
#model2_i.add(layers.BatchNormalization())
#model2_i.add(layers.Bidirectional(layers.LSTM(units=68, return_sequences=True, input_shape=(oD_train_i.shape[1],1))))
model2_i.add(layers.Bidirectional(layers.LSTM(units=90, return_sequences=True, input_shape=(oD_train_i.shape[1],1))))
#model2_r.add(layers.Dropout(0.5))
#model2_i.add(layers.Dropout(0.5))

#model2_r.add(layers.Bidirectional(layers.LSTM(90)))
#model2_r.add(layers.Bidirectional(layers.LSTM(80)))
model2_r.add(layers.Bidirectional(layers.LSTM(90)))
#model2_i.add(layers.Bidirectional(layers.LSTM(90)))
#model2_i.add(layers.Bidirectional(layers.LSTM(80)))
model2_i.add(layers.Bidirectional(layers.LSTM(90)))
#model2_r.add(layers.Dropout(0.5))
#model2_i.add(layers.Dropout(0.5))
model2_r.add(layers.Dense(N))
model2_i.add(layers.Dense(N))
#model2_r.add(layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001)))
#model2_i.add(layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001)))
#model2_r.add(layers.Dropout(0.5))
#model2_i.add(layers.Dropout(0.5))
model2_r.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
model2_i.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
#optimizer_r = keras.optimizers.RMSprop(learning_rate=0.0001)
#optimizer_i = keras.optimizers.RMSprop(learning_rate=0.0001)
#model2_r.compile(loss='mse', optimizer=optimizer_r, metrics=['mae'])
#model2_i.compile(loss='mse', optimizer=optimizer_i, metrics=['mae'])
#model2_r.summary()
#model2_i.summary()


print(oD_train_r.shape)
print(oD_train_i.shape)
print(iD_train_r.shape)
print(iD_train_i.shape)
n_epoches = 1000;
epoch_num=[1, 1, 1, 2, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10] #, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50] 
epoch_num2=[1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] #,150,200,250,300,350,400,450,500,550,600,650,700] 
for idx in range(0,len(epoch_num)):
    idx1 = randint(0,9)
    #o1_r = oD_train_r[0][idx1*512000:(idx1+2)*512000] 
    #o1_i = oD_train_i[0][idx1*512000:(idx1+2)*512000]
    o1_r = oD_train_r 
    o1_i = oD_train_i
    i1_r = iD_train_r
    i1_i = iD_train_i
    print("Real Part")
#    hist_r = model2_r.fit(o1_r, i1_r, epochs=epoch_num[idx], batch_size=512)
    hist_r = model2_r.fit(i1_r, o1_r, epochs=epoch_num[idx], batch_size=128)
    print("Imaginary Part")
    print(i1_i.shape, o1_i.shape)
#    hist_i = model2_i.fit(o1_i, i1_i, epochs=epoch_num[idx], batch_size=512)
    hist_i = model2_i.fit(i1_i, o1_i, epochs=epoch_num[idx], batch_size=128)
#    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#    hist_r = model2_r.fit(i1_r, o1_r, epochs=epoch_num[idx], batch_size=512, validation_split=0.2, callbacks=[early_stopping])
#    hist_i = model2_i.fit(i1_i, o1_i, epochs=epoch_num[idx], batch_size=512, validation_split=0.2, callbacks=[early_stopping])

    idx1 = randint(0,9)
    #res2_r = model2_r.evaluate(oD_test_r[0][idx1*512000:(idx1+2)*512000], iD_test_r[0], batch_size=32)
    #res2_i = model2_i.evaluate(oD_test_i[0][idx1*512000:(idx1+2)*512000], iD_test_i[0], batch_size=32)
    #res2_r = model2_r.evaluate(oD_test_r, iD_test_r, batch_size=512)
    #res2_i = model2_i.evaluate(oD_test_i, iD_test_i, batch_size=512)
    name_r = 'vlc_lstm_model9_'+str(epoch_num2[idx])+'_r.h5'
    name_i = 'vlc_lstm_model9_'+str(epoch_num2[idx])+'_i.h5'
    print(name_r, name_i)
    model2_r.save(name_r)
    model2_i.save(name_i)
    #print("loss",res2_r[0],"mae",res2_r[1])
    #print("loss",res2_i[0],"mae",res2_i[1])
    print("evaluate shape: ", iD_test_r.shape, oD_test_r.shape)
    model2_r.evaluate(iD_test_r, oD_test_r)
    model2_i.evaluate(iD_test_i, oD_test_i)


#xhat2_r = oD_test_r
#xhat2_i = oD_test_i
xhat2_r = iD_test_r
xhat2_i = iD_test_i
yhat2_r = model2_r.predict(xhat2_r)
yhat2_i = model2_i.predict(xhat2_i)
#yhat2_c =zeros(len(yhat2_r), complex);
#yhat2_c.real = yhat2_r;
#yhat_c.imag = yhat2_i;

print("yhat2_r[0:100]\n", yhat2_r[0:100, 0])
print("oD_test_r[0:100]\n", oD_test_r[0:100])
#print("Evaluate : {}".format(np.average((yhat2_r[0:1024] - iD_test_n[0:1024].real)**2)))
print("Evaluate : {}".format(np.average((yhat2_r[0:1024] - oD_test_r[0:1024].real.squeeze(-1))**2)))

#plt.figure()
#day=30
#plt.plot(yhat_r[0:100], label = "RNN")
#plt.plot(yhat2_r[0:100], label = "LSTM")
#plt.plot(iD_test_n[0:100].real,label = "actual")
#plt.savefig("ex3.png")




