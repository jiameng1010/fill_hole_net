import matplotlib.pyplot as plt
import numpy as np

curve_val = np.empty(shape=(40))
curve = np.empty(shape=(40))
for i in range(40):
    curve_val[i] = 0
    curve[i] = 0
#for i in range(1, 39):
#    filetrain_baseline = './trained_models_baseline/model_epoch_train' + str(i) + '.npy'
#    filetrain = './trained_models/model_epoch_train' + str(i) + '.npy'
#    loss_baseline = np.load(filetrain_baseline)
#    loss = np.load(filetrain)
#    curve_base[i] = loss_baseline.item()['conv2d_21_metric_L1_real'][0]
#    curve[i] = loss.item()['add_6_metric_L1_real'][0]

for i in range(1, 40):
    filetrain_val = '../../exp_data/trained_models/model_epoch_val' + str(i) + '.npy'
    filetrain = '../../exp_data/trained_models/model_epoch_train' + str(i) + '.npy'
    loss_val = np.load(filetrain_val)
    loss = np.load(filetrain)
    #curve_base[i] = loss_baseline.item()['add_1_loss'][0]
    curve[i] = loss.item()['conv2d_21_loss'][0]
    #curve_base[i] = loss_baseline[6]
    curve_val[i] = loss_val[6]

plt.plot(curve_val)
plt.plot(curve)
plt.show()



print('\n')