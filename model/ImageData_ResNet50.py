from matplotlib import pyplot as plt
import os
import ktrain
import math
import parameter
import re
from ktrain import vision as vis
import tensorflow as tf
import keras
# from keras.optimizers import Adam
# from tensorflow import keras
from PIL import ImageFile
# from keras.models import Model
# from keras.layers import Dense, Flatten

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# Functions
def show_prediction(fname):
    fname = DATADIR + '/' + fname
    pred = round(predictor.predict_filename(fname)[0], 2)
    actual = p.search(fname).group(1)
    print("Predicted Price: %s | Actual Price: %s" % (pred, actual))

# Find optimal learning rate
class LRFind(tf.keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, n_rounds):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_up = (max_lr/min_lr) ** (1 / n_rounds)
        self.lrs = []
        self.losses = []

    def on_train_begin(self, logs=None):
        self.weights = self.model.get_weights()
        self.model.optimizer.lr = self.min_lr

    def on_train_batch_end(self, batch, logs=None):
        self.lrs.append(self.model.optimizer.lr.numpy())
        self.losses.append(logs["loss"])
        self.model.optimizer.lr = self.model.optimizer.lr * self.step_up
        if self.model.optimizer.lr > self.max_lr:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.model.set_weights(self.weights)


# Calculate difference between predicted and actual price
def calc_difference(fname):
    pred = round(predictor.predict_filename(fname)[0], 2)
    actual = float(p.search(fname).group(1))
    diff = abs(pred - actual)
    return diff


# Calculate difference for MAPE
def calc_difference_mape(fname):
    pred = round(predictor.predict_filename(fname)[0], 2)
    actual = float(p.search(fname).group(1))
    diff = abs(pred - actual)
    diff_mape = abs(diff / actual)
    return diff_mape


# Calculate MAE
def mae(test_data):
    differences = 0
    for element in test_data:
        fname = DATADIR + '/' + element
        differences += calc_difference(fname)
    mae = differences / len(test_data)
    return mae


# Calculate RMSE
def rmse(test_data):
    differences = 0
    for element in test_data:
        fname = DATADIR + '/' + element
        differences += (calc_difference(fname))**2
    rmse = math.sqrt(differences / len(test_data))
    return rmse


# Calculate MAPE
def mape(test_data):
    differences = 0
    for element in test_data:
        fname = DATADIR + '/' + element
        differences += calc_difference_mape(fname)
    mape = (differences / len(test_data)) * 100
    return mape



# Filter price information from filename
pattern = r'_([^/]+)_\d+_.jpg$'
p = re.compile(pattern)
r = p.search('_0.01_1099561093123_.jpg')
print(r.group(1))

# Specify directory
DATADIR = parameter.Baseline_ResNet50.dir

# Select train and test data
(train_data, test_data, preproc) = vis.images_from_fname(DATADIR, pattern = pattern,
                    is_regression = True,
                    random_state = 42)

# Selection of possible models
vis.print_image_regression_models()

# Model specification
pretrained_model = vis.image_regression_model('pretrained_resnet50',
                                   train_data = train_data,
                                  val_data = test_data,)

# Remove last three layers from pretrained model
pretrained_model = keras.models.Model(pretrained_model.input,
                         pretrained_model.layers[-3].output)

model = keras.models.Sequential()
model.add(pretrained_model)
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(1))

# Print complete model
print(model.summary())

# Specify learner with parameters
learner = ktrain.get_learner(model = model,
                            train_data = train_data,
                            val_data = test_data,
                           batch_size = 256)

# Compile learner
learner.model.compile(optimizer="Adam",
                      loss=tf.keras.losses.MeanAbsoluteError(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])

# Optional: optimal learning rate finder
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# lr_find = LRFind(1e-10, 1e1, 400)
# model.fit(
#     train_data,
#     steps_per_epoch=400,
#     epochs=1,
#     callbacks=[lr_find]
# )
#
# plt.plot(lr_find.lrs, lr_find.losses)
# plt.xscale('log')
# plt.xlabel('Learning Rate')
# plt.ylabel("MAE")
# plt.savefig("learning_rate.jpg")
# plt.show()


# Training of model
ImageFile.LOAD_TRUNCATED_IMAGES = True
# First layers are trained with weight adjustment
history = learner.fit_onecycle(lr=1e-4, epochs=10)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss_diagram.jpg')
plt.show()

# layers frozen
ImageFile.LOAD_TRUNCATED_IMAGES = True
learner.freeze(15)
# Last layers are trained with weight adjustment
ImageFile.LOAD_TRUNCATED_IMAGES = True
history2 = learner.fit_onecycle(lr=1e-4, epochs=10)

# Plot loss over epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Specify predictor
predictor = ktrain.get_predictor(learner.model, preproc)
# Optional: Save predictor
# ktrain.get_predictor(learner.model, preproc).save("predictor_bs128")

# Store test data in a list
validation_data = list(test_data.filenames)

# Calculate error metrics
mae_value = mae(validation_data)
rmse_value = rmse(validation_data)
mape_value = mape(validation_data)
print(str(mae_value), str(rmse_value), str(mape_value))

# Print predicted prices for first 50 test datapoints
for element in validation_data[:50]:
    show_prediction(element)


















