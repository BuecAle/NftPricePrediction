import matplotlib
import os
import ktrain
import math
import parameter
import re
from ktrain import vision as vis
from PIL import ImageFile


# Functions
def show_predition(fname):
    fname = DATADIR + '/' + fname
    pred = round(predictor.predict_filename(fname)[0])
    actual = p.search(fname).group(1)
    vis.show_image(fname)
    print("Predicted Price: %s | Actual Price: %s:" % (pred, actual))


def calc_difference(fname):
    pred = round(predictor.predict_filename(fname)[0], 2)
    actual = float(p.search(fname).group(1))
    diff = abs(pred - actual)
    return diff


def calc_difference_mape(fname):
    pred = round(predictor.predict_filename(fname)[0], 2)
    actual = float(p.search(fname).group(1))
    diff = abs(pred - actual)
    diff_mape = abs(diff / actual)
    return diff_mape


def mae(test_data):
    differences = 0
    for element in test_data:
        fname = DATADIR + '/' + element
        differences += calc_difference(fname)
    mae = differences / len(test_data)
    return mae


def rmse(test_data):
    differences = 0
    for element in test_data:
        fname = DATADIR + '/' + element
        differences += (calc_difference(fname))**2
    rmse = math.sqrt(differences / len(test_data))
    return rmse


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
model = vis.image_regression_model('pretrained_resnet50',
                                   train_data = train_data,
                                  val_data = test_data,)

print(model.summary())

# Specify learner with parameters
learner = ktrain.get_learner(model = model,
                            train_data = train_data,
                            val_data = test_data,
                           batch_size = 64)

# Training of model
ImageFile.LOAD_TRUNCATED_IMAGES = True
# First layers are trained with weight adjustment
learner.fit_onecycle(1e-4, 2)
# 15 layers frozen
learner.freeze(15)
# Last layers are trained with weight adjustment
learner.fit_onecycle(1e-4, 2)

# Specify predictor
predictor = ktrain.get_predictor(learner.model, preproc)

validation_data = list(test_data.filenames)


print(mae(validation_data))
print(rmse(validation_data))
print(mape(validation_data))























