import matplotlib
import os
import re
import ktrain
import parameter
import math
import pandas as pd
import matplotlib.pyplot as plt
from ktrain import vision as vis
from PIL import ImageFile
import tensorflow as tf
from keras.models import Model, load_model
from keras import Input
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from tensorflow.keras import backend as K


# Function to filter asset_id from filename
def filter_asset_id(original_list, pattern=r'_(\d+)_'):
    output_list = []
    for element in original_list:
        p = re.compile(pattern)
        r = p.search(element)
        output_list.append(int(r.group(1)))
    return output_list


pattern = r'_([^/]+)_\d+_.jpg$'

p = re.compile(pattern)
r = p.search('_0.01_1099561093123_.jpg')
print(r.group(1))

data_dir = parameter.ImageAttributes_ResNet50.dir

# Split dataset to training and test data
(train_data, test_data, preproc) = vis.images_from_fname(data_dir,
                    pattern = pattern,
                    is_regression = True,
                    random_state = 42)

# Get pretrained model from ktrain
pretrained_model = vis.image_regression_model('pretrained_resnet50',
                                   train_data = train_data,
                                  val_data = test_data,)

# Remove the last two layers of the pretrained model
pretrained_model = Model(pretrained_model.input,
                         pretrained_model.layers[-3].output)

# Add two dense layer and create model
# Give output layer specific name
model = Sequential()
model.add(pretrained_model)
model.add(Dense(512, activation="relu", name="attribute_layer"))
model.add(Dense(1, activation="relu"))

print(model.summary())

# Get learner
learner = ktrain.get_learner(model = model,
                            train_data = test_data,
                            val_data = train_data,
                           batch_size = 128)

# Compile learner
learner.model.compile(optimizer=Adam(learning_rate=1e-10),
                      loss=tf.keras.losses.MeanAbsoluteError(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError()])

# Train model
ImageFile.LOAD_TRUNCATED_IMAGES = True
learner.fit_onecycle(1e-10, 2)

ImageFile.LOAD_TRUNCATED_IMAGES = True
learner.freeze(15)
learner.fit_onecycle(1e-10, 2)

# Get predictor
predictor = ktrain.get_predictor(learner.model, preproc)

# Create predictor with specific layer as output and run
predictor2 = Model(inputs=predictor.model.input,
                   outputs=predictor.model.get_layer('attribute_layer').output)
ImageFile.LOAD_TRUNCATED_IMAGES = True
attributes_train = predictor2.predict(train_data)
attributes_test = predictor2.predict(test_data)

# Store output attributes in dataframe
df_train = pd.DataFrame(attributes_train)
df_test = pd.DataFrame(attributes_test)

# Use function to filter asset_id from filename
# Also store asset_ids in dataframes
train_data_list = filter_asset_id(list(train_data.filenames))
df_train["asset_id"] = train_data_list
test_data_list = filter_asset_id(list(test_data.filenames))
df_test["asset_id"] = test_data_list

# Combine output attributes and asset ids
df_ResNet = pd.concat([df_train, df_test])
df_ResNet.head()

# Load textual dataset
df_textual = pd.read_csv(parameter.ImageAttributes_ResNet50.textual_dataset_dir)

# Merge datasets on asset_id, drop empty rows and store file
df = pd.merge(df_textual, df_ResNet, on=['asset_id'], how="left")
df = df.dropna()
df.to_csv(parameter.ImageAttributes_ResNet50.final_dataset_dir)
















