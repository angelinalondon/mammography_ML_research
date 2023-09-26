from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
import collections
from collections import Counter

batch_size_training = 4
batch_size_validation = 1
preloaded_images_train = preload_images(training_data, num_preloaded=4)
preloaded_images_val = preload_images(validation_data, num_preloaded=1)



num_channels = 1  # Assuming grayscale images

standard_height, standard_width = 3518, 2800
image_input = Input(shape=(standard_height, standard_width, num_channels), name="image_input")

# Define CNN layers
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

# Flatten and dense layers
flatten = Flatten()(x)


# Define tabular data input layer for two features: 'laterality' and 'view_position'
feature_input = Input(shape=(2,), name='feature_input')  # Two features

# Concatenate flattened image data and feature data
merged = Concatenate()([flatten, feature_input])
#
# new_layer = Dense(12, activation='relu')(merged)  # or any number of units you want

birads_output = Dense(5, activation='softmax', name='birads_output')(merged)
density_output = Dense(4, activation='softmax', name='density_output')(merged)

# Complete the model
model = Model(inputs=[image_input, feature_input], outputs=[birads_output, density_output])

model.compile(optimizer='adam',
              loss={'birads_output': 'sparse_categorical_crossentropy',
                    'density_output': 'sparse_categorical_crossentropy'},
              metrics={'birads_output': ['accuracy', tf.keras.metrics.AUC(name='auc_birads')],
                       'density_output': ['accuracy', tf.keras.metrics.AUC(name='auc_density')]
                       },
              sample_weight_mode={
                    'birads_output': None,  # https://faroit.com/keras-docs/1.0.0/models/model/
                    'density_output': None
                    }
              )

checkpoint = ModelCheckpoint(filepath='/content/drive/MyDrive/Colab/Callbacks/First_Attempt_Batches',
                             save_weights_only=True,
                             save_freq=batch_size_training)  # Save after every 150 images (i.e., one batch)


def get_sample_weights(y):
    # Flatten the array back to a 1D array before using Counter
    y_flat = y.flatten()
    counter = Counter(y_flat)
    max_val = float(max(counter.values()))
    sample_weights = np.array([max_val / counter[i] for i in y_flat])
    return sample_weights


y_train_birads = training_data['breast_birads'].values
y_train_density = training_data['breast_density'].values

y_val_birads = validation_data['breast_birads'].values
y_val_density = validation_data['breast_density'].values

# Reshape the labels
y_train_birads = y_train_birads.reshape(-1, 1)  # Reshape to shape (n_samples_train, 1)
y_train_density = y_train_density.reshape(-1, 1)  # Reshape to shape (n_samples_train, 1)

y_val_birads = y_val_birads.reshape(-1, 1)  # Reshape to shape (n_samples_val, 1)
y_val_density = y_val_density.reshape(-1, 1)  # Reshape to shape (n_samples_val, 1)



sample_weights_birads_train = get_sample_weights(y_train_birads.flatten())
sample_weights_density_train = get_sample_weights(y_train_density.flatten())

sample_weights_birads_val = get_sample_weights(y_val_birads.flatten())
sample_weights_density_val = get_sample_weights(y_val_density.flatten())


# Map the class weights to the individual samples in the dataset
sample_weights_train = {'birads_output': sample_weights_birads_train,
                        'density_output': sample_weights_density_train}

sample_weights_val = {'birads_output': sample_weights_birads_val,
                      'density_output': sample_weights_density_val}

print('\n sample_weights_val', sample_weights_val,'\n sample_weights_train', sample_weights_train )
print('\n _____________ \n',
      "\n Shape of model output:", model.output_shape, " Data type:", str(model.output.dtype) if hasattr(model.output, 'dtype') else "N/A",
      "\n Shape of image_input:", image_input.shape, " Data type:", str(image_input.dtype) if hasattr(image_input, 'dtype') else "N/A",
      "\n Shape of feature_input:", feature_input.shape, " Data type:", str(feature_input.dtype) if hasattr(feature_input, 'dtype') else "N/A",
      "\n Shape of birads label:", y_train_birads.shape, " Data type:", str(y_train_birads.dtype) if hasattr(y_train_birads, 'dtype') else "N/A",
      "\n Shape of birads weight:", sample_weights_birads_train.shape, " Data type:", str(sample_weights_birads_train.dtype) if hasattr(sample_weights_birads_train, 'dtype') else "N/A",
      "\n Shape of density label:", y_train_density.shape, " Data type:", str(y_train_density.dtype) if hasattr(y_train_density, 'dtype') else "N/A",
      "\n Shape of density weight:", sample_weights_density_train.shape, " Data type:", str(sample_weights_density_train.dtype) if hasattr(sample_weights_density_train, 'dtype') else "N/A",
      '\n _____________ \n')

model.summary()


history = model.fit(
    my_data_generator(
        image_input,
        batch_size_training,
        preloaded_images_train,
        feature_input,
        birads_output,
        density_output,
        sample_weights_birads_train,
        sample_weights_density_train
    ),
    epochs = 2,
    verbose = 2,
    validation_data = my_data_generator(
        image_input,
        batch_size_validation,
        preloaded_images_val,
        feature_input,
        birads_output,
        density_output,
        sample_weights_birads_val,
        sample_weights_density_val
    ),
        callbacks=[checkpoint]
)



print('Model fit is completed')

# Get AUC values
train_auc_birads = history.history['birads_output_auc_birads']
val_auc_birads = history.history['val_birads_output_auc_birads']

train_auc_density = history.history['density_output_auc_density']
val_auc_density = history.history['val_density_output_auc_density']

print('I am trying to print AUC')
# Plotting the training and validation AUC for birads_output
plt.figure()
plt.plot(train_auc_birads, label='Training AUC for birads_output')
plt.plot(val_auc_birads, label='Validation AUC for birads_output')
plt.title('Training and Validation AUC for birads_output')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.show()

# Plotting the training and validation AUC for density_output
plt.figure()
plt.plot(train_auc_density, label='Training AUC for density_output')
plt.plot(val_auc_density, label='Validation AUC for density_output')
plt.title('Training and Validation AUC for density_output')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.show()

model.save_weights('/content/drive/MyDrive/Colab/model_saved')
print('Model Saved!')