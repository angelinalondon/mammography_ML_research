from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
import collections
from collections import Counter




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

new_layer = Dense(12, activation='relu')(merged)  # or any number of units you want

birads_output = Dense(5, activation='softmax', name='birads_output')(new_layer)
density_output = Dense(4, activation='softmax', name='density_output')(new_layer)

# Complete the model
model = Model(inputs=[image_input, feature_input], outputs=[birads_output, density_output])

model.compile(optimizer='adam',
              loss={'birads_output': 'sparse_categorical_crossentropy',
                    'density_output': 'sparse_categorical_crossentropy'},
              metrics={'birads_output': ['accuracy', tf.keras.metrics.AUC(name='auc_birads')],
                       'density_output': ['accuracy', tf.keras.metrics.AUC(name='auc_density')]
                       })


# def get_class_weights(y, smooth_factor=0.1):
#     """
#     Returns the weights for each class based on the frequencies of the samples.
#     """
#     counter = Counter(y)
#     total = len(y)
#     class_weights = {cls: freq for cls, freq in counter.items()}
#     max_val = float(max(class_weights.values()))
#     class_weights = {cls: max_val / (total * val) for cls, val in class_weights.items()}
#     return class_weights
#

checkpoint = ModelCheckpoint(filepath='/content/drive/MyDrive/Colab/Callbacks/First_Attempt_Batches',
                             save_weights_only=True,
                             save_freq=150)  # Save after every 150 images (i.e., one batch)

#
# mapping_dict_birads = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
# training_data['breast_birads'] = training_data['breast_birads'].map(mapping_dict_birads)
# birads_classes = [0, 1, 2, 3, 4]
# birads_weights = compute_class_weight(class_weight='balanced', classes=np.unique(birads_classes),
#                                       y=training_data['breast_birads'])
#
# birads_weight_dict = {cls: weight for cls, weight in zip(birads_classes, birads_weights)}
#
#
#
# #
# mapping_dict_density = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
# training_data['breast_density'] = training_data['breast_density'].map(mapping_dict_density)
# density_classes = [0, 1, 2, 3]
# density_weights = compute_class_weight(class_weight='balanced', classes=density_classes,
#                                        y=training_data['breast_density'])
# density_weight_dict = {cls: weight for cls, weight in zip(density_classes, density_weights)}
#

# birads_weight_dict_class_weight = {k: v for k, v in birads_weight_dict.items() if k > 1}

# print("\n ====================\n birads_weight_dict \n", class_weight)



def get_sample_weights(y):
    counter = Counter(y)
    max_val = float(max(counter.values()))
    sample_weights = np.array([max_val / counter[i] for i in y])
    return sample_weights

y_train_birads = training_data['breast_birads'].values
y_train_density = training_data['breast_density'].values

y_val_birads = validation_data['breast_birads'].values
y_val_density = validation_data['breast_density'].values



# Assuming y_train_birads and y_train_density contain your actual training labels
sample_weights_birads_train = get_sample_weights(y_train_birads)
sample_weights_density_train = get_sample_weights(y_train_density)

sample_weights_birads_val = get_sample_weights(y_val_birads)
sample_weights_density_val = get_sample_weights(y_val_density)

# Map the class weights to the individual samples in the dataset
sample_weights_train = {'birads_output': sample_weights_birads_train,
                        'density_output': sample_weights_density_train}

sample_weights_val = {'birads_output': sample_weights_birads_val,
                      'density_output': sample_weights_density_val}

print('\n sample_weights_val', sample_weights_val,'\n sample_weights_train', sample_weights_train )
print("Shape of model output:", model.output_shape)
model.summary()

history = model.fit(
    my_data_generator(
        training_data,
        batch_size=150,
        sample_weights_birads=sample_weights_birads_train,
        sample_weights_density=sample_weights_density_train,
    ),
    steps_per_epoch=len(training_data) // 150,
    epochs=10,
    validation_data=my_data_generator(
        validation_data,
        batch_size=32,
        sample_weights_birads=sample_weights_birads_val,
        sample_weights_density=sample_weights_density_val,
    ),
    validation_steps=len(validation_data) // 32,
    callbacks=[checkpoint]
)



# Get AUC values
train_auc_birads = history.history['birads_output_auc_birads']
val_auc_birads = history.history['val_birads_output_auc_birads']

train_auc_density = history.history['density_output_auc_density']
val_auc_density = history.history['val_density_output_auc_density']

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