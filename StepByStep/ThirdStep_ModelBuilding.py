from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight




num_channels = 1  # Assuming grayscale images

standard_height, standard_width = 3518, 2800
input_layer = Input(shape=(standard_height, standard_width, num_channels))  # Dynamically adapt to any size

# Define CNN layers
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

# Flatten and dense layers
flatten = Flatten()(x)
birads_output = Dense(5, activation='softmax', name='birads_output')(flatten)
density_output = Dense(4, activation='softmax', name='density_output')(flatten)

# Complete the model
model = Model(inputs=input_layer, outputs=[birads_output, density_output])


model.compile(optimizer='adam',
              loss={'birads_output': 'sparse_categorical_crossentropy',
                    'density_output': 'sparse_categorical_crossentropy'},
              metrics={'birads_output': ['accuracy', tf.keras.metrics.AUC(name='auc_birads')],
                       'density_output': ['accuracy', tf.keras.metrics.AUC(name='auc_density')]})


def get_class_weights(y, smooth_factor=0.1):
    """
    Returns the weights for each class based on the frequencies of the samples.
    """
    counter = Counter(y)
    total = len(y)
    class_weights = {cls: freq for cls, freq in counter.items()}
    max_val = float(max(class_weights.values()))
    class_weights = {cls: max_val / (total * val) for cls, val in class_weights.items()}
    return class_weights

checkpoint = ModelCheckpoint(filepath='/content/drive/MyDrive/Colab/Callbacks/First_Attempt_Batches',
                             save_weights_only=True,
                             save_freq=1000)  # Save after every 1000 images (i.e., one batch)

def compute_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: float(majority/count) for cls, count in counter.items()}

birads_classes = [1, 2, 3, 4, 5]
birads_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(birads_classes), y = training_data['breast_birads'])
birads_weight_dict = {cls: weight for cls, weight in zip(birads_classes, birads_weights)}

mapping_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
training_data['breast_density'] = training_data['breast_density'].map(mapping_dict)
density_classes = [0, 1, 2, 3]
density_weights = compute_class_weight(class_weight = 'balanced', classes = density_classes, y = training_data['breast_density'])
density_weight_dict = {cls: weight for cls, weight in zip(density_classes, density_weights)}



history = model.fit(my_data_generator(training_data, batch_size=1000),
                    steps_per_epoch=len(training_data) // 1000,
                    epochs=10,
                    validation_data=my_data_generator(validation_data, batch_size=214),
                    validation_steps=len(validation_data) // 214,
                    callbacks=[checkpoint],
                    class_weight={'birads_output': birads_weight_dict, 'density_output': density_weight_dict})

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
