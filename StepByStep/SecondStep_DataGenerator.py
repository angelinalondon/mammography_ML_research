!pip
install
opencv - python
!pip
install
pydicom

import numpy as np
import pydicom
import cv2
import gc
import random

height = 3518  # The height of your images
width = 2800  # The width of your images
num_features = 2  # Number of feature columns, here you have 'laterality' and 'view_position'


def conditional_resize(img, img_id, target_height=height, target_width=width):
    # Get the original image dimensions
    original_height, original_width = img.shape[:2]

    # Initialize the output image as the input image
    output_img = img

    # Case 1: If the original image matches the target dimensions, leave it as it is
    if original_height == target_height and original_width == target_width:
        output_img = img

    else:
        # Case 2: If the original image is smaller than the target dimensions, pad it
        if original_height < target_height and original_width < target_width:
            pad_h = target_height - original_height
            pad_w = target_width - original_width
            output_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        # Case 3: If the original image is larger than the target dimensions, crop it
        elif original_height > target_height and original_width > target_width:
            crop_h = (original_height - target_height) // 2
            crop_w = (original_width - target_width) // 2
            output_img = img[crop_h:crop_h + target_height, crop_w:crop_w + target_width]

        # Case 4: If original image HEIGHT is larger than target AND WIDTH is smaller than target
        elif original_height > target_height and original_width < target_width:
            # Crop the height
            crop_h = (original_height - target_height) // 2
            cropped_img = img[crop_h:crop_h + target_height, :]

            # Pad the width
            pad_w = target_width - original_width
            output_img = cv2.copyMakeBorder(cropped_img, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        # Case 5: If original image HEIGHT is smaller than target AND WIDTH is larger than target
        elif original_height < target_height and original_width > target_width:
            # Pad the height
            pad_h = target_height - original_height
            padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=0)

            # Crop the width
            crop_w = (original_width - target_width) // 2
            output_img = padded_img[:, crop_w:crop_w + target_width]

    print("resized IMAGE:", img_id, original_height, original_width, output_img.shape)
    return output_img


def preload_images(df, num_preloaded):
    print("preload images started")
    preloaded_images = {}

    # Shuffle the DataFrame and select the first num_preloaded rows
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    selected_df = shuffled_df.head(num_preloaded)

    for idx, row in selected_df.iterrows():
        img_id = row['image_id']
        study_id = row['study_id']
        img_path = f"/content/drive/MyDrive/Colab/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images/{study_id}/{img_id}.dicom"
        preloaded_images[img_id] = pydicom.dcmread(img_path).pixel_array
        print("preloaded image", img_id)

    print("preload images completed")
    return preloaded_images


def my_data_generator(df, batch_size, preloaded_images, feature_input, sample_weights_birads, sample_weights_density):
    print('\n =================\n',
          'entered my_data_generator'.upper(),
          '\n====================')

    mapping_dict_density = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    mapping_dict_laterality = {'left': 0, 'right': 1}
    mapping_dict_view_position = {'CC': 0, 'MLO': 1}
    filtered_df = df[df['image_id'].isin(preloaded_images.keys())]
    unused_indices = np.arange(len(filtered_df))

    while True:
        if len(unused_indices) < batch_size:
            # Reset unused_indices if there are not enough for a batch
            unused_indices = np.arange(len(filtered_df))

        batch_indices = np.random.choice(a=unused_indices, size=batch_size, replace=False)
        batch_df = filtered_df.iloc[batch_indices]

        # Remove used indices from unused_indices
        unused_indices = np.setdiff1d(unused_indices, batch_indices)

        # Initialize your arrays
        batch_images = np.zeros((batch_size, height, width), dtype=np.float32)  # Adjust dtype as needed
        batch_labels_birads = np.zeros((batch_size, 1), dtype=int)  # Assuming birads labels are integers
        batch_labels_density = np.zeros((batch_size, 1), dtype=int)  # Assuming density labels are integers
        batch_features = np.zeros((batch_size, num_features), dtype=int)  # Assuming features are integers
        batch_weights_birads = np.zeros((batch_size,), dtype=np.float32)  # Assuming weights are float numbers
        batch_weights_density = np.zeros((batch_size,), dtype=np.float32)  # Assuming weights are float numbers

        for i, original_idx in enumerate(batch_indices):
            print("\n i'm in FOR statement:", i, original_idx)
            row = batch_df.iloc[i]
            img_id = row['image_id']
            # if img_id in preloaded_images:
            img = preloaded_images[img_id]
            img = conditional_resize(img, img_id)
            # else:
            #     study_id = row['study_id']
            #     img_path = f"/content/drive/MyDrive/Colab/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images/{study_id}/{img_id}.dicom"
            #     img = pydicom.dcmread(img_path).pixel_array
            #     img = conditional_resize(img, img_id)

            # img = conditional_resize(img, img_id)
            batch_images[i] = img
            batch_labels_birads[i, 0] = row['breast_birads'] - 1
            batch_labels_density[i, 0] = mapping_dict_density.get(row['breast_density'], -1)
            batch_features[i] = [mapping_dict_laterality.get(row['laterality'], -1),
                                 mapping_dict_view_position.get(row['view_position'], -1)]
            batch_weights_birads[i] = sample_weights_birads[original_idx]
            batch_weights_density[i] = sample_weights_density[original_idx]
        #     WAITS changed their type from float32 to float64?

        print('AFTER RESIZING:',
              "\n Shape of training image_input:", batch_images.shape, " Data type:",
              str(batch_images.dtype) if hasattr(batch_images, 'dtype') else "N/A",
              "\n Shape of training features:", batch_features.shape, " Data type:",
              str(batch_features.dtype) if hasattr(batch_features, 'dtype') else "N/A",
              "\n Shape of training birads labels:", batch_labels_birads.shape, " Data type:",
              str(batch_labels_birads.dtype) if hasattr(batch_labels_birads, 'dtype') else "N/A",
              "\n Shape of training birads weights:", batch_weights_birads.shape, " Data type:",
              str(batch_weights_birads.dtype) if hasattr(batch_weights_birads, 'dtype') else "N/A",
              "\n Shape of training density labels:", batch_labels_density.shape, " Data type:",
              str(batch_labels_density.dtype) if hasattr(batch_labels_density, 'dtype') else "N/A",
              "\n Shape of training density weights:", batch_weights_density.shape, " Data type:",
              str(batch_weights_density.dtype) if hasattr(batch_weights_density, 'dtype') else "N/A",
              "\n batch_labels_birads", type(batch_labels_birads), 'batch_weights_density', type(batch_weights_density),
              "\n batch_labels_birads.shape", type(batch_labels_birads.shape), 'batch_weights_density.shape',
              type(batch_weights_density.shape))

        return {
            'image_input': batch_images,
            'feature_input': batch_features
        }, {
            'birads_output': batch_labels_birads,
            'density_output': batch_labels_density
        }, {
            'birads_output': batch_weights_birads,
            'density_output': batch_weights_density
        }

        del batch_images, batch_labels_birads, batch_labels_density, batch_features, batch_weights_birads, batch_weights_density
        gc.collect()
        print('gc.collect done')
