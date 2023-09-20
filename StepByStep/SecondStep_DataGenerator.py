!pip install opencv-python
!pip install pydicom

import numpy as np
import pydicom
import cv2

def my_data_generator(df, sample_weights_birads, sample_weights_density, batch_size=150):
    # img_gen = ImageDataGenerator(rescale=1. / 255.)  # assuming images are in 0-255 range

    mapping_dict_density = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    mapping_dict_laterality = {'left': 0, 'right': 1}
    mapping_dict_view_position = {'CC': 0, 'MLO': 1}

    while True:
        # Select files (IDs) and labels for the batch
        batch_indices = np.random.choice(a=len(df), size=batch_size)
        batch_df = df.iloc[batch_indices]

        # Initialize batch arrays
        # batch_images = []
        # batch_labels_birads = []
        # batch_labels_density = []
        # batch_features = []
        # batch_weights_birads = []
        # batch_weights_density = []
        # Assuming height and width are the dimensions of your image, and num_features is the number of feature columns
        batch_size = 150  # The number of samples per batch
        height = 3518  # The height of your images
        width = 2800  # The width of your images
        num_features = 2  # Number of feature columns, here you have 'laterality' and 'view_position'

        # Initialize your arrays
        batch_images = np.zeros((batch_size, height, width), dtype=np.float32)  # Adjust dtype as needed
        batch_labels_birads = np.zeros((batch_size, 1), dtype=np.int)  # Assuming birads labels are integers
        batch_labels_density = np.zeros((batch_size, 1), dtype=np.int)  # Assuming density labels are integers
        batch_features = np.zeros((batch_size, num_features), dtype=np.int)  # Assuming features are integers
        batch_weights_birads = np.zeros((batch_size,), dtype=np.float32)  # Assuming weights are float numbers
        batch_weights_density = np.zeros((batch_size,), dtype=np.float32)  # Assuming weights are float numbers

        # ... and so on

        for i, original_idx in enumerate(batch_indices):
            row = batch_df.iloc[i]

            img_id = row['image_id']
            study_id = row['study_id']
            img_path = f"/content/drive/MyDrive/Colab/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images/{study_id}/{img_id}.dicom"
            img = pydicom.dcmread(img_path).pixel_array


            # Resize image based on the height and width from CSV if needed
            def conditional_resize(img, target_height=3518, target_width=2800):
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

                print("\n IMAGE:", img_id, original_height, original_width, output_img.shape)
                return output_img

            img = conditional_resize(img)

        #     # Append to batch
        #     batch_images.append(img)
        #     batch_labels_birads.append(row['breast_birads'] - 1)
        #     batch_labels_density.append(mapping_dict_density.get(row['breast_density'], -1))  # Use mapping here
        #     # batch_labels_density.append(row['breast_density'])
        #
        #     mapped_laterality = mapping_dict_laterality.get(row['laterality'], -1)  # -1 for missing or unknown
        #     mapped_view_position = mapping_dict_view_position.get(row['view_position'], -1)  # -1 for missing or unknown
        #
        #     batch_features.append([mapped_laterality, mapped_view_position])
        #
        #
        #
        #     batch_weights_birads.append(sample_weights_birads[original_idx])
        #     batch_weights_density.append(sample_weights_density[original_idx])
        #
        #
        # batch_images = np.array(batch_images)
        # batch_labels_birads = np.array(batch_labels_birads)
        # batch_labels_density = np.array(batch_labels_density)
        # batch_features = np.array(batch_features)
        # batch_weights_birads = np.array(batch_weights_birads)
        # batch_weights_density = np.array(batch_weights_density)
        #
        # batch_labels_birads = batch_labels_birads.reshape(-1, 1)  # Reshape to shape (batch_size, 1)
        # batch_labels_density = batch_labels_density.reshape(-1, 1)  # Reshape to shape (batch_size, 1)

        # Assign to pre-allocated NumPy arrays
        batch_images[i] = img
        batch_labels_birads[i, 0] = row['breast_birads'] - 1
        batch_labels_density[i, 0] = mapping_dict_density.get(row['breast_density'], -1)
        batch_features[i] = [mapping_dict_laterality.get(row['laterality'], -1),
                             mapping_dict_view_position.get(row['view_position'], -1)]
        batch_weights_birads[i] = sample_weights_birads[original_idx]
        batch_weights_density[i] = sample_weights_density[original_idx]

        print('\n birads_output', batch_labels_birads, ' Data type:',
              str(batch_labels_birads.dtype) if hasattr(batch_labels_birads, 'dtype') else "N/A",
              '\n density_output', batch_labels_density, ' Data type:',
              str(batch_labels_density.dtype) if hasattr(batch_labels_density, 'dtype') else "N/A")

        print("\n Shape of training image_input:", batch_images.shape, " Data type:",
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

        yield {'image_input': batch_images, 'feature_input': batch_features}, \
            {'birads_output': batch_labels_birads, 'density_output': batch_labels_density}, \
            {'birads_output': batch_weights_birads, 'density_output': batch_weights_density}
