# python test_model.py model=iphone_orig dped_dir=dped/ test_subset=full iteration=all resolution=orig use_gpu=true

import imageio.v2 as imageio
from PIL import Image
import numpy as np
import tensorflow as tf
from models import resnet
import utils
import os
import sys

tf.compat.v1.disable_v2_behavior()

# process command arguments
phone, dped_dir, test_subset, iteration, resolution, use_gpu = utils.process_test_model_args(sys.argv)

# get all available image resolutions
res_sizes = utils.get_resolutions()

# get the specified image resolution
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE = utils.get_specified_res(res_sizes, phone, resolution)

# disable gpu if specified
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if use_gpu == "false" else None

# create placeholders for input images
x_ = tf.compat.v1.placeholder(tf.float32, [None, IMAGE_SIZE])
x_image = tf.reshape(x_, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

# generate enhanced image
enhanced = resnet(x_image)

with tf.compat.v1.Session(config=config) as sess:

    test_dir = dped_dir + phone.replace("_orig", "") + "/test_data/full_size_test_images/"
    test_photos = [f for f in os.listdir(test_dir) if os.path.isfile(test_dir + f)]

    if test_subset == "small":
        # use five first images only
        test_photos = test_photos[0:5]

    if phone.endswith("_orig"):

        # load pre-trained model
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "DPED/models_orig/" + phone)

        for photo in test_photos:

            # load training image and crop it if necessary

          # Load and process the image
          original_image = Image.open(test_dir + photo)
          resized_image = original_image.resize([res_sizes[phone][1], res_sizes[phone][0]])
          image = np.array(resized_image, dtype=np.float32) / 255  # Ensure float32 and normalize
  
          image_crop = utils.extract_crop(image, resolution, phone, res_sizes)
          image_crop_2d = np.reshape(image_crop, [1, IMAGE_SIZE])
  
          # Get enhanced image
          enhanced_2d = sess.run(enhanced, feed_dict={x_: image_crop_2d})
          enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
  
          # Ensure the pixel values are in the range [0, 1]
          enhanced_image = np.clip(enhanced_image, 0, 1)
  
          # Concatenate images for side-by-side comparison
          before_after = np.hstack((image_crop, enhanced_image))
  
          # Convert to 8-bit format for saving
          before_after_8bit = (before_after * 255).astype(np.uint8)
          enhanced_image_8bit = (enhanced_image * 255).astype(np.uint8)
  
          # Save the results as .png images
          photo_name = photo.rsplit(".", 1)[0]
          Image.fromarray(before_after_8bit, 'RGB').save("DPED/visual_results/" + phone + "_" + photo_name + "_before_after.png")
          Image.fromarray(enhanced_image_8bit, 'RGB').save("DPED/visual_results/" + phone + "_" + photo_name + "_enhanced.png")
    else:
        num_saved_models = int(len([f for f in os.listdir("DPED/models/") if f.startswith(phone + "_iteration")]) / 2)
    
        if iteration == "all":
            iteration = np.arange(1, num_saved_models) * 1000
        else:
            iteration = [int(iteration)]
    
        for i in iteration:
            # Load pre-trained model
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, "DPED/models/" + phone + "_iteration_" + str(i) + ".ckpt")
    
            for photo in test_photos:
                # Load and process the image
                original_image = Image.open(test_dir + photo)
                resized_image = original_image.resize([res_sizes[phone][1], res_sizes[phone][0]])
                image = np.array(resized_image, dtype=np.float32) / 255  # Ensure float32 and normalize
    
                image_crop = utils.extract_crop(image, resolution, phone, res_sizes)
                image_crop_2d = np.reshape(image_crop, [1, IMAGE_SIZE])
    
                # Get enhanced image
                enhanced_2d = sess.run(enhanced, feed_dict={x_: image_crop_2d})
                enhanced_image = np.reshape(enhanced_2d, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    
                # Ensure the pixel values are in the range [0, 1]
                enhanced_image = np.clip(enhanced_image, 0, 1)
    
                # Concatenate images for side-by-side comparison
                before_after = np.hstack((image_crop, enhanced_image))
    
                # Convert to 8-bit format for saving
                before_after_8bit = (before_after * 255).astype(np.uint8)
                enhanced_image_8bit = (enhanced_image * 255).astype(np.uint8)
    
                # Save the results as .png images
                photo_name = photo.rsplit(".", 1)[0]
                Image.fromarray(before_after_8bit, 'RGB').save("DPED/visual_results2/" + phone + "_" + photo_name + "_iteration_" + str(i) + "_before_after.png")
                Image.fromarray(enhanced_image_8bit, 'RGB').save("DPED/visual_results2/" + phone + "_" + photo_name + "_iteration_" + str(i) + "_enhanced.png")
