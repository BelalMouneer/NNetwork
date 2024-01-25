import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define the custom loss function 
def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred)))

# Register the custom loss function
tf.keras.utils.get_custom_objects()['rmse'] = rmse

# Load the trained autoencoder model with custom_objects parameter
loaded_ae = tf.keras.models.load_model("autoencoder_model_updated.h5", custom_objects={'rmse': rmse})

# Function to compress an image
def compress_image(image):
    # Reshape and normalize the input image
    image = np.reshape(image, (1, 784)).astype("float32") / 255.0
    
    # Use the encoder part of the autoencoder to get the compressed representation
    compressed_representation = loaded_ae.layers[1].predict(image)
    
    return compressed_representation

# Function to decompress a compressed representation and apply enhancements
def decompress_and_enhance(compressed_representation):
    # Use the decoder part of the autoencoder to reconstruct the image
    reconstructed_image = loaded_ae.layers[2].predict(compressed_representation)
    
    # Reshape the image to its original shape
    reconstructed_image = np.reshape(reconstructed_image, (28, 28))

    # Apply Gaussian smoothing to the reconstructed image
    smoothed_image = cv2.GaussianBlur(reconstructed_image, (5, 5), 0)

    # Increase contrast, saturation, and intensity
    enhanced_image = cv2.convertScaleAbs(smoothed_image, alpha=1.5, beta=20)

    return enhanced_image



# For simplicity, let's use some images from the MNIST dataset again
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Choose a random image for testing
test_image = x_test[np.random.randint(0, x_test.shape[0])]

# Display the original image
plt.subplot(1, 3, 1)
plt.imshow(test_image, cmap="gray")
plt.title("Original Image")

# Compress and decompress the image using the loaded autoencoder
compressed_representation = compress_image(test_image)

# Display the reconstructed image
plt.subplot(1, 3, 2)
reconstructed_image = decompress_and_enhance(compressed_representation)
plt.imshow(reconstructed_image, cmap="gray")
plt.title("Reconstructed and Enhanced Image")

# Apply thresholding to the reconstructed image
thresholded_image = decompress_and_enhance(compressed_representation)

# Display the reconstructed image with thresholding
plt.subplot(1, 3, 3)
plt.imshow(thresholded_image, cmap="gray")
plt.title("Enhanced Image with Thresholding")

# Show the plots
plt.show()
