import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define the custom loss function (replace with your actual implementation)
def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred)))

# Register the custom loss function
tf.keras.utils.get_custom_objects()['rmse'] = rmse

# Load the trained autoencoder model with custom_objects parameter
loaded_ae = tf.keras.models.load_model("autoencoder_model_updated.h5", custom_objects={'rmse': rmse})

# Extract the encoder part of the model
encoder_model = loaded_ae.layers[1]

# Function to compress an image
def compress_image(image):
    # Resize the input image to 28x28 and normalize
    resized_image = cv2.resize(image, (28, 28)).astype("float32") / 255.0
    
    # Reshape the image to match the expected input shape of the encoder model
    reshaped_image = np.reshape(resized_image, (1, 784))
    
    # Use the encoder part of the autoencoder to get the compressed representation
    compressed_representation = loaded_ae.layers[1].predict(reshaped_image)
    
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
    enhanced_image = cv2.convertScaleAbs(smoothed_image, alpha=1.2, beta=20)

    return enhanced_image

# Specify the image path
image_path = r"C:\Users\byaya\Downloads\Example-of-a-MNIST-input-An-image-is-passed-to-the-network-as-a-matrix-of-28-by-28.png"

# Load the specified image
user_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(user_image, cmap="gray")
plt.title("Original Image")

# Compress and decompress the user image using the loaded autoencoder
compressed_representation = compress_image(user_image)

# Display the reconstructed and enhanced image
plt.subplot(1, 2, 2)

reconstructed_image = decompress_and_enhance(compressed_representation)
plt.imshow(reconstructed_image, cmap="gray")
plt.title("Reconstructed and Enhanced Image")

# Show the plots
plt.show()


