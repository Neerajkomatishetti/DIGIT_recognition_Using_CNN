import matplotlib
matplotlib.use('Agg')

import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D
from PIL import Image

app = Flask(__name__)

# Load your pre-trained digit recognition model (ensure 'mnist_conv_model.h5' is available)
model = load_model('mnist_conv_model.h5')

dummy_input = np.zeros((1, 28, 28, 1))
model.predict(dummy_input)

# Create an activation model to extract outputs from all convolution layers
layer_outputs = [layer.output for layer in model.layers if isinstance(layer, Conv2D)]
if not layer_outputs:
    print("Warning: No convolution layers found in the model. Activation maps will not be generated.")
    activation_model = None
else:
    activation_model = Model(inputs=model.inputs, outputs=layer_outputs)

def preprocess_image(pil_image):
    """
    Preprocess the PIL image for prediction:
      - Convert to grayscale (if not already)
      - Resize to 28x28 pixels
      - Invert colors (if necessary for MNIST)
      - Normalize pixel values to [0, 1]
      - Reshape for model input
    """
    image = pil_image.convert('L')  # Ensure grayscale
    image = image.resize((28, 28))
    img_array = np.array(image)
    # Invert colors if the digit is black-on-white (MNIST expects white-on-black)
    img_array = 255 - img_array
    # Normalize and prepare for prediction (add a batch and channel dimension)
    normalized_array = img_array / 255.0
    processed_array = normalized_array.reshape(1, 28, 28, 1)
    
    # For saving the output image, convert normalized array to uint8
    img_array_uint8 = (normalized_array * 255).astype('uint8')
    # Remove extra dimensions (if any) to get a 2D array
    img_array_uint8 = np.squeeze(img_array_uint8)
    
    processed_pil = Image.fromarray(img_array_uint8)
    processed_pil.save("output_image.png")
    return processed_array

def generate_activation_images(activations):
    """
    Convert each convolution layer's activations into a grid image.
    Returns a list of base64-encoded PNG images.
    """
    images = []
    for layer_activation in activations:
        # Each activation has shape: (1, height, width, n_filters)
        n_filters = layer_activation.shape[-1]
        height = layer_activation.shape[1]
        width = layer_activation.shape[2]

        # Determine grid size (approximately square)
        n_cols = int(np.sqrt(n_filters))
        n_rows = int(np.ceil(n_filters / n_cols))
        display_grid = np.zeros((height * n_rows, width * n_cols))

        for i in range(n_filters):
            filter_activation = layer_activation[0, :, :, i]
            # Normalize the activation for display purposes
            filter_activation -= filter_activation.mean()
            if filter_activation.std() > 0:
                filter_activation /= filter_activation.std()
            filter_activation *= 64
            filter_activation += 128
            filter_activation = np.clip(filter_activation, 0, 255).astype('uint8')

            row = i // n_cols
            col = i % n_cols
            display_grid[row * height:(row + 1) * height,
                         col * width:(col + 1) * width] = filter_activation

        # Create a matplotlib figure and encode the image to base64
        plt.figure(figsize=(n_cols, n_rows))
        plt.title("Activation Map")
        plt.axis('off')
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        images.append(encoded)
        plt.close()
    return images

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('file')
        if not files or all(file.filename == "" for file in files):
            return redirect(request.url)
        
        results = []
        for file in files:
            if file.filename == "":
                continue

            file_stream = io.BytesIO(file.read())
            pil_image = Image.open(file_stream)
            
            preview_buffer = io.BytesIO()
            pil_image.save(preview_buffer, format="PNG")
            preview_buffer.seek(0)
            preview_base64 = base64.b64encode(preview_buffer.getvalue()).decode('utf-8')
            
            # Preprocess the image and save the processed output image
            processed_image = preprocess_image(pil_image)
            prediction = model.predict(processed_image)
            predicted_digit = np.argmax(prediction)
            
            if activation_model:
                activations = activation_model.predict(processed_image)
                activation_images = generate_activation_images(activations)
            else:
                activation_images = []
            
            # Read the processed output image, encode to base64, and add to result
            with open("output_image.png", "rb") as output_file:
                output_bytes = output_file.read()
            output_base64 = base64.b64encode(output_bytes).decode('utf-8')
            
            results.append({
                "filename": file.filename,
                "predicted_digit": predicted_digit,
                "input_image": preview_base64,
                "output_image": output_base64,
                "activation_images": activation_images
            })
        return render_template('result.html', results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
