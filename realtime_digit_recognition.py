import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, simpledialog

# Load the trained model
model = tf.keras.models.load_model("models/handwriting_model.h5")
print("Model loaded successfully!")

def preprocess_image(image_path):
    """Load, preprocess, and prepare the image for model prediction."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    image = cv2.resize(image, (28, 28))  # Resize to match model input
    image = image / 255.0  # Normalize pixel values (0-1)
    image = 1 - image  # Invert colors (model expects white digit on black)
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    return image

def upload_and_predict():
    """Open file dialog, preprocess image, and predict digit."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

    if not file_path:
        print("No file selected.")
        return

    # Preprocess image
    processed_image = preprocess_image(file_path)

    # Make prediction
    prediction = model.predict(processed_image)
    digit = np.argmax(prediction)

    # Display the uploaded image
    image = cv2.imread(file_path)
    cv2.putText(image, f"Predicted: {digit}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv2.imshow("Uploaded Image", image)
    cv2.waitKey(5000)  # Show for 5 seconds
    cv2.destroyAllWindows()

    print(f"Predicted Digit: {digit}")

def draw_on_pad():
    """Create a writing pad where the user can draw, clear, and predict a digit."""
    canvas_size = 400  # Size of the drawing area
    image = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255  # White background
    drawing = False

    def draw(event, x, y, flags, param):
        """Mouse callback function to draw on the canvas."""
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(image, (x, y), 10, (0), -1)  # Draw black circles
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    # Create drawing window
    cv2.namedWindow("Draw a Digit")
    cv2.setMouseCallback("Draw a Digit", draw)

    while True:
        cv2.imshow("Draw a Digit", image)
        key = cv2.waitKey(1)

        if key == ord("q"):  # Press 'q' to quit
            break
        elif key == ord("c"):  # Press 'c' to clear the canvas
            image[:] = 255
            print("Canvas cleared. Draw again!")
        elif key == ord("p"):  # Press 'p' to predict
            # Resize and preprocess the drawn digit
            resized_image = cv2.resize(image, (28, 28))
            processed_image = resized_image / 255.0  # Normalize
            processed_image = 1 - processed_image  # Invert colors
            processed_image = processed_image.reshape(1, 28, 28, 1)  # Reshape for model

            # Predict the digit
            prediction = model.predict(processed_image)
            digit = np.argmax(prediction)

            print(f"Predicted Digit: {digit}")

            # Display the drawn image with the prediction
            temp_img = image.copy()
            cv2.putText(temp_img, f"Predicted: {digit}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            cv2.imshow("Predicted Digit", temp_img)
            cv2.waitKey(2000)  # Show prediction for 2 seconds

    cv2.destroyAllWindows()

def main():
    """Main menu to choose between image upload or drawing pad."""
    root = tk.Tk()
    root.withdraw()  # Hide Tkinter root window
    choice = simpledialog.askinteger("Digit Recognition", 
                                     "Choose an option:\n1. Upload Image\n2. Draw on Writing Pad\nEnter choice (1/2):")
    if choice == 1:
        upload_and_predict()
    elif choice == 2:
        draw_on_pad()
    else:
        print("Invalid choice. Please run the program again.")

# Run the main menu
main()
