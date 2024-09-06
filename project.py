#C:\Users\Suresh kumar N\Downloads\cnn_vgg16_Knee (1).h5
#C:\Users\Suresh kumar N\Downloads\back.jpeg
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Load your pre-trained model
model = load_model(r'C:\Users\Suresh kumar N\Downloads\cnn_vgg16_Knee (1).h5')

# Define the labels
labels = ['glaucoma', 'non glaucoma']

def load_and_predict():
    # Prompt the user to select an image file
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize the image
       
        # Make predictions
        preds = model.predict(x)
        predicted_label = labels[np.argmax(preds)]

        # Display the image and prediction
        display_image(file_path, predicted_label)

def display_image(file_path, prediction):
    # Create a new window for displaying the image and prediction
    window = tk.Toplevel(root)
    window.title("Prediction Result")

    # Load and display the image
    img = Image.open(file_path)
    img = img.resize((300, 300))  # Resize the image for better display
    photo = ImageTk.PhotoImage(img)
    label = tk.Label(window, image=photo)
    label.image = photo
    label.pack()

    # Display the prediction
    prediction_label = tk.Label(window, text="Predicted Label: " + prediction, font=("Helvetica", 16))
    prediction_label.pack(pady=10)

# Create the main window
root = tk.Tk()
root.title("Image Classifier")

# Load background image
background_image = Image.open(r"C:\Users\Suresh kumar N\Downloads\back.jpeg")
background_image = background_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), resample=Image.LANCZOS)
background_photo = ImageTk.PhotoImage(background_image)

# Create a canvas for the background image
canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=background_photo, anchor="nw")
label_title = tk.Label(canvas, text="Glaucoma Detection", font=("Helvetica", 24), fg="#156cae")
label_title.place(relx=0.02, rely=0.02, anchor="nw")
label_instruction = tk.Label(canvas, text="Drop an image to predict", font=("Arial", 14), bg='#a985c7', fg='white', padx=20, pady=10)
label_instruction.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
# Create a button to load and predict
btn = tk.Button(root, text="Upload Image", command=load_and_predict, font=("Arial", 14), bg='#a985c7', fg='white', padx=20, pady=10)
btn.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Start the GUI event loop
root.mainloop()