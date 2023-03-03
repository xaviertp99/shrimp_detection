from tkinter import Tk
from PIL import Image, ImageTk

# Load the image using PIL
logo = Image.open("C:\\Users\\User\\OneDrive\\Imágenes\\LOGO.jpg")

# Create a PhotoImage object using the ImageTk module
logo_photo = ImageTk.PhotoImage(logo)

# Create the Tk object
root = Tk()

# Set the icon using the PhotoImage object
root.iconphoto(True, logo_photo)

# Run the main loop
root.mainloop()