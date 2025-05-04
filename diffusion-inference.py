import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from stablepy import Model_Diffusers 
import torch
import os

def run_diffusion():
    """
    Runs the Stable Diffusion process with the parameters from the GUI.
    Handles errors and updates the GUI accordingly.
    """
    model_name = model_path.get()
    prompt = prompt_entry.get()
    negative_prompt = negative_prompt_entry.get()
    image_path = image_entry.get()
    img_width = int(width_entry.get())
    img_height = int(height_entry.get())
    num_images = int(num_images_entry.get())
    num_steps = int(num_steps_entry.get())
    guidance_scale = float(guidance_scale_entry.get())
    sampler = sampler_var.get()
    upscaler_model_path = upscaler_model_path_entry.get()
    upscaler_increases_size = float(upscaler_increases_size_entry.get())
    hires_steps = int(hires_steps_entry.get())
    lora_A = lora_A_entry.get()
    lora_scale_A = float(lora_scale_A_entry.get())
    ip_adapter_image_path = ip_adapter_image_entry.get()
    ip_adapter_mask_path = ip_adapter_mask_entry.get()
    ip_adapter_model_name = ip_adapter_model_var.get() 
    ip_adapter_scale = float(ip_adapter_scale_entry.get())
    ip_adapter_mode = ip_adapter_mode_var.get()
    image_storage_location = image_storage_location_entry.get()
    task = task_var.get()
    use_ip_adapter = ip_adapter_enabled.get() 

    # Basic input validation
    if not os.path.exists(model_name):
        update_status("Error: Model path is invalid.")
        return
    if not prompt:
        update_status("Error: Prompt cannot be empty.")
        return
    
    
    if use_ip_adapter:
        if ip_adapter_image_path and not os.path.exists(ip_adapter_image_path):
            update_status("Error: IP Adapter Image path is invalid.")
            return
        

    update_status("Running Stable Diffusion...")
    try:
        model = Model_Diffusers(
            base_model_id=model_name,
            task_name=task,
        )
        
        ip_adapter_image_list = [ip_adapter_image_path] if use_ip_adapter and ip_adapter_image_path else [] # Condition on use_ip_adapter
        ip_adapter_mask_list = [ip_adapter_mask_path] if use_ip_adapter and ip_adapter_mask_path else [] # Condition on use_ip_adapter
        ip_adapter_model_list = [ip_adapter_model_name] if use_ip_adapter and ip_adapter_model_name else [] # Condition on use_ip_adapter # Use the changed variable

        images, image_list = model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_path,
            img_width=img_width,
            img_height=img_height,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sampler=sampler,
            upscaler_model_path=upscaler_model_path,
            upscaler_increases_size=upscaler_increases_size,
            hires_steps=hires_steps,
            lora_A=lora_A,
            lora_scale_A=lora_scale_A,
            ip_adapter_image=ip_adapter_image_list,
            ip_adapter_mask=ip_adapter_mask_list,
            ip_adapter_model=ip_adapter_model_list, 
            ip_adapter_scale=[ip_adapter_scale],
            ip_adapter_mode=[ip_adapter_mode],
            image_storage_location=image_storage_location,
        )
        if images:
            update_status("Image(s) generated successfully!")
            img = images[0]
            img.save("generated_image.png")
            display_image("generated_image.png")
        else:
            update_status("Error: No images were generated.")

    except Exception as e:
        update_status(f"Error: {e}")

def browse_model():
    """Opens a file dialog to select the Stable Diffusion model."""
    file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.safetensors")])
    model_path.set(file_path)

def browse_image():
    """Opens a file dialog to select the input image."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    image_entry.set(file_path)

def browse_upscaler_model():
    file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pth")])
    upscaler_model_path_entry.delete(0, tk.END)  # Clear previous content
    upscaler_model_path_entry.insert(0, file_path) # Insert the path

def browse_lora_A():
    file_path = filedialog.askopenfilename(filetypes=[("LORA files", "*.safetensors")])
    lora_A_entry.delete(0, tk.END) 
    lora_A_entry.insert(0, file_path)
    
def browse_ip_adapter_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    ip_adapter_image_entry.delete(0, tk.END) 
    ip_adapter_image_entry.insert(0, file_path)

def browse_ip_adapter_mask():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    ip_adapter_mask_entry.delete(0, tk.END)
    ip_adapter_mask_entry.insert(0, file_path)

def browse_image_storage_location():
    folder_path = filedialog.askdirectory()
    image_storage_location_entry.delete(0, tk.END) 
    image_storage_location_entry.insert(0, folder_path)
    
def update_status(text):
    """Updates the status label with the given text."""
    status_label.config(text=text)
    window.update_idletasks()

def display_image(image_path):
    """Displays the image in the GUI."""
    try:
        img = Image.open(image_path)
        img.thumbnail((1080, 1080))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
    except Exception as e:
        update_status(f"Error displaying image: {e}")
        image_label.config(image=None)
        image_label.image = None

# Create the main window
window = tk.Tk()
window.title("Stable Diffusion GUI")


# Global variables for storing file paths
model_path = tk.StringVar()
image_entry = tk.StringVar()
upscaler_model_path_entry = tk.StringVar()
lora_A_entry = tk.StringVar()
ip_adapter_image_entry = tk.StringVar()
ip_adapter_mask_entry = tk.StringVar()
image_storage_location_entry = tk.StringVar()
ip_adapter_enabled = tk.BooleanVar() 

# --- Main Frames ---
input_frame = ttk.Frame(window)
input_frame.pack(side="left", padx=10, pady=10, fill="y")

image_display_frame = ttk.Frame(window)
image_display_frame.pack(side="right", padx=10, pady=10, fill="both", expand=True)

# --- Input Frame Content ---
# Prompt Frame
prompt_frame = ttk.Frame(input_frame)
prompt_frame.pack(pady=10, fill="x")

# Prompt
ttk.Label(prompt_frame, text="Prompt:").grid(row=0, column=0, padx=5, sticky="w")
prompt_entry = ttk.Entry(prompt_frame, width=40)
prompt_entry.grid(row=0, column=1, padx=5, sticky="ew")

# Negative Prompt
ttk.Label(prompt_frame, text="Negative Prompt:").grid(row=1, column=0, padx=5, sticky="w")
negative_prompt_entry = ttk.Entry(prompt_frame, width=40)
negative_prompt_entry.grid(row=1, column=1, padx=5, sticky="ew")

# Model and Image Selection
model_image_frame = ttk.Frame(input_frame)
model_image_frame.pack(pady=10, fill="x")
ttk.Label(model_image_frame, text="Model Path:").grid(row=0, column=0, padx=5, sticky="w")
ttk.Entry(model_image_frame, textvariable=model_path, width=40).grid(row=0, column=1, padx=5, sticky="ew")
ttk.Button(model_image_frame, text="Browse", command=browse_model).grid(row=0, column=2, padx=5, sticky="w")

ttk.Label(model_image_frame, text="Input Image:").grid(row=1, column=0, padx=5, sticky="w")
ttk.Entry(model_image_frame, textvariable=image_entry, width=40).grid(row=1, column=1, padx=5, sticky="ew")
ttk.Button(model_image_frame, text="Browse", command=browse_image).grid(row=1, column=2, padx=5, sticky="w")

# Configuration Frame
config_frame = ttk.Frame(input_frame)
config_frame.pack(pady=10, fill="x")

# Task Selection
ttk.Label(config_frame, text="Task:").grid(row=0, column=0, padx=5, sticky="w")
tasks = ["txt2img", "img2img","pattern","ip2p"]
task_var = tk.StringVar()
task_var.set(tasks[0])
task_dropdown = ttk.Combobox(config_frame, textvariable=task_var, values=tasks, width=10)
task_dropdown.grid(row=0, column=1, padx=5, sticky="ew")

# Image Dimensions
ttk.Label(config_frame, text="Image Width:").grid(row=1, column=0, padx=5, sticky="w")
width_entry = ttk.Entry(config_frame, width=10)
width_entry.grid(row=1, column=1, padx=5, sticky="ew")
width_entry.insert(0, "512")

ttk.Label(config_frame, text="Image Height:").grid(row=1, column=2, padx=5, sticky="w")
height_entry = ttk.Entry(config_frame, width=10)
height_entry.grid(row=1, column=3, padx=5, sticky="ew")
height_entry.insert(0, "512")

# Number of Images
ttk.Label(config_frame, text="Number of Images:").grid(row=2, column=0, padx=5, sticky="w")
num_images_entry = ttk.Entry(config_frame, width=10)
num_images_entry.grid(row=2, column=1, padx=5, sticky="ew")
num_images_entry.insert(0, "1")

# Number of Steps
ttk.Label(config_frame, text="Number of Steps:").grid(row=2, column=2, padx=5, sticky="w")
num_steps_entry = ttk.Entry(config_frame, width=10)
num_steps_entry.grid(row=2, column=3, padx=5, sticky="ew")
num_steps_entry.insert(0, "10")

# Guidance Scale
ttk.Label(config_frame, text="Guidance Scale:").grid(row=3, column=0, padx=5, sticky="w")
guidance_scale_entry = ttk.Entry(config_frame, width=10)
guidance_scale_entry.grid(row=3, column=1, padx=5, sticky="ew")
guidance_scale_entry.insert(0, "8.0")

# Sampler
ttk.Label(config_frame, text="Sampler:").grid(row=3, column=2, padx=5, sticky="w")
samplers = ["LCM Auto-Loader","TCD Auto-Loader","Euler a","DPM++ 2M SDE","DPM++ 2M SDE Ef"]
sampler_var = tk.StringVar()
sampler_var.set(samplers[0])
sampler_dropdown = ttk.Combobox(config_frame, textvariable=sampler_var, values=samplers, width=10)
sampler_dropdown.grid(row=3, column=3, padx=5, sticky="ew")

# Upscaler Model Path
ttk.Label(config_frame, text="Upscaler Model:").grid(row=4, column=0, padx=5, sticky="w")
upscaler_model_path_entry = ttk.Entry(config_frame, width=40)
upscaler_model_path_entry.grid(row=4, column=1, padx=5, sticky="ew")
ttk.Button(config_frame, text="Browse", command=browse_upscaler_model).grid(row=4, column=2, padx=5, sticky="w")

# Upscaler Increase Size
ttk.Label(config_frame, text="Upscaler Size Increase:").grid(row=5, column=0, padx=5, sticky="w")
upscaler_increases_size_entry = ttk.Entry(config_frame, width=10)
upscaler_increases_size_entry.grid(row=5, column=1, padx=5, sticky="ew")
upscaler_increases_size_entry.insert(0, "1.5")

# Hires Steps
ttk.Label(config_frame, text="Hires Steps:").grid(row=5, column=2, padx=5, sticky="w")
hires_steps_entry = ttk.Entry(config_frame, width=10)
hires_steps_entry.grid(row=5, column=3, padx=5, sticky="ew")
hires_steps_entry.insert(0, "0")

# Lora A Path
ttk.Label(config_frame, text="Lora A Path:").grid(row=6, column=0, padx=5, sticky="w")
lora_A_entry = ttk.Entry(config_frame, width=40)
lora_A_entry.grid(row=6, column=1, padx=5, sticky="ew")
ttk.Button(config_frame, text="Browse", command=browse_lora_A).grid(row=6, column=2, padx=5, sticky="w")

# Lora Scale A
ttk.Label(config_frame, text="Lora Scale A:").grid(row=7, column=0, padx=5, sticky="w")
lora_scale_A_entry = ttk.Entry(config_frame, width=10)
lora_scale_A_entry.grid(row=7, column=1, padx=5, sticky="ew")
lora_scale_A_entry.insert(0, "0.8")

# IP Adapter Checkbox
ip_adapter_enabled = tk.BooleanVar()
ip_adapter_checkbox = ttk.Checkbutton(config_frame, text="Enable IP Adapter", variable=ip_adapter_enabled)
ip_adapter_checkbox.grid(row=7, column=2, padx=5, sticky="w")

# IP Adapter Image
ttk.Label(config_frame, text="IP Adapter Image:").grid(row=8, column=0, padx=5, sticky="w")
ip_adapter_image_entry = ttk.Entry(config_frame, width=40)
ip_adapter_image_entry.grid(row=8, column=1, padx=5, sticky="ew")
ttk.Button(config_frame, text="Browse", command=browse_ip_adapter_image).grid(row=8, column=2, padx=5, sticky="w")

# IP Adapter Mask
ttk.Label(config_frame, text="IP Adapter Mask:").grid(row=9, column=0, padx=5, sticky="w")
ip_adapter_mask_entry = ttk.Entry(config_frame, width=40)
ip_adapter_mask_entry.grid(row=9, column=1, padx=5, sticky="ew")
ttk.Button(config_frame, text="Browse", command=browse_ip_adapter_mask).grid(row=9, column=2, padx=5, sticky="w")

# IP Adapter Model
ttk.Label(config_frame, text="IP-Adapter Model:").grid(row=7, column=3, padx=5, sticky="w") # Changed row
ip_adapter_models = ["faceid", "plus_face","base"]
ip_adapter_model_var = tk.StringVar()
ip_adapter_model_var.set(ip_adapter_models[0])
ip_adapter_model_dropdown = ttk.Combobox(config_frame, textvariable=ip_adapter_model_var, values=ip_adapter_models, width=15)
ip_adapter_model_dropdown.grid(row=8, column=3, padx=5, sticky="ew") # Changed row

# IP Adapter Scale
ttk.Label(config_frame, text="IP-Adapter Scale:").grid(row=9, column=3, padx=5, sticky="w")
ip_adapter_scale_entry = ttk.Entry(config_frame, width=10)
ip_adapter_scale_entry.grid(row=9, column=4, padx=5, sticky="ew")
ip_adapter_scale_entry.insert(0, "0.9")

# IP Adapter Mode
ttk.Label(config_frame, text="IP-Adapter Mode:").grid(row=10, column=3, padx=5, sticky="w")
ip_adapter_modes = ["original", "style","layout"]
ip_adapter_mode_var = tk.StringVar()
ip_adapter_mode_var.set(ip_adapter_modes[0])
ip_adapter_mode_dropdown = ttk.Combobox(config_frame, textvariable=ip_adapter_mode_var, values=ip_adapter_modes, width=15)
ip_adapter_mode_dropdown.grid(row=10, column=4, padx=5, sticky="ew")

# Image Storage Location
ttk.Label(config_frame, text="Image Storage Location:").grid(row=11, column=0, padx=5, sticky="w") 
image_storage_location_entry = ttk.Entry(config_frame, width=40)
image_storage_location_entry.grid(row=11, column=1, padx=5, sticky="ew")
ttk.Button(config_frame, text="Browse", command=browse_image_storage_location).grid(row=11, column=2, padx=5, sticky="w")
image_storage_location_entry.insert(0, "D:/images")

# Run Button and Status
run_button = ttk.Button(input_frame, text="Run Diffusion", command=run_diffusion)
run_button.pack(pady=10, fill="x")

status_label = ttk.Label(input_frame, text="Ready")
status_label.pack(pady=5, fill="x")

# --- Image Display Frame Content ---
image_label = ttk.Label(image_display_frame)
image_label.pack(fill="both", expand=True)

# Make the input frame more resizable
input_frame.grid_rowconfigure(0, weight=1)
input_frame.grid_columnconfigure(0, weight=1)

# Run the GUI
window.mainloop()
