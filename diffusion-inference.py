import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from stablepy import Model_Diffusers
from stablepy import scheduler_names
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import threading  # Add this line
# 
SETTINGS_FILE = "image_generator_settings.json"

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)

def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def generate_image():
    prompt = prompt_entry.get()
    negative_prompt = negative_prompt_entry.get()
    model_path = model_path_entry.get()
    upscaler_path = upscaler_path_entry.get()
    output_dir = output_dir_entry.get()
    task = task_combobox.get()
    image_path = image_path_entry.get()
    mask_path = mask_path_entry.get()
    img_width = int(width_entry.get())
    img_height = int(height_entry.get())
    num_steps = int(steps_entry.get())
    guidance_scale = float(guidance_scale_entry.get())
    upscaler_increases_size = float(size_entry.get())
    hires_steps = int(hi_steps_entry.get())
    lora_A = lora_A_entry.get() # Added for LoRa Adapters  
    lora_scale_A = float(lora_scale_A_entry.get()) # LoRA scale 
	
    settings = {
          "prompt": prompt,
          "negative_prompt": negative_prompt,
          "model_path": model_path,
          "upscaler_path": upscaler_path,
          "output_dir": output_dir,
          "task": task,
          "image_path": image_path,
          "mask_path": mask_path,
          "img_width": img_width, 
          "img_height": img_height,
          "num_steps": num_steps,
          "guidance_scale": guidance_scale,
          "upscaler_increases_size": upscaler_increases_size,
          "hires_steps": hires_steps,
	  "lora_A": lora_A,
          "lora_scale_A": lora_scale_A, 
        }
    save_settings(settings)

    try:
        progress_bar['value'] = 0
        progress_bar.update()

        # Simulate file fetching (20%)
        progress_bar['value'] = 20
        progress_bar.update()

        model = Model_Diffusers(
            base_model_id=model_path,
            #type_model_precision='torch.float16',
            task_name=task,
        )

        progress_bar['value'] = 40
        progress_bar.update()

        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "img_width": img_width,
            "img_height": img_height,
            "num_images": 1,
            "sampler": "DPM++ 2M",
            "upscaler_model_path": upscaler_path,
            "upscaler_increases_size": upscaler_increases_size,
            "hires_steps": hires_steps,
            "image_storage_location": output_dir,
	    "lora_A": lora_A,
            "lora_scale_A": lora_scale_A,
        }

        if task in ['img2img', 'ip2p'] and image_path:
            kwargs["image"] = image_path
        if task == 'ip2p' and mask_path:
            kwargs["image_mask"] = mask_path

        image, info_image = model(**kwargs)
        # Simulate inference steps completed (40%)
        progress_bar['value'] = 80
        progress_bar.update()


        if image:
            if isinstance(image, list):
                image = image[0]
            image_path_save = f"{output_dir}/generated_image.png"
            image.save(image_path_save)
            messagebox.showinfo("Success", f"Image saved to {image_path_save}")

            display_image(image)

        else:
            messagebox.showerror("Error", "Image generation failed.")

        # Simulate upscaling completed (20%)
        progress_bar['value'] = 100
        progress_bar.update()

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def generate_image_thread(): # Move the function definition here
    threading.Thread(target=generate_image).start()

def update_progress(progress):
    pass

def browse_model():
    filename = filedialog.askopenfilename(title="Select Model File")
    model_path_entry.delete(0, tk.END)
    model_path_entry.insert(0, filename)

def browse_lora_A():
    filename = filedialog.askopenfilename(title="Select LoRa Model File")
    lora_A_entry.delete(0, tk.END)
    lora_A_entry.insert(0, filename)

def browse_upscaler():
    filename = filedialog.askopenfilename(title="Select Upscaler File")
    upscaler_path_entry.delete(0, tk.END)
    upscaler_path_entry.insert(0, filename)

def browse_output_dir():
    dirname = filedialog.askdirectory(title="Select Output Directory")
    output_dir_entry.delete(0, tk.END)
    output_dir_entry.insert(0, dirname)

def browse_image():
    filename = filedialog.askopenfilename(title="Select Image File")
    image_path_entry.delete(0, tk.END)
    image_path_entry.insert(0, filename)

def browse_mask():
    filename = filedialog.askopenfilename(title="Select Mask Image File")
    mask_path_entry.delete(0, tk.END)
    mask_path_entry.insert(0, filename)

def display_image(image):
    global original_image, displayed_image
    original_image = image
    displayed_image = ImageTk.PhotoImage(original_image)
    image_label.config(image=displayed_image)
    image_label.image = displayed_image
    canvas.config(scrollregion=canvas.bbox("all"))

root = tk.Tk()
root.title("StablePy Image Generator")

settings = load_settings()

input_frame = tk.Frame(root)
input_frame.grid(row=0, column=0, sticky="nsew")

# Image Frame
image_frame = tk.Frame(root)
image_frame.grid(row=0, column=1, sticky="nsew")

# ... (your input fields and labels) ...
tk.Label(input_frame, text="Prompt:").grid(row=0, column=0, sticky="w")
prompt_entry = tk.Entry(input_frame, width=50)
prompt_entry.grid(row=0, column=1, columnspan=2)
prompt_entry.insert(0, settings.get("prompt", ""))




tk.Label(input_frame, text="Negative Prompt:").grid(row=1, column=0, sticky="w")
negative_prompt_entry = tk.Entry(input_frame, width=50)
negative_prompt_entry.grid(row=1, column=1, columnspan=2)
negative_prompt_entry.insert(0, settings.get("negative_prompt", ""))

tk.Label(input_frame, text="Model Path:").grid(row=2, column=0, sticky="w")
model_path_entry = tk.Entry(input_frame, width=40)
model_path_entry.grid(row=2, column=1)
model_browse_button = tk.Button(input_frame, text="Browse", command=browse_model)
model_browse_button.grid(row=2, column=2)
model_path_entry.insert(0, settings.get("model_path", ""))

tk.Label(input_frame, text="Upscaler Path:").grid(row=3, column=0, sticky="w")
upscaler_path_entry = tk.Entry(input_frame, width=40)
upscaler_path_entry.grid(row=3, column=1)
upscaler_browse_button = tk.Button(input_frame, text="Browse", command=browse_upscaler)
upscaler_browse_button.grid(row=3, column=2)
upscaler_path_entry.insert(0, settings.get("upscaler_path", ""))

tk.Label(input_frame, text="Output Directory:").grid(row=4, column=0, sticky="w")
output_dir_entry = tk.Entry(input_frame, width=40)
output_dir_entry.grid(row=4, column=1)
output_dir_browse_button = tk.Button(input_frame, text="Browse", command=browse_output_dir)
output_dir_browse_button.grid(row=4, column=2)
output_dir_entry.insert(0, settings.get("output_dir", ""))

tk.Label(input_frame, text="Task:").grid(row=5, column=0, sticky="w")
task_combobox = ttk.Combobox(input_frame, values=['txt2img', 'img2img', 'ip2p'])
task_combobox.grid(row=5, column=1, columnspan=2, sticky="ew")
task_combobox.set(settings.get("task", "txt2img"))

tk.Label(input_frame, text="Image Path:").grid(row=6, column=0, sticky="w")
image_path_entry = tk.Entry(input_frame, width=40)
image_path_entry.grid(row=6, column=1)
image_browse_button = tk.Button(input_frame, text="Browse", command=browse_image)
image_browse_button.grid(row=6, column=2)
image_path_entry.insert(0, settings.get("image_path", ""))

tk.Label(input_frame, text="Mask Path:").grid(row=7, column=0, sticky="w")
mask_path_entry = tk.Entry(input_frame, width=40)
mask_path_entry.grid(row=7, column=1)
mask_browse_button = tk.Button(input_frame, text="Browse", command=browse_mask)
mask_browse_button.grid(row=7, column=2)
mask_path_entry.insert(0, settings.get("mask_path", ""))

tk.Label(input_frame, text="Width:").grid(row=8, column=0)
width_entry = tk.Entry(input_frame, width=10)
width_entry.grid(row=8, column=1)
width_entry.insert(0, settings.get("img_width", "336"))

tk.Label(input_frame, text="Height:").grid(row=9, column=0)
height_entry = tk.Entry(input_frame, width=10)
height_entry.grid(row=9, column=1)
height_entry.insert(0, settings.get("img_height", "512"))

tk.Label(input_frame, text="Steps:").grid(row=10, column=0)
steps_entry = tk.Entry(input_frame, width=10)
steps_entry.grid(row=10, column=1)
steps_entry.insert(0, settings.get("num_steps", "50"))

tk.Label(input_frame, text="Guidance Scale:").grid(row=13, column=0)
guidance_scale_entry = tk.Entry(input_frame, width=10)
guidance_scale_entry.grid(row=13, column=1)
guidance_scale_entry.insert(0, settings.get("guidance_scale", "7")) # example default value

tk.Label(input_frame, text="Upscaler Size:").grid(row=14, column=0)
size_entry = tk.Entry(input_frame, width=10)
size_entry.grid(row=14, column=1)
size_entry.insert(0, settings.get("upscaler_increases_size", "1.5")) # example default value

tk.Label(input_frame, text="Hires Steps:").grid(row=15, column=0)
hi_steps_entry = tk.Entry(input_frame, width=10)
hi_steps_entry.grid(row=15, column=1)
hi_steps_entry.insert(0, settings.get("hires_steps", "50"))

tk.Label(input_frame, text="LoRA Model:").grid(row=16, column=0, sticky="w")
lora_A_entry = tk.Entry(input_frame, width=40)
lora_A_entry.grid(row=16, column=1)
lora_A_browse_button = tk.Button(input_frame, text="Browse", command=browse_lora_A)
lora_A_browse_button.grid(row=16, column=2)
lora_A_entry.insert(0, settings.get("lora_A", ""))

tk.Label(input_frame, text="LoRA Scale:").grid(row=17, column=0)
lora_scale_A_entry = tk.Entry(input_frame, width=10)
lora_scale_A_entry.grid(row=17, column=1)
lora_scale_A_entry.insert(0, settings.get("lora_scale_A", "0.8"))

generate_button = tk.Button(input_frame, text="Generate Image", command=generate_image_thread)
generate_button.grid(row=16, column=1, pady=10)

progress_bar = ttk.Progressbar(input_frame, orient='horizontal', mode='determinate', length=280)
progress_bar.grid(row=18, column=0, columnspan=3, pady=10)

canvas = tk.Canvas(image_frame, scrollregion=(0, 0, 0, 0))
canvas.pack(fill="both", expand=True)

scrollbar = tk.Scrollbar(image_frame, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")

canvas.config(yscrollcommand=scrollbar.set)
root.configure(bg='black')
image_label = tk.Label(canvas)
canvas.create_window((0, 0), window=image_label, anchor='nw')

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

original_image = None
displayed_image = None

root.mainloop()
