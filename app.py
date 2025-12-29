import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import torch
from torchvision import transforms
from medical_resnet import MedicalCNN

# ===============================
# Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Load Model
# ===============================
model = MedicalCNN(num_classes=2)
state_dict = torch.load("best_model.pth", map_location=device)
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
# ===============================
# Transforms
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

classes = ["NORMAL", "PNEUMONIA"]

# ===============================
# Tkinter Window
# ===============================
win = tk.Tk()
win.title("Chest X-ray Classification")
win.geometry("1900x1100")
win.configure(bg="#0e0e12")

selected_image_path = None

# ===============================
# Title
# ===============================
title = tk.Label(
    win,
    text="Chest X-ray Classification",
    font=("Segoe UI", 22, "bold"),
    fg="#ffffff",
    bg="#0e0e12"
)
title.pack(pady=20)

# ===============================
# Canvas
# ===============================
canvas = tk.Canvas(
    win,
    width=400,
    height=350,
    bg="#1a1a1e",
)
canvas.pack(pady=15)

# ===============================
# Validate Image
# ===============================
def validate_image(image):
    img_np = np.array(image)

    # Grayscale check
    if len(img_np.shape) == 2:
        return True, ""

    # Saturation check
    img_hsv = image.convert("HSV")
    s = np.array(img_hsv)[:, :, 1]

    if np.mean(s) > 30:
        return False, "Image is not a valid chest X-ray"

    return True, ""

# ===============================
# Choose Image
# ===============================
def choose_image():
    global selected_image_path

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    img = Image.open(file_path).convert("RGB")
    valid, msg = validate_image(img)

    if not valid:
        result_label.config(text=msg, fg="#ef4444")
        canvas.delete("all")
        selected_image_path = None
        return

    selected_image_path = file_path

    img_resized = img.resize((400, 350))
    img_tk = ImageTk.PhotoImage(img_resized)

    canvas.delete("all")
    canvas.image = img_tk
    canvas.create_image(0, 0, anchor="nw", image=img_tk)

    result_label.config(text="", fg="white")

# ===============================
# Analyze Image
# ===============================
def analyze_image():
    if not selected_image_path:
        result_label.config(
            text="Please choose an image first",
            fg="#ef4444"
        )
        return

    image = Image.open(selected_image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    cls = classes[pred.item()]
    conf = confidence.item() * 100

    if cls == "NORMAL":
        color = "#22c55e"
        text = "Normal"
    else:
        color = "#ef4444"
        text = "Pneumonia"

    result_label.config(
        text=f"Prediction: {text}\nConfidence: {conf:.2f}%",
        fg=color
    )

# ===============================
# Buttons
# ===============================
choose_btn = tk.Button(
    win,
    text="Choose Chest X-ray",
    font=("Segoe UI", 14, "bold"),
    command=choose_image,
    width=25
)
choose_btn.pack(pady=10)

analyze_btn = tk.Button(
    win,
    text="Analyze Image",
    font=("Segoe UI", 14, "bold"),
    command=analyze_image,
    width=25
)
analyze_btn.pack(pady=5)

# ===============================
# Result Label
# ===============================
result_label = tk.Label(
    win,
    text="",
    font=("Segoe UI", 18, "bold"),
    bg="#1e1e24",
    fg="white",
    # width=40,
    # height=3
)
result_label.pack(pady=20)

# ===============================
# Run
# ===============================
win.mainloop()
