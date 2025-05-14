import numpy as np
import joblib
from PIL import Image, ImageTk
from rembg import remove
import tkinter as tk
from tkinter import filedialog, Label, Button
from scratch_50 import renk_ozellik, renk_moment_ozellik, boyut_ozellik, haralick_ozellik

class TahminUygulamasi:
    def __init__(self, root):
        self.root = root
        self.root.title("Nohut vs Mercimek Tahmin Uygulaması")

        self.model_path = "knn_modelim.joblib"
        self.scaler_path = "minmax_scaler.joblib"

        # Görsel etiketleri
        self.original_label = Label(root, text="Orijinal Görsel")
        self.original_label.grid(row=0, column=0)

        self.processed_label = Label(root, text="İşlenmiş Görsel")
        self.processed_label.grid(row=0, column=1)

        self.original_image_panel = Label(root)
        self.original_image_panel.grid(row=1, column=0, padx=10, pady=10)

        self.processed_image_panel = Label(root)
        self.processed_image_panel.grid(row=1, column=1, padx=10, pady=10)

        # Buton
        self.select_button = Button(root, text="Görsel Seç ve Tahmin Et", command=self.choose_image)
        self.select_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Tahmin sonucu etiketi
        self.result_label = Label(root, text="", font=("Arial", 14))
        self.result_label.grid(row=3, column=0, columnspan=2, pady=10)

    def choose_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.predict_new_image(file_path)

    def predict_new_image(self, image_path):
        img = Image.open(image_path).convert("RGBA")
        img_no_bg = remove(img)
        img_resized = img_no_bg.resize((500, 500), Image.LANCZOS)
        img_array = np.array(img_resized)

        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        renk = renk_ozellik(img_array)
        renk_moment = renk_moment_ozellik(img_array)
        boyut = boyut_ozellik(img_array)
        doku = haralick_ozellik(img_array)
        features = renk + renk_moment + boyut + doku

        scaler = joblib.load(self.scaler_path)
        X = scaler.transform([features])

        model = joblib.load(self.model_path)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

        tahmin_label = "MERCİMEK" if prediction == 2 else "NOHUT"
        yuzde = round(100 * max(probability), 2)
        self.result_label.config(text=f"Tahmin: {tahmin_label} ({yuzde}%)")

        # Görselleri arayüzde göster
        original_img = Image.open(image_path).resize((250, 250), Image.LANCZOS)
        original_photo = ImageTk.PhotoImage(original_img)
        self.original_image_panel.configure(image=original_photo)
        self.original_image_panel.image = original_photo

        processed_img = img_resized.resize((250, 250), Image.LANCZOS)
        processed_photo = ImageTk.PhotoImage(processed_img)
        self.processed_image_panel.configure(image=processed_photo)
        self.processed_image_panel.image = processed_photo

# Arayüz başlat
if __name__ == "__main__":
    root = tk.Tk()
    app = TahminUygulamasi(root)
    root.mainloop()
