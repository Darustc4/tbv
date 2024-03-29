from network.network import Predictor

import threading
import numpy as np
import os

import tkinter as tk
import customtkinter as ctk
import nrrd
from PIL import Image, ImageTk, ImageOps

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("dark-blue")

class BrainVisualizer(ctk.CTk):
    def __init__(self):
        self.default_path = "/home/daru/code/tbv/dataset/original/1_01-03-2018.nrrd"
        self.weights_path = "/home/daru/repos/tbv/visualizer/network/weights.pt"
        self.norm_path = "/home/daru/repos/tbv/visualizer/network/norm.json"
        
        self.predictor = Predictor(self.weights_path, self.norm_path)

        self.nrrd_headers = None
        self.nrrd_data = None

        self.axis_list = ['X', 'Y', 'Z']
        self.axis = self.axis_list[0]
        self.zoom_list = ['x1', 'x2', 'x3', 'x4', 'x5']
        self.zoom = 1
        self.lenience_list = ['Strict', 'Normal', 'Lenient']
        self.lenience = self.predictor.Lenience.NORMAL
        self.security_list = ['Low', 'Medium', 'High']
        self.security = 50

        self.shape_list = ['Raw', 'Square']
        self.square = False

        self.current_slice = 0
        self.total_slices = 0
        self.current_slice_img = None

        self.age = -1
        self.patient_id = -1
        self.tbv = -1
        self.pred_tbv = -1

        self.predict_busy = False

        self.create_tk()

    def create_tk(self):
        super().__init__()

        self.title("Brain Visualizer")
        self.geometry("1500x1400")

        self.create_frames()
        self.create_widgets()

    def create_frames(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0) # File Frame
        self.grid_rowconfigure(1, weight=1) # Image Frame
        self.grid_rowconfigure(2, weight=0) # Control Frame

        # File Frame
        self.tk_file_frame = ctk.CTkFrame(self, corner_radius=0)
        self.tk_file_frame.grid(row=0, column=0, sticky="nsew")

        self.tk_file_frame.grid_columnconfigure(0, weight=0)
        self.tk_file_frame.grid_columnconfigure(1, weight=0)
        self.tk_file_frame.grid_columnconfigure(2, weight=1)
        self.tk_file_frame.grid_rowconfigure(0, weight=1)
        self.tk_file_frame.grid_rowconfigure(1, weight=1)

        # Image Frame
        self.tk_image_frame = ctk.CTkFrame(self, corner_radius=0)
        self.tk_image_frame.grid(row=1, column=0, sticky="nsew")

        self.tk_image_frame.grid_columnconfigure(0, weight=1)
        self.tk_image_frame.grid_rowconfigure(0, weight=1)

        # Control Frame
        self.tk_control_frame = ctk.CTkFrame(self, corner_radius=0)
        self.tk_control_frame.grid(row=2, column=0, sticky="nsew")

        self.tk_control_frame.grid_columnconfigure(0, weight=1) # Dashboard Frame
        self.tk_control_frame.grid_columnconfigure(1, weight=0) # Slicer Frame
        self.tk_control_frame.grid_rowconfigure(0, weight=1)

        self.tk_dashboard_frame = ctk.CTkFrame(self.tk_control_frame, corner_radius=0)
        self.tk_dashboard_frame.grid(row=0, column=0, sticky="nsew")
        self.tk_slicer_frame = ctk.CTkFrame(self.tk_control_frame, corner_radius=0)
        self.tk_slicer_frame.grid(row=0, column=1, sticky="nsew")

        self.tk_dashboard_frame.grid_columnconfigure(0, weight=1)
        self.tk_dashboard_frame.grid_columnconfigure(1, weight=1)
        self.tk_dashboard_frame.grid_columnconfigure(2, weight=1)
        self.tk_dashboard_frame.grid_columnconfigure(3, weight=1)
        self.tk_dashboard_frame.grid_columnconfigure(4, weight=1)
        self.tk_dashboard_frame.grid_columnconfigure(5, weight=1)
        self.tk_dashboard_frame.grid_columnconfigure(6, weight=1)
        self.tk_dashboard_frame.grid_columnconfigure(7, weight=1)
        self.tk_dashboard_frame.grid_columnconfigure(8, weight=1)
        self.tk_dashboard_frame.grid_columnconfigure(9, weight=1)
        self.tk_dashboard_frame.grid_columnconfigure(10, weight=1)
        self.tk_dashboard_frame.grid_rowconfigure(0, weight=1)
        self.tk_dashboard_frame.grid_rowconfigure(1, weight=1)
        self.tk_dashboard_frame.grid_rowconfigure(2, weight=1)

        self.tk_slicer_frame.grid_columnconfigure(0, weight=1)
        self.tk_slicer_frame.grid_columnconfigure(1, weight=1)
        self.tk_slicer_frame.grid_rowconfigure(0, weight=1)
        self.tk_slicer_frame.grid_rowconfigure(1, weight=1)
        self.tk_slicer_frame.grid_rowconfigure(2, weight=1)

    def create_widgets(self):
        # File Frame
        self.tk_load_button = ctk.CTkButton(self.tk_file_frame, text="Load", command=self.load_data)
        self.tk_load_button.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self.tk_browse_nrrd_button = ctk.CTkButton(self.tk_file_frame, text="Browse NRRD", command=self.browse_nrrd)
        self.tk_browse_nrrd_button.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)

        self.tk_nrrd_path_var = tk.StringVar(self.tk_file_frame, value=self.default_path)
        self.tk_nrrd_path_entry = ctk.CTkEntry(self.tk_file_frame, textvariable=self.tk_nrrd_path_var)
        self.tk_nrrd_path_entry.grid(row=0, column=2, sticky="nsew")

        self.tk_message_var = tk.StringVar(self.tk_file_frame, value="Please load a file to visualize.")
        self.tk_message_label = ctk.CTkLabel(self.tk_file_frame, textvariable=self.tk_message_var, anchor="w")
        self.tk_message_label.grid(row=2, column=0, sticky="nsew", columnspan=3, padx=5)

        # Image Frame
        self.tk_image_canvas = tk.Canvas(self.tk_image_frame, bg='black')
        self.tk_image_canvas.grid(row=0, column=0, sticky="nsew")

        # Control Frame / Dashboard
        self.tk_id_var = tk.StringVar(self.tk_dashboard_frame, value="N/A")
        self.tk_id_label = ctk.CTkLabel(self.tk_dashboard_frame, text="ID", fg_color="gray")
        self.tk_id_label.grid(row=0, column=0, sticky="nsew")
        self.tk_id_container = ctk.CTkLabel(self.tk_dashboard_frame, textvariable=self.tk_id_var)
        self.tk_id_container.grid(row=1, column=0, sticky="nsew")

        self.tk_age_var = tk.StringVar(self.tk_dashboard_frame, value="N/A")
        self.tk_age_label = ctk.CTkLabel(self.tk_dashboard_frame, text="Age (Days)", fg_color="gray")
        self.tk_age_label.grid(row=0, column=1, sticky="nsew")
        self.tk_age_container = ctk.CTkLabel(self.tk_dashboard_frame, textvariable=self.tk_age_var)
        self.tk_age_container.grid(row=1, column=1, sticky="nsew")

        self.tk_tbv_var = tk.StringVar(self.tk_dashboard_frame, value="N/A")
        self.tk_tbv_label = ctk.CTkLabel(self.tk_dashboard_frame, text="TBV", fg_color="gray")
        self.tk_tbv_label.grid(row=0, column=2, sticky="nsew")
        self.tk_tbv_container = ctk.CTkLabel(self.tk_dashboard_frame, textvariable=self.tk_tbv_var)
        self.tk_tbv_container.grid(row=1, column=2, sticky="nsew")

        self.tk_tbv_pred_var = tk.StringVar(self.tk_dashboard_frame, value="N/A")
        self.tk_tbv_pred_label = ctk.CTkLabel(self.tk_dashboard_frame, text="Predicted TBV", fg_color="gray")
        self.tk_tbv_pred_label.grid(row=0, column=3, sticky="nsew")
        self.tk_tbv_pred_container = ctk.CTkLabel(self.tk_dashboard_frame, textvariable=self.tk_tbv_pred_var)
        self.tk_tbv_pred_container.grid(row=1, column=3, sticky="nsew")

        self.tk_axis_var = tk.StringVar(self.tk_dashboard_frame, value=self.axis_list[0])
        self.tk_axis_label = ctk.CTkLabel(self.tk_dashboard_frame, text="Axis", fg_color="gray")
        self.tk_axis_label.grid(row=0, column=4, sticky="nsew")
        self.tk_axis_cbox = ctk.CTkOptionMenu(self.tk_dashboard_frame, values=self.axis_list, command=self.axis_changed, variable=self.tk_axis_var)
        self.tk_axis_cbox.grid(row=1, column=4, sticky="nsew", padx=15, pady=15)

        self.tk_zoom_var = tk.StringVar(self.tk_dashboard_frame, value=self.zoom_list[0])
        self.tk_zoom_label = ctk.CTkLabel(self.tk_dashboard_frame, text="Zoom", fg_color="gray")
        self.tk_zoom_label.grid(row=0, column=5, sticky="nsew")
        self.tk_zoom_cbox = ctk.CTkOptionMenu(self.tk_dashboard_frame, values=self.zoom_list, command=self.zoom_changed, variable=self.tk_zoom_var)
        self.tk_zoom_cbox.grid(row=1, column=5, sticky="nsew", padx=15, pady=15)

        self.tk_shape_var = tk.StringVar(self.tk_dashboard_frame, value=self.shape_list[0])
        self.tk_shape_label = ctk.CTkLabel(self.tk_dashboard_frame, text="Shape", fg_color="gray")
        self.tk_shape_label.grid(row=0, column=6, sticky="nsew")
        self.tk_shape_toggle = ctk.CTkSegmentedButton(self.tk_dashboard_frame, values=self.shape_list, command=self.shape_changed, variable=self.tk_shape_var)
        self.tk_shape_toggle.grid(row=1, column=6, sticky="nsew")

        self.tk_current_slice_var = tk.StringVar(self.tk_dashboard_frame, value=(str)(self.current_slice))
        self.tk_current_slice_label = ctk.CTkLabel(self.tk_dashboard_frame, text="Current Slice", fg_color="gray")
        self.tk_current_slice_label.grid(row=0, column=7, sticky="nsew")
        self.tk_current_slice_container = ctk.CTkLabel(self.tk_dashboard_frame, textvariable=self.tk_current_slice_var)
        self.tk_current_slice_container.grid(row=1, column=7, sticky="nsew", padx=2, pady=2)

        self.tk_total_slices_var = tk.StringVar(self.tk_dashboard_frame, value=(str)(self.total_slices))
        self.tk_total_slices_label = ctk.CTkLabel(self.tk_dashboard_frame, text="Total Slices", fg_color="gray")
        self.tk_total_slices_label.grid(row=0, column=8, sticky="nsew")
        self.tk_total_slices_container = ctk.CTkLabel(self.tk_dashboard_frame, textvariable=self.tk_total_slices_var)
        self.tk_total_slices_container.grid(row=1, column=8, sticky="nsew")

        self.tk_lenience_var = tk.StringVar(self.tk_dashboard_frame, value=self.lenience_list[1])
        self.tk_lenience_label = ctk.CTkLabel(self.tk_dashboard_frame, text="Lenience", fg_color="gray")
        self.tk_lenience_label.grid(row=0, column=9, sticky="nsew")
        self.tk_lenience_cbox = ctk.CTkOptionMenu(self.tk_dashboard_frame, values=self.lenience_list, command=self.lenience_changed, variable=self.tk_lenience_var)
        self.tk_lenience_cbox.grid(row=1, column=9, sticky="nsew", padx=10, pady=15)

        self.tk_security_var = tk.StringVar(self.tk_dashboard_frame, value=self.security_list[1])
        self.tk_security_label = ctk.CTkLabel(self.tk_dashboard_frame, text="Security", fg_color="gray")
        self.tk_security_label.grid(row=0, column=10, sticky="nsew")
        self.tk_security_cbox = ctk.CTkOptionMenu(self.tk_dashboard_frame, values=self.security_list, command=self.security_changed, variable=self.tk_security_var)
        self.tk_security_cbox.grid(row=1, column=10, sticky="nsew", padx=10, pady=15)

        self.tk_predict_button = ctk.CTkButton(self.tk_dashboard_frame, text="Predict", command=self.predict)
        self.tk_predict_button.grid(row=2, column=9, columnspan=2, sticky="nsew", padx=10, pady=5)

        # Control Frame / Slicer
        self.tk_prev_slice_button = ctk.CTkButton(self.tk_slicer_frame, text="<", command=self.slice_changer(-1))
        self.tk_prev_slice_button.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.tk_next_slice_button = ctk.CTkButton(self.tk_slicer_frame, text=">", command=self.slice_changer(1))
        self.tk_next_slice_button.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)

        self.tk_10prev_slice_button = ctk.CTkButton(self.tk_slicer_frame, text="<<<", command=self.slice_changer(-10))
        self.tk_10prev_slice_button.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.tk_10next_slice_button = ctk.CTkButton(self.tk_slicer_frame, text=">>>", command=self.slice_changer(10))
        self.tk_10next_slice_button.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)

        self.tk_slice_slider_var = tk.DoubleVar(self.tk_slicer_frame, value=self.current_slice)
        self.tk_slice_slider = ctk.CTkSlider(self.tk_slicer_frame, from_=0, to=1, orientation=tk.HORIZONTAL, command=self.slice_slider_changed, variable=self.tk_slice_slider_var)
        self.tk_slice_slider.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=2, pady=2)

    def axis_changed(self, choice):
        self.axis = choice

        if self.total_slices == 0:
            return

        self.reset_slice_count()
        self.update_slice()

    def zoom_changed(self, choice):
        self.zoom = self.zoom_list.index(choice)+1

        if self.total_slices == 0:
            return

        self.update_slice()

    def shape_changed(self, choice):
        self.square = choice == "Square"

        if self.total_slices == 0:
            return

        self.update_slice()
    
    def lenience_changed(self, choice):
        if choice == "Strict": self.lenience = Predictor.Lenience.STRICT
        elif choice == "Normal": self.lenience = Predictor.Lenience.NORMAL
        elif choice == "Lenient": self.lenience = Predictor.Lenience.LENIENT
    
    def security_changed(self, choice):
        if choice == "Low": self.security = 25
        elif choice == "Normal": self.security = 50
        elif choice == "High": self.security = 100

    def predict(self):
        if self.predict_busy:
            self.send_message("Prediction already in progress")
            return

        if self.total_slices == 0:
            self.send_message("Can't predict. No file loaded")
            return 

        self.predict_busy = True

        self.send_message("Predicting...")

        # Asynchronously predict the TBV
        self.predict_thread = threading.Thread(
            target=self._predict_async, 
            args=(self.nrrd_data, self.nrrd_headers["spacings"], self.security, self.lenience)
        )
        self.predict_thread.start()

    def _predict_async(self, nrrd_data, spacings, bayes_runs, lenience):
        try:
            predicted_tbv = self.predictor(nrrd_data, spacings, bayes_runs=bayes_runs, lenience=lenience)
            
            if predicted_tbv is None:
                self.send_message("Prediction refused. For this image, the prediction is not reliable enough. Try changing the security or lenience settings if needed.")
                return
            
            self.pred_tbv = predicted_tbv
            
            # Update the GUI
            self.tk_tbv_pred_var.set((str)(self.pred_tbv))
            self.send_message("Predicted TBV: " + (str)(self.pred_tbv) + " cm3")
        except Exception as e:
            self.send_message("Prediction failed.")
            print(e)
        finally:
            self.predict_busy = False

    def slice_changer(self, delta):
        def change_slice():
            if self.total_slices == 0:
                return

            self.current_slice += delta

            if self.current_slice < 0:
                self.current_slice = 0
            elif self.current_slice >= self.total_slices:
                self.current_slice = self.total_slices - 1

            self.tk_current_slice_var.set((str)(self.current_slice))
            self.update_slice()

        return change_slice

    def slice_slider_changed(self, value):
        if self.total_slices == 0:
            return
        self.current_slice = (int)(value)
        self.tk_current_slice_var.set((str)(self.current_slice))
        self.update_slice()

    def browse_nrrd(self):
        filename = tk.filedialog.askopenfilename(initialdir = ".",title = "Select NRRD", filetypes = (("nrrd files","*.nrrd"), ("all files","*.*")))
        self.tk_nrrd_path_var.set(filename)

    def load_data(self):
        if self.predict_busy:
            self.send_message("Can't load data. A volume prediction is in progress.")
            return

        nrrd_path = self.tk_nrrd_path_var.get()

        # Parse NRRD filename to obtain patient ID and scan date
        try:
            filename, ext = os.path.splitext(os.path.basename(nrrd_path))
            self.patient_id = filename.split("_")[0]
        except Exception as e:
            return

        if ext != ".nrrd":
            self.send_message("NRRD file not found or can not be opened.")
            return

        try:
            self.nrrd_data, self.nrrd_headers = nrrd.read(nrrd_path)
        except Exception as e:
            self.send_message("NRRD file not found or can not be opened.")
            print(e)
            return

        self.age = self.nrrd_headers['age_days'] if 'age_days' in self.nrrd_headers else -1
        self.tbv = self.nrrd_headers['tbv'] if 'tbv' in self.nrrd_headers else -1
        self.pred_tbv = -1

        self.reset_slice_count()
        self.update_slice()

        # Update the GUI
        self.tk_id_var.set(self.patient_id)
        if self.age != -1: self.tk_age_var.set(self.age)
        else:              self.tk_age_var.set("N/A")
        if self.tbv != -1: self.tk_tbv_var.set(self.tbv)
        else:              self.tk_tbv_var.set("N/A")
        self.tk_total_slices_var.set(self.total_slices)
        self.tk_tbv_pred_var.set("N/A")

        self.send_message("File loaded successfully.")

    def send_message(self, message):
        self.tk_message_var.set(message)
        print(message)

    def reset_slice_count(self):
        self.current_slice = 0
        self.total_slices = self.nrrd_data.shape[self.axis_list.index(self.axis)]

        self.tk_total_slices_var.set(self.total_slices)
        
        self.tk_slice_slider.configure(to=self.total_slices-1)
        self.tk_current_slice_var.set(0)
        self.tk_slice_slider_var.set(0)

    def update_slice(self):
        self.tk_slice_slider_var.set(self.current_slice)

        if self.axis == "X":   self.pixel_array = self.nrrd_data[self.current_slice, :, :]
        elif self.axis == "Y": self.pixel_array = self.nrrd_data[:, self.current_slice, :]
        elif self.axis == "Z": self.pixel_array = self.nrrd_data[:, :, self.current_slice]

        img = Image.fromarray(np.uint8(self.pixel_array)).convert('RGB')

        if self.zoom > 1 or self.square:
            height = self.pixel_array.shape[0]*self.zoom
            width = self.pixel_array.shape[1]*self.zoom

            if self.square: img = img.resize((height, height), Image.NEAREST)
            else:           img = img.resize((height, width), Image.NEAREST)

        self.current_slice_img = ImageTk.PhotoImage(ImageOps.expand(img, border=1, fill=(255,0,0)))
        self.tk_image_canvas.create_image(self.tk_image_canvas.winfo_width()/2, self.tk_image_canvas.winfo_height()/2, image=self.current_slice_img, anchor=tk.CENTER)

if __name__ == "__main__":
    app = BrainVisualizer()
    app.mainloop()

