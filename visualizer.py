import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import tkinter as tk
import customtkinter as ctk
import nrrd
from PIL import Image, ImageTk

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("dark-blue")

class BrainVisualizer(ctk.CTk):
    def __init__(self):
        self.default_nrrd_path = "/home/daru/code/projects/tbv/dataset/01_4146518/2018_04_13_4146518.nrrd"
        self.default_meta_path = "/home/daru/code/projects/tbv/dataset/labels.csv"

        self.nrrd_data = None
        self.meta_data = None

        self.expected_meta_columns = set(["id", "birthdate", "scandate", "tbv"])

        self.axis_list = ['X', 'Y', 'Z']
        self.axis = self.axis_list[0]
        self.current_slice = 0
        self.total_slices = 0
        self.current_slice_img = None

        self.birth_date = "Not Loaded"
        self.scan_date = "Not Loaded"
        self.age = 0
        self.patient_id = 0
        self.measured_tbv = 0

        self.create_tk()

    def create_tk(self):
        super().__init__()

        self.title("Brain Visualizer")
        self.geometry("1000x900")

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
        self.tk_control_frame.grid_columnconfigure(0, weight=1)
        self.tk_control_frame.grid_columnconfigure(1, weight=1)
        self.tk_control_frame.grid_columnconfigure(2, weight=1)
        self.tk_control_frame.grid_columnconfigure(3, weight=1)
        self.tk_control_frame.grid_columnconfigure(4, weight=1)
        self.tk_control_frame.grid_columnconfigure(5, weight=1)
        self.tk_control_frame.grid_columnconfigure(6, weight=1)
        self.tk_control_frame.grid_columnconfigure(7, weight=1)
        self.tk_control_frame.grid_rowconfigure(0, weight=1)
        self.tk_control_frame.grid_rowconfigure(1, weight=1)
        self.tk_control_frame.grid_rowconfigure(2, weight=1)

    def create_widgets(self):
        # File Frame
        self.tk_load_button = ctk.CTkButton(self.tk_file_frame, text="Load", command=self.load_data)
        self.tk_load_button.grid(row=0, column=0, sticky="nsew", rowspan=2, padx=2, pady=2)

        self.tk_browse_nrrd_button = ctk.CTkButton(self.tk_file_frame, text="Browse NRRD", command=self.browse_nrrd)
        self.tk_browse_nrrd_button.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)

        self.tk_browse_meta_button = ctk.CTkButton(self.tk_file_frame, text="Browse Metadata", command=self.browse_meta)
        self.tk_browse_meta_button.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)

        self.tk_nrrd_path_var = tk.StringVar(self.tk_file_frame, value=self.default_nrrd_path)
        self.tk_nrrd_path_entry = ctk.CTkEntry(self.tk_file_frame, textvariable=self.tk_nrrd_path_var)
        self.tk_nrrd_path_entry.grid(row=0, column=2, sticky="nsew")

        self.tk_meta_path_var = tk.StringVar(self.tk_file_frame, value=self.default_meta_path)
        self.tk_meta_path_entry = ctk.CTkEntry(self.tk_file_frame, textvariable=self.tk_meta_path_var)
        self.tk_meta_path_entry.grid(row=1, column=2, sticky="nsew")

        self.tk_message_var = tk.StringVar(self.tk_file_frame, value="Please load a file to visualize.")
        self.tk_message_label = ctk.CTkLabel(self.tk_file_frame, textvariable=self.tk_message_var, anchor="w")
        self.tk_message_label.grid(row=2, column=0, sticky="nsew", columnspan=3, padx=5)

        # Image Frame
        self.tk_image_canvas = tk.Canvas(self.tk_image_frame, bg='black')
        self.tk_image_canvas.pack(fill="both", expand=True)

        # Control Frame
        self.tk_id_var = tk.StringVar(self.tk_control_frame, value=(str)(self.patient_id))
        self.tk_id_label = ctk.CTkLabel(self.tk_control_frame, text="ID", fg_color="gray80")
        self.tk_id_label.grid(row=0, column=0, sticky="nsew")
        self.tk_id_container = ctk.CTkLabel(self.tk_control_frame, textvariable=self.tk_id_var)
        self.tk_id_container.grid(row=1, column=0, sticky="nsew")

        self.tk_birth_var = tk.StringVar(self.tk_control_frame, value=self.birth_date)
        self.tk_birth_label = ctk.CTkLabel(self.tk_control_frame, text="Birth Date", fg_color="gray80")
        self.tk_birth_label.grid(row=0, column=1, sticky="nsew")
        self.tk_birth_container = ctk.CTkLabel(self.tk_control_frame, textvariable=self.tk_birth_var)
        self.tk_birth_container.grid(row=1, column=1, sticky="nsew")

        self.tk_scan_var = tk.StringVar(self.tk_control_frame, value=self.scan_date)
        self.tk_scan_label = ctk.CTkLabel(self.tk_control_frame, text="Scan Date", fg_color="gray80")
        self.tk_scan_label.grid(row=0, column=2, sticky="nsew")
        self.tk_scan_container = ctk.CTkLabel(self.tk_control_frame, textvariable=self.tk_scan_var)
        self.tk_scan_container.grid(row=1, column=2, sticky="nsew")

        self.tk_age_var = tk.StringVar(self.tk_control_frame, value=(str)(self.age))
        self.tk_age_label = ctk.CTkLabel(self.tk_control_frame, text="Age (Days)", fg_color="gray80")
        self.tk_age_label.grid(row=0, column=3, sticky="nsew")
        self.tk_age_container = ctk.CTkLabel(self.tk_control_frame, textvariable=self.tk_age_var)
        self.tk_age_container.grid(row=1, column=3, sticky="nsew")

        self.tk_tbv_var = tk.StringVar(self.tk_control_frame, value=(str)(self.measured_tbv))
        self.tk_tbv_label = ctk.CTkLabel(self.tk_control_frame, text="TBV", fg_color="gray80")
        self.tk_tbv_label.grid(row=0, column=4, sticky="nsew")
        self.tk_tbv_container = ctk.CTkLabel(self.tk_control_frame, textvariable=self.tk_tbv_var)
        self.tk_tbv_container.grid(row=1, column=4, sticky="nsew")

        self.tk_axis_var = tk.StringVar(self.tk_control_frame, value=self.axis)
        self.tk_axis_label = ctk.CTkLabel(self.tk_control_frame, text="Axis", fg_color="gray80")
        self.tk_axis_label.grid(row=0, column=5, sticky="nsew")

        self.tk_axis_cbox = ctk.CTkComboBox(self.tk_control_frame, values=self.axis_list, command=self.axis_changed, state="readonly", variable=self.tk_axis_var)
        self.tk_axis_cbox.grid(row=1, column=5, sticky="nsew", padx=2, pady=2)

        self.tk_current_slice_var = tk.StringVar(self.tk_control_frame, value=(str)(self.current_slice))
        self.tk_current_slice_label = ctk.CTkLabel(self.tk_control_frame, text="Current Slice", fg_color="gray80")
        self.tk_current_slice_label.grid(row=0, column=6, sticky="nsew")
        self.tk_current_slice_container = ctk.CTkLabel(self.tk_control_frame, textvariable=self.tk_current_slice_var)
        self.tk_current_slice_container.grid(row=1, column=6, sticky="nsew")

        self.tk_total_slices_var = tk.StringVar(self.tk_control_frame, value=(str)(self.total_slices))
        self.tk_total_slices_label = ctk.CTkLabel(self.tk_control_frame, text="Total Slices", fg_color="gray80")
        self.tk_total_slices_label.grid(row=0, column=7, sticky="nsew")
        self.tk_total_slices_container = ctk.CTkLabel(self.tk_control_frame, textvariable=self.tk_total_slices_var)
        self.tk_total_slices_container.grid(row=1, column=7, sticky="nsew")

        self.tk_prev_slice_button = ctk.CTkButton(self.tk_control_frame, text="<", command=self.prev_slice)
        self.tk_prev_slice_button.grid(row=0, column=8, sticky="nsew", rowspan=2, padx=2, pady=2)
        self.tk_next_slice_button = ctk.CTkButton(self.tk_control_frame, text=">", command=self.next_slice)
        self.tk_next_slice_button.grid(row=0, column=9, sticky="nsew", rowspan=2, padx=2, pady=2)

    def axis_changed(self, choice):
        self.axis = choice

        self.reset_slice_count()
        self.update_slice()

    def next_slice(self):
        if self.current_slice >= self.total_slices - 1:
            return

        self.current_slice += 1
        self.tk_current_slice_var.set((str)(self.current_slice))
        self.update_slice()

    def prev_slice(self):
        if self.current_slice <= 0:
            return

        self.current_slice -= 1
        self.tk_current_slice_var.set((str)(self.current_slice))
        self.update_slice()

    def browse_nrrd(self):
        filename = tk.filedialog.askopenfilename(initialdir = ".",title = "Select NRRD", filetypes = (("nrrd files","*.nrrd"), ("all files","*.*")))
        self.tk_nrrd_path_var.set(filename)

    def browse_meta(self):
        filename = tk.filedialog.askopenfilename(initialdir = ".",title = "Select Metadata", filetypes = (("csv files","*.csv"), ("all files","*.*")))
        self.tk_meta_path_var.set(filename)

    def load_data(self):
        nrrd_path = self.tk_nrrd_path_var.get()
        meta_path = self.tk_meta_path_var.get()

        # Parse NRRD filename to obtain patient ID and scan date
        try:
            filename, ext = os.path.splitext(os.path.basename(nrrd_path))
            split_filename = filename.split("_")
            self.patient_id = split_filename[-1]
            self.scan_date = datetime.datetime(year=(int)(split_filename[0]), month=(int)(split_filename[1]), day=(int)(split_filename[2]))
        except Exception as e:
            print(e)
            self.send_message("The NRRD file must be named following the pattern '<year>_<month>_<day>_<id>'.")
            return

        if ext != ".nrrd":
            self.send_message("NRRD file not found or can not be opened.")
            return

        try:
            self.nrrd_data = nrrd.read(nrrd_path)
        except Exception as e:
            self.send_message("NRRD file not found or can not be opened.")
            return

        # Parse metadata
        try:
            self.meta_data = pd.read_csv(meta_path)
        except Exception as e:
            self.send_message("Metadata file not found or can not be opened.")
            return

        if not self.expected_meta_columns.issubset(set(self.meta_data.columns)):
            self.send_message("The metadata file does not contain the columns 'id', 'birthdate', 'scandate' and 'tbv'.")
            return

        string_scan_date = self.scan_date.strftime("%d/%m/%Y")              # IMPORTANT: Format the scan date to match the format in the metadata file
        entry = self.meta_data[(self.meta_data["id"] == (int)(self.patient_id)) & (self.meta_data["scandate"] == string_scan_date)]

        if entry.empty:
            self.send_message(f"The metadata file does not contain an entry for the patient with id '{self.patient_id}' and scan date '{string_scan_date}'.")
            return

        entry = entry.iloc[0]
        string_birth_date = entry["birthdate"]
        self.birth_date = datetime.datetime.strptime(string_birth_date, "%d/%m/%Y") # Again, format accordingly
        self.age = (self.scan_date - self.birth_date).days
        self.measured_tbv = entry["tbv"]

        # Update the GUI
        self.tk_id_var.set(self.patient_id)
        self.tk_scan_var.set(string_scan_date)
        self.tk_birth_var.set(string_birth_date)
        self.tk_age_var.set(self.age)
        self.tk_tbv_var.set(self.measured_tbv)
        self.tk_total_slices_var.set(self.total_slices)

        self.reset_slice_count()
        self.update_slice()

        self.send_message("Files loaded successfully.")

    def send_message(self, message):
        self.tk_message_var.set(message)
        print(message)

    def reset_slice_count(self):
        self.current_slice = 0
        self.total_slices = self.nrrd_data[0].shape[self.axis_list.index(self.axis)]
        self.tk_total_slices_var.set(self.total_slices)

    def update_slice(self):
        if self.axis == "X":   self.pixel_array = self.nrrd_data[0][self.current_slice, :, :]
        elif self.axis == "Y": self.pixel_array = self.nrrd_data[0][:, self.current_slice, :]
        elif self.axis == "Z": self.pixel_array = self.nrrd_data[0][:, :, self.current_slice]

        img = Image.fromarray(np.uint8(self.pixel_array)).convert('RGB')
        self.current_slice_img = ImageTk.PhotoImage(img)
        self.tk_image_canvas.create_image(self.tk_image_canvas.winfo_width()/2, self.tk_image_canvas.winfo_height()/2, image=self.current_slice_img, anchor=tk.CENTER)

if __name__ == "__main__":
    app = BrainVisualizer()
    app.mainloop()

