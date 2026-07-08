import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import os
import datetime
import csv
import numpy as np

# Set appearance mode and color theme
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Attendance App - Face Recognition")
        self.geometry("1000x650")

        # Set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Attendance System", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_dashboard = ctk.CTkButton(self.sidebar_frame, text="Dashboard", command=self.show_dashboard)
        self.btn_dashboard.grid(row=1, column=0, padx=20, pady=10)

        self.btn_register = ctk.CTkButton(self.sidebar_frame, text="Register Face", command=self.show_register)
        self.btn_register.grid(row=2, column=0, padx=20, pady=10)

        self.btn_attendance = ctk.CTkButton(self.sidebar_frame, text="Mark Attendance", command=self.show_attendance)
        self.btn_attendance.grid(row=3, column=0, padx=20, pady=10)

        # --- Main Frames ---
        self.dashboard_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.register_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.attendance_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        
        # Current frame pointer
        self.current_frame = None

        # Setup variables
        self.video_capture = None
        self.is_camera_running = False
        self.after_id = None
        
        self.cadets_dir = "resources/cadets"
        if not os.path.exists(self.cadets_dir):
            os.makedirs(self.cadets_dir)
            
        self.recognizer = None
        self.name_dict = {}
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        self.setup_dashboard()
        self.setup_register()
        self.setup_attendance()

        # Initialize with Dashboard
        self.show_dashboard()

    def train_model(self):
        faces = []
        labels = []
        self.name_dict = {}
        current_id = 0

        for filename in os.listdir(self.cadets_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(self.cadets_dir, filename)
                
                img = cv2.imread(img_path)
                if img is None: continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                detected_faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                if len(detected_faces) > 0:
                    (x, y, w, h) = detected_faces[0]
                    faces.append(gray[y:y+h, x:x+w])
                    labels.append(current_id)
                    self.name_dict[current_id] = name
                    current_id += 1

        if len(faces) > 0:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.train(faces, np.array(labels))
            print("Model trained successfully.")
            return True
        return False

    def get_stats(self):
        registered = len([name for name in os.listdir(self.cadets_dir) if name.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        attendance_count = 0
        if os.path.exists('attendance.csv'):
            with open('attendance.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3 and row[2] == today:
                        attendance_count += 1
                        
        return registered, attendance_count

    # --- Frame Setups ---
    def setup_dashboard(self):
        self.dashboard_frame.grid_columnconfigure(0, weight=1)
        self.dashboard_frame.grid_columnconfigure(1, weight=1)
        
        self.lbl_title = ctk.CTkLabel(self.dashboard_frame, text="Dashboard Overview", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_title.grid(row=0, column=0, columnspan=2, pady=20)
        
        self.lbl_reg_count = ctk.CTkLabel(self.dashboard_frame, text="Registered Faces: 0", font=ctk.CTkFont(size=18))
        self.lbl_reg_count.grid(row=1, column=0, pady=20)
        
        self.lbl_att_count = ctk.CTkLabel(self.dashboard_frame, text="Today's Attendance: 0", font=ctk.CTkFont(size=18))
        self.lbl_att_count.grid(row=1, column=1, pady=20)

        self.btn_refresh = ctk.CTkButton(self.dashboard_frame, text="Refresh Data", command=self.update_dashboard_stats)
        self.btn_refresh.grid(row=2, column=0, columnspan=2, pady=20)
        
    def update_dashboard_stats(self):
        reg, att = self.get_stats()
        self.lbl_reg_count.configure(text=f"Registered Faces: {reg}")
        self.lbl_att_count.configure(text=f"Today's Attendance: {att}")

    def setup_register(self):
        self.register_frame.grid_columnconfigure(0, weight=1)
        
        self.lbl_reg_title = ctk.CTkLabel(self.register_frame, text="Register New Face", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_reg_title.grid(row=0, column=0, pady=20)
        
        # Camera display label
        self.reg_video_label = ctk.CTkLabel(self.register_frame, text="")
        self.reg_video_label.grid(row=1, column=0, pady=10)
        
        # Name input
        self.name_entry = ctk.CTkEntry(self.register_frame, placeholder_text="Enter Name")
        self.name_entry.grid(row=2, column=0, pady=10)
        
        self.btn_capture = ctk.CTkButton(self.register_frame, text="Capture & Register", command=self.capture_face)
        self.btn_capture.grid(row=3, column=0, pady=10)
        
        self.lbl_reg_status = ctk.CTkLabel(self.register_frame, text="")
        self.lbl_reg_status.grid(row=4, column=0, pady=10)

    def setup_attendance(self):
        self.attendance_frame.grid_columnconfigure(0, weight=1)
        
        self.lbl_att_title = ctk.CTkLabel(self.attendance_frame, text="Live Attendance", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_att_title.grid(row=0, column=0, pady=20)
        
        # Camera display label
        self.att_video_label = ctk.CTkLabel(self.attendance_frame, text="")
        self.att_video_label.grid(row=1, column=0, pady=10)
        
        self.lbl_att_log = ctk.CTkLabel(self.attendance_frame, text="System Ready", font=ctk.CTkFont(size=14))
        self.lbl_att_log.grid(row=2, column=0, pady=10)

    # --- Navigation ---
    def select_frame(self, name):
        self.stop_camera()
        
        if self.current_frame is not None:
            self.current_frame.grid_forget()
            
        if name == "dashboard":
            self.update_dashboard_stats()
            self.current_frame = self.dashboard_frame
        elif name == "register":
            self.lbl_reg_status.configure(text="")
            self.name_entry.delete(0, 'end')
            self.current_frame = self.register_frame
            self.start_camera(self.update_register_camera)
        elif name == "attendance":
            self.train_model() # Train before starting attendance
            self.current_frame = self.attendance_frame
            self.start_camera(self.update_attendance_camera)
            
        self.current_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

    def show_dashboard(self): self.select_frame("dashboard")
    def show_register(self): self.select_frame("register")
    def show_attendance(self): self.select_frame("attendance")

    # --- Camera Logic ---
    def start_camera(self, callback):
        if not self.is_camera_running:
            self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if self.video_capture.isOpened():
                self.is_camera_running = True
                self.after_id = self.after(10, callback)

    def stop_camera(self):
        if self.is_camera_running:
            self.is_camera_running = False
            if self.after_id:
                self.after_cancel(self.after_id)
            if self.video_capture:
                self.video_capture.release()

    def update_register_camera(self):
        if self.is_camera_running:
            ret, frame = self.video_capture.read()
            if ret:
                # Keep original frame for capture
                self.current_frame_image = frame.copy()
                
                # Draw boxes for display
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                # Convert to ImageTk format
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_image)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(640, 480))
                
                self.reg_video_label.configure(image=ctk_img)
                self.reg_video_label.image = ctk_img
                
            self.after_id = self.after(15, self.update_register_camera)

    def update_attendance_camera(self):
        if self.is_camera_running:
            ret, frame = self.video_capture.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
                
                for (x, y, w, h) in faces:
                    face_roi = gray[y:y+h, x:x+w]
                    color = (0, 0, 255)
                    name = "Unknown"
                    
                    if self.recognizer is not None:
                        label_id, confidence = self.recognizer.predict(face_roi)
                        if confidence < 75:
                            name = self.name_dict.get(label_id, "Unknown")
                            color = (0, 255, 0)
                            self.mark_attendance(name)
                            
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_image)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(640, 480))
                
                self.att_video_label.configure(image=ctk_img)
                self.att_video_label.image = ctk_img
                
            self.after_id = self.after(15, self.update_attendance_camera)

    # --- Actions ---
    def capture_face(self):
        name = self.name_entry.get().strip()
        if not name:
            self.lbl_reg_status.configure(text="Please enter a name!", text_color="red")
            return
            
        if hasattr(self, 'current_frame_image'):
            # Save the raw image
            filepath = os.path.join(self.cadets_dir, f"{name}.jpg")
            cv2.imwrite(filepath, self.current_frame_image)
            self.lbl_reg_status.configure(text=f"Successfully registered {name}!", text_color="green")
            self.name_entry.delete(0, 'end')
        else:
            self.lbl_reg_status.configure(text="Camera not ready", text_color="red")

    def mark_attendance(self, name):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        attendance_file = 'attendance.csv'
        
        already_marked = False
        if os.path.exists(attendance_file):
            with open(attendance_file, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 3 and row[0] == name and row[2] == date_str:
                        already_marked = True
                        break

        if not already_marked:
            with open(attendance_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, time_str, date_str])
            msg = f"Marked {name} Present at {time_str}"
            self.lbl_att_log.configure(text=msg, text_color="green")

if __name__ == "__main__":
    app = App()
    app.mainloop()