import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import os
import datetime
import csv
import numpy as np
import shutil

# ── Resolve paths relative to this file ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
FACES_DIR  = os.path.join(DATA_DIR,   "faces")
ATT_FILE   = os.path.join(DATA_DIR,   "attendance.csv")
CASCADE    = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")

# -- Appearance --
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# -- Monochrome palette --
COL_BG        = "#0a0a0a"   # near-black window background
COL_PANEL     = "#141414"   # panel / card background
COL_BORDER    = "#2a2a2a"   # subtle border
COL_BTN       = "#1f1f1f"   # default button fill
COL_BTN_HOVER = "#2e2e2e"   # button hover
COL_BTN_DEST  = "#2a2a2a"   # destructive button (delete / clear)
COL_BTN_DEST_H= "#3a3a3a"
COL_TEXT      = "#ffffff"   # primary text
COL_MUTED     = "#888888"   # secondary / muted text
COL_SUCCESS   = "#ffffff"   # success feedback (white)
COL_WARN      = "#cccccc"   # warning / capture feedback
COL_ERR       = "#aaaaaa"   # error feedback


class App(ctk.CTk):
    # ── Init ──────────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__()

        self.title("Attendance System — Face Recognition")
        self.geometry("1250x750")
        self.minsize(1100, 650)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ── Paths / State ──────────────────────────────────────────────────────
        self.cadets_dir = FACES_DIR
        os.makedirs(self.cadets_dir, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        self.attendance_file = ATT_FILE

        self.video_capture = None
        self.is_camera_running = False
        self.after_id = None
        self.current_nav_frame = None
        self.photo_taken = False
        self.current_frame_image = None
        self.session_marks = []           # attendance entries made this session

        self.recognizer = None
        self.name_dict = {}
        self.face_cascade = cv2.CascadeClassifier(CASCADE)

        # ── Build UI ───────────────────────────────────────────────────────────
        self._build_sidebar()
        self._build_dashboard_frame()
        self._build_register_frame()
        self._build_attendance_frame()

        self.show_dashboard()

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=200, corner_radius=0,
                          fg_color=COL_PANEL)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(
            sb, text="Attendance\nSystem",
            font=ctk.CTkFont(size=18, weight="bold"), justify="center",
            text_color=COL_TEXT
        ).grid(row=0, column=0, padx=20, pady=(28, 20))

        nav_cfg = [
            ("Dashboard",       self.show_dashboard,   1),
            ("Register Face",   self.show_register,    2),
            ("Mark Attendance", self.show_attendance,  3),
        ]
        self._sidebar_btns = {}
        for text, cmd, row in nav_cfg:
            btn = ctk.CTkButton(
                sb, text=text, anchor="w", width=160, command=cmd,
                fg_color=COL_BTN, hover_color=COL_BTN_HOVER,
                text_color=COL_TEXT
            )
            btn.grid(row=row, column=0, padx=20, pady=6)
            self._sidebar_btns[text] = btn

        ctk.CTkLabel(
            sb, text="v2.0", font=ctk.CTkFont(size=11),
            text_color=COL_MUTED
        ).grid(row=6, column=0, padx=20, pady=10, sticky="s")

    # ══════════════════════════════════════════════════════════════════════════
    # DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    def _build_dashboard_frame(self):
        f = ctk.CTkFrame(self, corner_radius=0, fg_color=COL_BG)
        self.dashboard_frame = f

        f.grid_columnconfigure(0, weight=1)
        f.grid_columnconfigure(1, weight=1)
        f.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(
            f, text="Dashboard Overview",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=COL_TEXT
        ).grid(row=0, column=0, columnspan=2, pady=(24, 8))

        # ── Stat Cards ────────────────────────────────────────────────────────
        stat_row = ctk.CTkFrame(f, fg_color="transparent")
        stat_row.grid(row=1, column=0, columnspan=2, pady=8)

        self.card_registered = self._stat_card(stat_row, "Registered Faces", "0", 0)
        self.card_attendance = self._stat_card(stat_row, "Today's Attendance", "0", 1)

        # -- Left panel: Registered People --
        left = ctk.CTkFrame(f, fg_color=COL_PANEL)
        left.grid(row=2, column=0, sticky="nsew", padx=(20, 8), pady=12)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        hdr_left = ctk.CTkFrame(left, fg_color="transparent")
        hdr_left.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        hdr_left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hdr_left, text="Registered People",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=COL_TEXT
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            hdr_left, text="+ Register New",
            width=120, height=28,
            fg_color=COL_BTN, hover_color=COL_BTN_HOVER,
            text_color=COL_TEXT,
            command=self.show_register
        ).grid(row=0, column=1, sticky="e")

        # Scrollable list
        self.people_scroll = ctk.CTkScrollableFrame(left)
        self.people_scroll.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.people_scroll.grid_columnconfigure(0, weight=1)

        # -- Right panel: Attendance Log --
        right = ctk.CTkFrame(f, fg_color=COL_PANEL)
        right.grid(row=2, column=1, sticky="nsew", padx=(8, 20), pady=12)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        hdr_right = ctk.CTkFrame(right, fg_color="transparent")
        hdr_right.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        hdr_right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hdr_right, text="Today's Attendance Log",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=COL_TEXT
        ).grid(row=0, column=0, sticky="w")

        btn_row = ctk.CTkFrame(hdr_right, fg_color="transparent")
        btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        ctk.CTkButton(
            btn_row, text="Refresh", width=90, height=28,
            fg_color=COL_BTN, hover_color=COL_BTN_HOVER,
            text_color=COL_TEXT,
            command=self.update_dashboard
        ).grid(row=0, column=0, padx=(0, 6))

        ctk.CTkButton(
            btn_row, text="Clear Today", width=110, height=28,
            fg_color=COL_BTN_DEST, hover_color=COL_BTN_DEST_H,
            text_color=COL_TEXT,
            command=self._clear_today_attendance
        ).grid(row=0, column=1)

        self.att_log_scroll = ctk.CTkScrollableFrame(right)
        self.att_log_scroll.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.att_log_scroll.grid_columnconfigure((0, 1, 2), weight=1)

    def _stat_card(self, parent, label, value, col):
        card = ctk.CTkFrame(parent, width=180, height=80, fg_color=COL_PANEL)
        card.grid(row=0, column=col, padx=14, pady=4)
        card.grid_propagate(False)
        card.grid_rowconfigure((0, 1), weight=1)
        card.grid_columnconfigure(0, weight=1)

        val_lbl = ctk.CTkLabel(card, text=value,
                               font=ctk.CTkFont(size=30, weight="bold"),
                               text_color=COL_TEXT)
        val_lbl.grid(row=0, column=0, sticky="s")
        ctk.CTkLabel(card, text=label, font=ctk.CTkFont(size=12),
                     text_color=COL_MUTED).grid(row=1, column=0, sticky="n")
        return val_lbl

    def update_dashboard(self):
        """Rebuild both panels with fresh data."""
        reg_names = self._get_registered_names()
        today_rows = self._get_today_attendance()

        # Update stat cards
        self.card_registered.configure(text=str(len(reg_names)))
        self.card_attendance.configure(text=str(len(today_rows)))

        # ── Rebuild people list ────────────────────────────────────────────────
        for w in self.people_scroll.winfo_children():
            w.destroy()

        if not reg_names:
            ctk.CTkLabel(
                self.people_scroll,
                text="No faces registered yet.\nClick '+ Register New' to add someone.",
                text_color=COL_MUTED, justify="center"
            ).pack(pady=20)
        else:
            for idx, name in enumerate(sorted(reg_names)):
                row_f = ctk.CTkFrame(self.people_scroll, fg_color=COL_BORDER)
                row_f.pack(fill="x", padx=4, pady=3)
                row_f.grid_columnconfigure(1, weight=1)

                ctk.CTkLabel(
                    row_f, text=f"  {idx + 1}.",
                    font=ctk.CTkFont(size=13), width=30,
                    text_color=COL_MUTED
                ).grid(row=0, column=0, padx=(6, 0), pady=8)

                ctk.CTkLabel(
                    row_f, text=name,
                    font=ctk.CTkFont(size=14, weight="bold"),
                    anchor="w", text_color=COL_TEXT
                ).grid(row=0, column=1, sticky="w", padx=8)

                ctk.CTkButton(
                    row_f, text="Delete", width=80, height=28,
                    fg_color=COL_BTN_DEST, hover_color=COL_BTN_DEST_H,
                    text_color=COL_TEXT,
                    command=lambda n=name: self._delete_person(n)
                ).grid(row=0, column=2, padx=8, pady=6)

        # ── Rebuild attendance log ─────────────────────────────────────────────
        for w in self.att_log_scroll.winfo_children():
            w.destroy()

        # Header row
        for col, (txt, anchor) in enumerate([("Name", "w"), ("Time", "center"), ("Date", "center")]):
            ctk.CTkLabel(
                self.att_log_scroll, text=txt,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="gray", anchor=anchor
            ).grid(row=0, column=col, sticky="ew", padx=6, pady=(4, 2))

        if not today_rows:
            ctk.CTkLabel(
                self.att_log_scroll,
                text="No attendance marked today.",
                text_color=COL_MUTED
            ).grid(row=1, column=0, columnspan=3, pady=14)
        else:
            for r_idx, row in enumerate(today_rows, start=1):
                bg = COL_BORDER if r_idx % 2 == 0 else COL_BTN
                for c_idx, val in enumerate(row[:3]):
                    lbl = ctk.CTkLabel(
                        self.att_log_scroll, text=val,
                        font=ctk.CTkFont(size=13),
                        fg_color=bg, text_color=COL_TEXT
                    )
                    lbl.grid(row=r_idx, column=c_idx, sticky="ew", padx=2, pady=1)

    def _delete_person(self, name):
        """Delete a registered person's image and retrain the model."""
        for ext in (".jpg", ".jpeg", ".png"):
            path = os.path.join(self.cadets_dir, f"{name}{ext}")
            if os.path.exists(path):
                os.remove(path)
                break
        self.train_model()
        self.update_dashboard()

    def _clear_today_attendance(self):
        """Remove today's rows from the attendance CSV."""
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(self.attendance_file):
            return
        kept = []
        with open(self.attendance_file, "r", newline="") as f:
            kept = [row for row in csv.reader(f) if not (len(row) >= 3 and row[2] == today)]
        with open(self.attendance_file, "w", newline="") as f:
            csv.writer(f).writerows(kept)
        self.update_dashboard()

    # ══════════════════════════════════════════════════════════════════════════
    # REGISTER FRAME
    # ══════════════════════════════════════════════════════════════════════════
    def _build_register_frame(self):
        f = ctk.CTkFrame(self, corner_radius=0, fg_color=COL_BG)
        self.register_frame = f
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            f, text="Register New Face",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=COL_TEXT
        ).grid(row=0, column=0, pady=(24, 8))

        self.reg_video_label = ctk.CTkLabel(f, text="Starting camera...",
                                             font=ctk.CTkFont(size=16),
                                             text_color=COL_MUTED)
        self.reg_video_label.grid(row=1, column=0, pady=6)

        ctrl = ctk.CTkFrame(f, fg_color="transparent")
        ctrl.grid(row=2, column=0, pady=10)

        self.name_entry = ctk.CTkEntry(
            ctrl, placeholder_text="Enter full name", width=240, height=38,
            text_color=COL_TEXT
        )
        self.name_entry.grid(row=0, column=0, padx=8)

        self.btn_capture = ctk.CTkButton(
            ctrl, text="Take Photo", width=130, height=38,
            fg_color=COL_BTN, hover_color=COL_BTN_HOVER,
            text_color=COL_TEXT,
            command=self.take_photo
        )
        self.btn_capture.grid(row=0, column=1, padx=8)

        self.btn_confirm = ctk.CTkButton(
            ctrl, text="Confirm & Save", width=150, height=38,
            fg_color=COL_BTN, hover_color=COL_BTN_HOVER,
            text_color=COL_TEXT,
            state="disabled", command=self.confirm_photo
        )
        self.btn_confirm.grid(row=0, column=2, padx=8)

        self.lbl_reg_status = ctk.CTkLabel(
            f, text="Position your face in the frame, then click Take Photo.",
            text_color=COL_MUTED
        )
        self.lbl_reg_status.grid(row=3, column=0, pady=8)

    # ══════════════════════════════════════════════════════════════════════════
    # ATTENDANCE FRAME
    # ══════════════════════════════════════════════════════════════════════════
    def _build_attendance_frame(self):
        f = ctk.CTkFrame(self, corner_radius=0, fg_color=COL_BG)
        self.attendance_frame = f
        f.grid_columnconfigure(0, weight=2)
        f.grid_columnconfigure(1, weight=1)
        f.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            f, text="Mark Attendance - Live",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=COL_TEXT
        ).grid(row=0, column=0, columnspan=2, pady=(24, 8))

        cam_col = ctk.CTkFrame(f, fg_color="transparent")
        cam_col.grid(row=1, column=0, sticky="nsew", padx=(20, 8), pady=8)
        cam_col.grid_rowconfigure(0, weight=1)
        cam_col.grid_columnconfigure(0, weight=1)

        self.att_video_label = ctk.CTkLabel(cam_col, text="Starting camera...",
                                             font=ctk.CTkFont(size=16),
                                             text_color=COL_MUTED)
        self.att_video_label.grid(row=0, column=0)

        self.lbl_att_log = ctk.CTkLabel(cam_col, text="System Ready",
                                         font=ctk.CTkFont(size=14),
                                         text_color=COL_MUTED)
        self.lbl_att_log.grid(row=1, column=0, pady=8)

        log_col = ctk.CTkFrame(f, fg_color=COL_PANEL)
        log_col.grid(row=1, column=1, sticky="nsew", padx=(8, 20), pady=8)
        log_col.grid_rowconfigure(1, weight=1)
        log_col.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            log_col, text="Session Log",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=COL_TEXT
        ).grid(row=0, column=0, pady=(12, 4), padx=12, sticky="w")

        self.session_log_scroll = ctk.CTkScrollableFrame(log_col)
        self.session_log_scroll.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.session_log_scroll.grid_columnconfigure(0, weight=1)

    def _refresh_session_log(self):
        for w in self.session_log_scroll.winfo_children():
            w.destroy()
        if not self.session_marks:
            ctk.CTkLabel(
                self.session_log_scroll,
                text="No attendance marked yet.", text_color=COL_MUTED
            ).pack(pady=12)
            return
        for entry in reversed(self.session_marks):
            row_f = ctk.CTkFrame(self.session_log_scroll, fg_color=COL_BORDER)
            row_f.pack(fill="x", padx=4, pady=3)
            ctk.CTkLabel(row_f, text=entry['name'],
                         font=ctk.CTkFont(size=13, weight="bold"),
                         anchor="w", text_color=COL_TEXT).pack(side="left", padx=8, pady=6)
            ctk.CTkLabel(row_f, text=entry["time"],
                         font=ctk.CTkFont(size=12),
                         text_color=COL_MUTED, anchor="e").pack(side="right", padx=8)

    # ══════════════════════════════════════════════════════════════════════════
    # NAVIGATION
    # ══════════════════════════════════════════════════════════════════════════
    def _select_frame(self, name):
        self._stop_camera()

        if self.current_nav_frame is not None:
            self.current_nav_frame.grid_forget()

        if name == "dashboard":
            self.update_dashboard()
            self.current_nav_frame = self.dashboard_frame

        elif name == "register":
            self.lbl_reg_status.configure(
                text="Position your face in the frame, then click Take Photo.",
                text_color=COL_MUTED)
            self.name_entry.delete(0, "end")
            self.photo_taken = False
            self.btn_capture.configure(text="Take Photo")
            self.btn_confirm.configure(state="disabled")
            self.current_nav_frame = self.register_frame
            self._start_camera(self._update_register_camera)

        elif name == "attendance":
            self.session_marks = []
            self._refresh_session_log()
            self.train_model()
            self.current_nav_frame = self.attendance_frame
            self._start_camera(self._update_attendance_camera)

        self.current_nav_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)

    def show_dashboard(self):  self._select_frame("dashboard")
    def show_register(self):   self._select_frame("register")
    def show_attendance(self): self._select_frame("attendance")

    # ══════════════════════════════════════════════════════════════════════════
    # CAMERA
    # ══════════════════════════════════════════════════════════════════════════
    def _start_camera(self, callback):
        if not self.is_camera_running:
            self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if self.video_capture.isOpened():
                self.is_camera_running = True
                self.after_id = self.after(10, callback)

    def _stop_camera(self):
        if self.is_camera_running:
            self.is_camera_running = False
            if self.after_id:
                self.after_cancel(self.after_id)
            if self.video_capture:
                self.video_capture.release()

    def _update_register_camera(self):
        if not self.is_camera_running:
            return
        if not self.photo_taken:
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame_image = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    cv2.putText(frame, "Face Detected", (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                self._show_frame(self.reg_video_label, frame)
            else:
                self.lbl_reg_status.configure(
                    text="Camera error - is it used by another app?",
                    text_color=COL_ERR)
        self.after_id = self.after(15, self._update_register_camera)

    def _update_attendance_camera(self):
        if not self.is_camera_running:
            return
        ret, frame = self.video_capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))
            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                # White for recognized, gray for unknown
                color, name = (160, 160, 160), "Unknown"
                if self.recognizer is not None:
                    label_id, confidence = self.recognizer.predict(face_roi)
                    if confidence < 75:
                        name = self.name_dict.get(label_id, "Unknown")
                        color = (255, 255, 255)
                        self._mark_attendance(name)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x, y - 8),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
            self._show_frame(self.att_video_label, frame)
        self.after_id = self.after(15, self._update_attendance_camera)

    @staticmethod
    def _show_frame(label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(640, 460))
        label.configure(image=ctk_img)
        label.image = ctk_img

    # ══════════════════════════════════════════════════════════════════════════
    # REGISTER ACTIONS
    # ══════════════════════════════════════════════════════════════════════════
    def take_photo(self):
        if not self.photo_taken:
            if self.current_frame_image is None:
                self.lbl_reg_status.configure(
                    text="No camera feed yet.", text_color=COL_ERR)
                return
            self.photo_taken = True
            self.btn_capture.configure(text="Retake")
            self.btn_confirm.configure(state="normal")
            self.lbl_reg_status.configure(
                text="Photo captured! Enter name and click Confirm & Save.",
                text_color=COL_WARN)
        else:
            self.photo_taken = False
            self.btn_capture.configure(text="Take Photo")
            self.btn_confirm.configure(state="disabled")
            self.lbl_reg_status.configure(
                text="Camera resumed. Reposition and click Take Photo.",
                text_color=COL_MUTED)

    def confirm_photo(self):
        name = self.name_entry.get().strip()
        if not name:
            self.lbl_reg_status.configure(text="⚠️  Please enter a name first!", text_color="red")
            return
        if self.current_frame_image is None:
            self.lbl_reg_status.configure(text="⚠️  No image captured yet.", text_color="red")
            return

        filepath = os.path.join(self.cadets_dir, f"{name}.jpg")
        cv2.imwrite(filepath, self.current_frame_image)
        self.train_model()

        self.lbl_reg_status.configure(
            text=f"{name} registered successfully!", text_color=COL_SUCCESS)
        self.after(1200, self.show_dashboard)

    # ══════════════════════════════════════════════════════════════════════════
    # ATTENDANCE ACTION
    # ══════════════════════════════════════════════════════════════════════════
    def _mark_attendance(self, name):
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.datetime.now().strftime("%H:%M:%S")

        # Avoid duplicate in CSV today
        already = False
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, "r", newline="") as f:
                for row in csv.reader(f):
                    if len(row) >= 3 and row[0] == name and row[2] == date_str:
                        already = True
                        break

        if not already:
            with open(self.attendance_file, "a", newline="") as f:
                csv.writer(f).writerow([name, time_str, date_str])
            self.session_marks.append({"name": name, "time": time_str, "date": date_str})
            self._refresh_session_log()
            self.lbl_att_log.configure(
                text=f"Marked {name} present at {time_str}",
                text_color=COL_SUCCESS)

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL / DATA HELPERS
    # ══════════════════════════════════════════════════════════════════════════
    def train_model(self):
        faces, labels, self.name_dict = [], [], {}
        current_id = 0
        for filename in os.listdir(self.cadets_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                name = os.path.splitext(filename)[0]
                img = cv2.imread(os.path.join(self.cadets_dir, filename))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dets = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(dets) > 0:
                    (x, y, w, h) = dets[0]
                    faces.append(gray[y:y + h, x:x + w])
                    labels.append(current_id)
                    self.name_dict[current_id] = name
                    current_id += 1
        if faces:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.train(faces, np.array(labels))
            print("Model trained successfully.")
            return True
        self.recognizer = None
        return False

    def _get_registered_names(self):
        return [
            os.path.splitext(f)[0]
            for f in os.listdir(self.cadets_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def _get_today_attendance(self):
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        rows = []
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, "r", newline="") as f:
                for row in csv.reader(f):
                    if len(row) >= 3 and row[2] == today:
                        rows.append(row)
        return rows


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()