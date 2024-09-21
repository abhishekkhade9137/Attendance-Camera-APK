
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager,Screen
import datetime
import cv2

import os

import face_recognition
class Home(Screen):
    def __init__(self, **kwargs):
        super(Home, self).__init__(**kwargs)
        self.name = 'home'  # Set the name of the screen

        layout = BoxLayout(orientation='vertical')  # Create a vertical BoxLayout

        self.attendance_button = Button(text='Attendance', size_hint=(1, 0.5))
        self.attendance_button.bind(on_press=self.switch_to_cam_interface)
        layout.add_widget(self.attendance_button)

        self.add_button = Button(text='Add', size_hint=(1, 0.5))
        layout.add_widget(self.add_button)

        self.add_widget(layout)  # Add the layout to the screen

    def switch_to_cam_interface(self, instance):
        self.manager.current = 'cam_interface' # Add the layout to the screen  # Add the layout to the screen
        

        #self.view_button = Button(text='View')
        #self.view_button.bind(on_press=self.attendance)
        #self.add_widget(self.view_button)

class Add(BoxLayout):
    def __init__(self, **kwargs):
        super(Home, self).__init__(**kwargs)
        cam=0
        self.camera = Camera(play=True, index=cam)  # Use primary camera
        self.add_widget(self.camera)

    


class CamInterface(Screen):
    def __init__(self, **kwargs):
        self.counter=0
        cam=0
        super(CamInterface, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.name = 'cam_interface'
        self.orientation = 'vertical'
        self.camera = Camera(play=True, index=cam)  # Use primary camera
        layout.add_widget(self.camera)
        self.capture_button = Button(text='Capture')
        self.capture_button.bind(on_press=self.capture)
        layout.add_widget(self.capture_button)

        self.flip_button = Button(text='Flip')
        self.flip_button.bind(on_press=self.flip)
        layout.add_widget(self.flip_button)

        self.play_button = Button(text='Play')
        self.play_button.bind(on_press=self.play)
        layout.add_widget(self.play_button)

        #self.add_button = Button(text='Add')
        #self.add_button.bind(on_press=self.add)
        #self.add_widget(self.submit_button)

        self.submit_button = Button(text='Submit')
        self.submit_button.bind(on_press=self.submit)
        layout.add_widget(self.submit_button)
        self.add_widget(layout)
    def capture(self, instance):
        self.counter=self.counter+1
        self.camera.export_to_png("C:\\Users\\abhis\\Music\\Projects\\Attendance-Webcam\\resources\\takenimages\\IMG_{}.png".format(self.counter))
        #print("Captured")
    def flip(self,instance):
        if(self.choice==0):
            self.choice=1
        else:
            self.choice=0
    def play(self, instance):
        self.camera.play = not self.camera.play

    #def add(self,instance):


    def submit(self,instance):
        self.clear_widgets()
        self.process = ProcessPhotos()  # Create an instance of ProcessPhotos
        self.img_dir ="C:\\Users\\abhis\\Music\\Projects\\Attendance-Webcam\\resources\\takenimages"
        self.knowncadets="C:\\Users\\abhis\\Music\\Projects\\Attendance-Webcam\\resources\\cadets"
        self.process.scanfaces()
        self.process.match_faces(self.img_dir,self.knowncadets)
        self.process.empty_takenimages()
    



          # Remove all widgets, including the BoxLayout
        # or
          # Remove the specific BoxLayout instance
        

class ProcessPhotos(BoxLayout):
    def __init__(self, **kwargs):
        self.faces = []
        super(ProcessPhotos, self).__init__(**kwargs)

    def scanfaces(self, **kwargs):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img_dir ="C:\\Users\\abhis\\Music\\Projects\\Attendance-Webcam\\resources\\takenimages"
        for filename in os.listdir(img_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Check if the file is an image
                img_path = os.path.join(img_dir, filename)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.faces = face_cascade.detectMultiScale(gray,1.6, 4)
                for (x, y, w, h) in self.faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imshow('img', img)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
        self.match_faces(img_dir, "C:\\Users\\abhis\\Music\\Projects\\Attendance-Webcam\\resources\\cadets")

    def match_faces(self, taken_photos_dir, cadet_dir):
        try:
            known_faces = {} #filename:encoding
            print("iterating through known cadets")
            for filename in os.listdir(cadet_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(cadet_dir, filename)
                    img = face_recognition.load_image_file(img_path)
                    face_encoding = face_recognition.face_encodings(img)[0]
                    known_faces[filename] = face_encoding
            print("iterating through unknown taken images")
            with open("attendance.txt", "a") as f:
                for filename in os.listdir(taken_photos_dir):
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        img_path = os.path.join(taken_photos_dir, filename)
                        img = face_recognition.load_image_file(img_path)
                        print(face_recognition.face_encodings(img))
                        unknown_face_encoding = face_recognition.face_encodings(img)[0]
                        for known_filename, known_face_encoding in known_faces.items():
                            match_results = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding,0.6)
                            print("match resutls are")
                            print(match_results)
                            if match_results[0]:
                                print(f"Match found: {filename} matches {known_filename}")
                                current_date = datetime.date.today().strftime("%Y-%m-%d")
                                f.write(f"{known_filename.split('.')[0]} , {current_date}\n")
                            break
        except Exception as e:
            print(f"An error occurred: {e}")
    

    def empty_takenimages(self):
        folder_path = 'C:\\Users\\abhis\\Music\\Projects\\Attendance-Webcam\\resources\\takenimages'
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")
        print("All files in taken_images folder have been deleted")
class MyApp(App):
    def build(self):
        self.sm = ScreenManager()
        home_screen = Home(name='home')
        cam_interface_screen = CamInterface(name='cam_interface')  # Create a CamInterface screen
        self.sm.add_widget(home_screen)
        self.sm.add_widget(cam_interface_screen)  # Add the CamInterface screen to the ScreenManager
        self.sm.current = 'home'  # Start with the home screen
        #self.cam_interface_screen.submit_button.bind(on_press=self.run_process)
        return self.sm
        

 

if __name__ == '__main__':
    MyApp().run()