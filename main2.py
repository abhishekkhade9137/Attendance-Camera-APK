
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.core.window import Window

import datetime
import cv2

import os

import face_recognition
class CamInterface(BoxLayout):
    def __init__(self, **kwargs):
        self.counter=0
        cam=0
        super(CamInterface, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.camera = Camera(play=True, index=cam)  # Use primary camera
        self.add_widget(self.camera)
        self.capture_button = Button(text='Capture')
        self.capture_button.bind(on_press=self.capture)
        self.add_widget(self.capture_button)
        self.flip_button = Button(text='Flip')
        self.flip_button.bind(on_press=self.flip)
        self.add_widget(self.flip_button)
        self.play_button = Button(text='Play')
        self.play_button.bind(on_press=self.play)
        self.add_widget(self.play_button)
        self.submit_button = Button(text='Submit')
        self.submit_button.bind(on_press=self.submit)
        self.add_widget(self.submit_button)
    
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
    def submit(self,instance):
            self.remove_widget(self.submit_button)
            self.remove_widget(self.play_button)
            self.remove_widget(self.flip_button)
            self.remove_widget(self.capture_button)
            self.remove_widget(self.camera)

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

                self.faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                
                for (x, y, w, h) in self.faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Display the output
                #cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        self.match_faces(img_dir, "C:\\Users\\abhis\\Music\\Projects\\Attendance-Webcam\\resources\\cadets")

    def match_faces(self, taken_photos_dir, cadet_dir):
        try:
            known_faces = {}
            for filename in os.listdir(cadet_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(cadet_dir, filename)
                    img = face_recognition.load_image_file(img_path)
                    face_encoding = face_recognition.face_encodings(img)[0]
                    known_faces[filename] = face_encoding

            with open("attendance.txt", "a") as f:
                for filename in os.listdir(taken_photos_dir):
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        img_path = os.path.join(taken_photos_dir, filename)
                        img = face_recognition.load_image_file(img_path)
                        unknown_face_encoding = face_recognition.face_encodings(img)[0]
                        for known_filename, known_face_encoding in known_faces.items():
                            match_results = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding)
                            if match_results[0]:
                                print(f"Match found: {filename} matches {known_filename}")
                                current_date = datetime.date.today().strftime("%Y-%m-%d")
                                f.write(f"{known_filename.split('.')[0]} , {current_date}\n")
                            break
        except Exception as e:
            print(f"An error occurred: {e}")
    
class MyApp(App):
    def build(self):
        self.capture = CamInterface()
        self.process = ProcessPhotos()
        #self.capture.capture_button.bind(on_press=self.run_process)
        self.capture.submit_button.bind(on_press=self.run_process)
        return self.capture

    def run_process(self, instance):
        self.img_dir ="C:\\Users\\abhis\\Music\\Projects\\Attendance-Webcam\\resources\\takenimages"
        self.knowncadets="C:\\Users\\abhis\\Music\\Projects\\Attendance-Webcam\\resources\\cadets"
        print("////////RUNNING MATCH_FACES////////////////")
        self.process.match_faces(self.img_dir,self.knowncadets)
        print("..................................................")
        

if __name__ == '__main__':
    MyApp().run()