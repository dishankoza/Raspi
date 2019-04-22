# Importing libraries
import sys
sys.path.append('/Users/kad99kev/anaconda3/envs/virtual/lib/python3.7/site-packages')
from threading import Thread
import cv2
import face_recognition
import numpy as np


class MyVideo:
    #Initialize variables
    def __init__(self):
        # Webcam reference
        self.video = cv2.VideoCapture(0)

        self.face_locations = []
        self.face_encodings = []

        self.known_face_encodings = []
        self.known_face_names = []

        self.face_number = 1;

    def run(self):
        process_this_frame = True
        cv2.namedWindow("Video")
        cv2.setMouseCallback('Video', self.mouseClick)

        while True:
            # Grab a single frame of video
            ret, frame = self.video.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)

                    # If a match was found in known_face_encodings, just use the first one.
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self.known_face_names[first_match_index]
                    else:
                        name = f"Face {self.face_number}"
                        self.known_face_names.append(name)
                        self.known_face_encodings.append(face_encoding)
                        self.face_number += 1

                    face_names.append(name)

            process_this_frame = not process_this_frame


            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(33) == ord('q'):
                break

            if cv2.waitKey(33) == ord('s'):
                print("File saved")
                np.savez('test', np.array(self.known_face_encodings), np.array(self.known_face_names))

            if cv2.waitKey(33) == ord('l'):
                print("File loaded")
                data = np.load('test.npz')
                self.known_face_encodings = data['arr_0'].tolist()
                self.known_face_names = data['arr_1'].tolist()


    def mouseClick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for location in self.face_locations:
                ymin = location[0]*4
                xmax = location[1]*4
                ymax = location[2]*4
                xmin = location[3]*4
            if xmin <= x <= xmax and ymin <= y <= ymax:
                self.known_face_names[self.face_locations.index(location)] = input("Enter new name: ")

    # Release handle to the webcam
    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

video = MyVideo()
video.run()


# Save the known_face_encodings and known_face_names
# Load the known_face_encodings and known_face_names
