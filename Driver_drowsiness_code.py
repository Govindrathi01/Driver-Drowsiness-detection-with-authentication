import cv2
import dlib
import numpy as np
import face_recognition
import pygame
from scipy.spatial import distance
import time

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_aspect_ratio = (A + B) / (2.0 * C)
    return eye_aspect_ratio

pygame.mixer.init()
pygame.mixer.music.load('C:\\Users\\HP\\Desktop\\dt2\\music.wav')  
cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

pygame.init()


unauthorized_sound = pygame.mixer.Sound("music.wav")

# Function to check if the detected face matches the driver's face
def is_driver_face(detected_face_encoding, driver_face_encodings):
    
    threshold = 0.6
    matches = face_recognition.compare_faces(driver_face_encodings, detected_face_encoding, tolerance=threshold)
    return any(matches)


authorized_drivers_encodings = []

# Prompt to choose whether to add a new driver or check an existing one
print("Press 'A' to Add New Driver or 'C' to Check Existing Driver")
key = cv2.waitKey(0) & 0xFF

if key == ord('a'):
    # Adding a new driver
    adding_new_driver = True
    print("Adding New Driver - Smile and Press 'C' to Capture")
else:
    # Using an existing driver
    adding_new_driver = False
    print("Checking for Existing Drivers")

counter = 0  # Counter for drowsiness detection

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if adding_new_driver:
        # Capture new driver's image
        faces = hog_face_detector(gray)

        for face in faces:
            # Extract the face region
            face_region = frame[face.top():face.bottom(), face.left():face.right()]

            # Check if face region is not empty before resizing
            if not face_region.size == 0:
                # Resize the face region to ensure consistent dimensions
                face_region_resized = cv2.resize(face_region, (128, 128))

                # Convert the face region to RGB format (required by face_recognition)
                face_region_rgb = cv2.cvtColor(face_region_resized, cv2.COLOR_BGR2RGB)

                # Extract the face encoding if faces are found
                face_encodings = face_recognition.face_encodings(face_region_rgb)
                if face_encodings:
                    detected_face_encoding = face_encodings[0]

                    # Update the list of authorized drivers
                    authorized_drivers_encodings.append(detected_face_encoding)
                    adding_new_driver = False

                    # Print success message
                    print("New driver added successfully!")

    else:
        # Checking for existing drivers
        faces = hog_face_detector(gray)

        for face in faces:
            # Extract the face region
            face_region = frame[face.top():face.bottom(), face.left():face.right()]

            # Check if face region is not empty before resizing
            if not face_region.size == 0:
                
                face_region_resized = cv2.resize(face_region, (128, 128))

                
                face_region_rgb = cv2.cvtColor(face_region_resized, cv2.COLOR_BGR2RGB)

                # Extract the face encoding if faces are found
                face_encodings = face_recognition.face_encodings(face_region_rgb)
                if face_encodings:
                    detected_face_encoding = face_encodings[0]

                    
                    if is_driver_face(detected_face_encoding, authorized_drivers_encodings):
                        # Drowsiness detection logic
                        face_landmarks = dlib_facelandmark(gray, face)
                        leftEye = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(36, 42)]
                        rightEye = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(42, 48)]

                        left_ear = calculate_EAR(leftEye)
                        right_ear = calculate_EAR(rightEye)

                        EAR = (left_ear + right_ear) / 2
                        EAR = round(EAR, 2)

                        if EAR < 0.20:
                            cv2.putText(frame, "DROWSY", (20, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
                            cv2.putText(frame, "Take a break", (20, 400),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                            print("Drowsy")
                            counter += 1
                            print(counter)

                            if counter >= 3:
                                print("Drowy")
                                pygame.mixer.music.play() 
                                break

                            time.sleep(1)

                        else:
                            counter = 0
                            print("else reached")

                   
                        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                    else:
                        
                        cv2.putText(frame, "Unauthorized Driver", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        unauthorized_sound.play()

   
    cv2.putText(frame, "Press 'A' to Add New Driver or 'C' to Check Existing Driver", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Driver's Face Recognition", frame)

    key = cv2.waitKey(1)


    if key == ord('a'):
        authorized_drivers_encodings = []  # Reset the list when adding a new driver
        adding_new_driver = True
        print("Adding New Driver - Smile and Press 'C' to Capture")


    elif key == ord('c'):
        
        print("Existing drivers:", len(authorized_drivers_encodings))

   
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()






   

