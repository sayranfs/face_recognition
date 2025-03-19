import numpy as np
import face_recognition as fr
import cv2
from engine import get_faces

familiar_faces, faces_names = get_faces()
video_capture = cv2.VideoCapture(0)

while True: 
    ret, frame = video_capture.read()
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
    face_localization = fr.face_locations(rgb_frame)
    unknown_faces = fr.face_encodings(rgb_frame, face_localization)

    for (top, right, bottom, left), unknown_face in zip(face_localization, unknown_faces):
        results = fr.compare_faces(familiar_faces, unknown_face)
        print(results)

        face_distances = fr.face_distance(familiar_faces, unknown_face)
        id = np.argmin(face_distances)
        if results[id]:
            name = faces_names[id]
        else:
            name = "Desconhecido"
            
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('WEBCAM - Face Recognition', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()