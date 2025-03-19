import face_recognition as fr

def recognize_face(url_photo):
    photo = fr.load_image_file(url_photo)
    faces = fr.face_encodings(photo)
    
    if(len(faces) > 0):
        return True, faces
    return False, []

def get_faces():
    familiar_faces = []
    faces_names = []
    user = recognize_face("./img/sayran.png")
    
    if(user[0]):
        familiar_faces.append(user[1][0])
        faces_names.append("Sayran")
    
    return familiar_faces, faces_names