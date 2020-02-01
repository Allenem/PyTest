import face_recognition

# Load the jpg files into numpy arrays
biden_image = face_recognition.load_image_file("img/biden.jpg")
obama_image = face_recognition.load_image_file("img/obama.jpg")
unknown_image = face_recognition.load_image_file("img/obama2.jpg")

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
try:
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

known_faces = [
    biden_face_encoding,
    obama_face_encoding
]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
results1 = face_recognition.compare_faces(known_faces, unknown_face_encoding)
results2 = face_recognition.face_distance(known_faces, unknown_face_encoding)

print("Is the unknown face a picture of Biden? {}".format(results1[0]))
print("Is the unknown face a picture of Obama? {}".format(results1[1]))
print("Is the unknown face a new person that we've never seen before? {}".format(not True in results1))

print("What't the distance between the unknown face & Biden? {}".format(results2[0]))
print("What't the distance between the unknown face & Obama? {}".format(results2[1]))