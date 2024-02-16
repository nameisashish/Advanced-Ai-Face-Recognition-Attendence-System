import cv2
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pickle
import csv
import face_recognition
from deepface import DeepFace
import numpy as np

class MultiFacerec:
    def __init__(self, model='face_recognition'):
        self.model = model
        self.frame_resizing = 0.5
        self.csv_file_path = os.path.join(os.path.expanduser("~"), "Downloads", "attendance.csv")
        self.absentees_csv_file_path = os.path.join(os.path.expanduser("~"), "Downloads", "absentees.csv")
        self.presentees = set()
        self.absentees = set()
        self.unique_entries = set()
        self.unique_names = set()

        # Lists to store known face encodings and names
        self.known_face_encodings = []
        self.known_face_names = []

        # Load or download face encodings
        self.load_or_download_encodings()

    def load_or_download_encodings(self):
        encoding_file_path = os.path.join(os.path.expanduser("~"), "Downloads", "face_encodings.pkl")

        # If encoding file exists, load it; otherwise, download and store it
        if os.path.exists(encoding_file_path):
            with open(encoding_file_path, 'rb') as file:
                encoding_data = pickle.load(file)
                self.known_face_encodings = encoding_data["encodings"]
                self.known_face_names = encoding_data["names"]
                print("Face encodings loaded from file.")
        else:
            self.download_and_store_encodings(encoding_file_path)

    def download_and_store_encodings(self, encoding_file_path):
        faces_folder_path = os.path.join(os.path.expanduser("~"), "Downloads", "Faces")

        for filename in os.listdir(faces_folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                name_usn, extension = os.path.splitext(filename)
                name, usn = name_usn.split('(')
                name = name.strip()
                usn = usn.rstrip(')').strip()

                suffix = ""
                count = self.known_face_names.count({"name": name, "usn": usn})
                if count > 0:
                    suffix = f"_{count + 1}"

                img_path = os.path.join(faces_folder_path, filename)
                img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_encoding = self.get_face_embedding(rgb_img)[0]
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append({"name": name, "usn": usn})
                print(f"Encoding image {name}{suffix} with USN {usn} loaded")

        encoding_data = {"encodings": self.known_face_encodings, "names": self.known_face_names}
        with open(encoding_file_path, 'wb') as file:
            pickle.dump(encoding_data, file)
            print("Face encodings saved to file.")

    def mark_attendance(self, name, usn, timestamp):
        if name == "Unknown" or (name, usn) in self.presentees or (name, usn) in self.unique_entries:
            return

        self.presentees.add((name, usn))
        self.unique_entries.add((name, usn))
        self.unique_names.add(name)

        if (name, usn) not in self.unique_names:
            self.known_face_names.append({"name": name, "usn": usn, "timestamp": timestamp})
            print(f"{name} ({usn}) marked present at {timestamp}")

    def mark_absentees(self):
        self.absentees = set((entry["name"], entry["usn"]) for entry in self.known_face_names)
        self.absentees -= self.presentees

    def sort_and_write_csv(self, file_path, data, has_timestamp=False, remove_unrecognized=True):
        unique_entries = set((entry["name"], entry["usn"]) for entry in data)
        unique_entries = unique_entries.intersection(self.presentees)
        sorted_data = sorted(data, key=lambda x: (x["name"], x["usn"]))

        unique_entries_dict = {}
        for entry in sorted_data:
            name_usn = (entry["name"], entry["usn"])
            if name_usn not in unique_entries_dict:
                unique_entries_dict[name_usn] = entry

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "USN", "Timestamp"] if has_timestamp else ["Name", "USN"])
            for entry in unique_entries_dict.values():
                if remove_unrecognized and (entry["name"], entry["usn"]) not in unique_entries:
                    continue

                usn = entry["usn"].split(')')[0]
                if has_timestamp:
                    writer.writerow([entry["name"], usn, entry.get("timestamp", "")])
                else:
                    writer.writerow([entry["name"], usn])

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_face, face_encoding, face_loc, frame) for face_encoding, face_loc in zip(face_encodings, face_locations)]

            for future in futures:
                future.result()

    def process_face(self, face_encoding, face_loc, frame):
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        confidence_tolerance = 0.6

        if np.min(face_distances) <= confidence_tolerance:
            best_match_index = np.argmin(face_distances)
            entry = self.known_face_names[best_match_index]
            name = entry["name"]
            usn = entry["usn"]

            if (name, usn) not in self.presentees:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                entry["timestamp"] = timestamp
                print(f"{entry['name']} ({entry['usn']}) marked present at {timestamp}")
                self.presentees.add((name, usn))

                face_loc = [coord * int(1 / self.frame_resizing) for coord in face_loc]
                self.draw_face_rectangle(frame, face_loc, name)

                self.mark_attendance(name, usn, timestamp)
        else:
            name = "Unknown"
            face_loc = [coord * int(1 / self.frame_resizing) for coord in face_loc]
            self.draw_face_rectangle(frame, face_loc, name)

    @staticmethod
    def draw_face_rectangle(frame, face_loc, name):
        y1, x2, y2, x1 = face_loc
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        cv2.putText(frame, f"{name}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

    def get_face_embedding(self, face_image):
        if self.model == 'face_recognition':
            face_locations = face_recognition.face_locations(face_image)
            face_encodings = face_recognition.face_encodings(face_image, face_locations)
        elif self.model == 'deepface':
            # Use DeepFace to get face embeddings
            face = DeepFace.detectFace(face_image, detector_backend='mtcnn')
            face_encodings = DeepFace.represent(face_image, model_name='OpenFace', enforce_detection=False)
        else:
            raise ValueError("Invalid model selection")

        return face_encodings

# Create an instance of MultiFacerec with the desired model
mfr = MultiFacerec(model='face_recognition')

# Open the URL using urllib
iphone_ip_address = 'http://192.168.137.104:4747/video/mjpeg'
cap = cv2.VideoCapture()
cap.open(iphone_ip_address)

if not cap.isOpened():
    print("Error: Couldn't open video stream.")
    exit()

frame_counter = 0
frame_skip = 4

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_counter += 1
    if frame_counter % frame_skip != 0:
        continue

    mfr.detect_known_faces(frame)

    resized_frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
    cv2.imshow("Frame", resized_frame)

    key = cv2.waitKey(1)
    if key == 27:
        mfr.mark_absentees()
        mfr.sort_and_write_csv(mfr.csv_file_path, mfr.known_face_names, has_timestamp=True, remove_unrecognized=True)

        if mfr.absentees:
            absentees_data = [{"name": name, "usn": usn} for name, usn in mfr.absentees]
            mfr.sort_and_write_csv(mfr.absentees_csv_file_path, absentees_data, has_timestamp=False, remove_unrecognized=False)

        break

cap.release()
cv2.destroyAllWindows()
