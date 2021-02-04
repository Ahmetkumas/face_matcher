from tensorflow.keras.models import load_model
from numpy import expand_dims
from numpy import asarray
import numpy as np
import imutils
import cv2
import time

MODEL_PATH = 'facenet_keras.h5'
DETECTION_MODEL_PATH = 'haarcascade_frontalface_default.xml'

class faceVerification():
    def __init__(self):
        self.model = self.load_model(MODEL_PATH)
        self.cascade_model = self.load_detection_model(DETECTION_MODEL_PATH)

    def load_model(self, model):
        # load facenet model
        return load_model(model)

    def load_detection_model(self, DETECTION_MODEL_PATH):
        # load face detection model
        return cv2.CascadeClassifier(DETECTION_MODEL_PATH)

    def load_img(self, image_path):
        # read image
        image = cv2.imread(image_path)
        return image

    def imshow(self, img):
        # display image
        cv2.imshow("image", img)
        key = cv2.waitKey(0)

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade_model.detectMultiScale(gray, 1.3, 5)
        if not len(list(faces)):
            return None
        #for (x, y, w, h) in faces:
            #img = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        (x, y, w, h)  = faces[0]
        return image[y:y + h, x:x + w]

    def get_embedding(self, model, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]

    def extract_face(self, face_pixels):
        # convert face pixels to feature vectors.
        newTrainX = list()
        embedding = self.get_embedding(self.model, face_pixels)
        newTrainX.append(embedding)
        newTrainX = asarray(newTrainX)
        return newTrainX

    def is_passed(self, value):
        # check the distance similarity
        # other distance methods can be implemented like cosine similarity
        match_result = False
        if value < 110:
            match_result = True
            return match_result
        return match_result

    def process(self, img1, img2):
        image1 = self.load_img(img1)
        image2 = self.load_img(img2)

        # detect faces using cascade
        face_a = self.detect_face(image1)
        face_b = self.detect_face(image2)

        if face_a is not None:
            resized_face_a = cv2.resize(face_a,(160,160))
            resized_face_b = cv2.resize(face_b,(160,160))

            face_features_a = self.extract_face(resized_face_a)
            face_features_b = self.extract_face(resized_face_b)

            distance = np.sum(np.abs(face_features_a - face_features_b))
            match_result = self.is_passed(distance)
            return {"match_result": match_result, 
                    "distance": distance}

        return {"match_result": None, 
                    "distance": None}

if __name__ == '__main__':
    FV = faceVerification()
    start = time.time()
    output = FV.process("ben1.jpg", "ben2.jpg")
    end = time.time()
    process_time = end - start
    print(" Match:",output["match_result"],"\n", "Distance:" ,output["distance"],"\n", "Process Time:", process_time)