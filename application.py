
# flask packages
from flask import Flask, send_file
from flask_restful import Resource, Api
from flask_cors import CORS
from flask_restful.reqparse import RequestParser

# image handling packages
import io
from PIL import Image
import werkzeug
import cv2
import numpy as np

# detector packages

from mtcnn import MTCNN

app = Flask(__name__)
CORS(app)
api = Api(app)
mtcnn_detector = MTCNN()

# class NoiseFilterGenerator():
#     def __init__(self, confidence_levels: list = [0.3, 0.5, 0.7, 0.9]):
#         self.confidence_levels = confidence_levels

#     @confidence_levels.setter
#     def confidence_levels(self, levels: list):
#         self.confidence_levels = levels

#     def generateImageWithNoise(self, img):
#         for l in self.confidence_levels:
#             pass


def detectFaces(img, detector_type='mtcnn'):
    try:
        if detector_type == 'mtcnn':
            faces = mtcnn_detector.detect_faces(img)
            boxes = [f['box'] for f in faces]
            confs = [[f['confidence'], ] for f in faces]
            return [b+c for b, c in zip(boxes, confs)]
        # elif detector_type == 's3fd':
        #     fd = face_detector.FaceAlignmentDetector()
        #     faces = fd.detect_face(img, with_landmarks=False)
        #     return [f[:4].tolist()+f[4].tolist() for f in faces]
        else:
            return []
    except Exception as e:
        return []


def getImageFromPayload(parser):
    parser.add_argument(
        'image', type=werkzeug.datastructures.FileStorage, location='files')
    args = parser.parse_args()
    file = args['image']
    file_type = file.mimetype
    file_ext = file.filename.split('.')[-1]
    filestr = file.read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.COLOR_BGR2RGB)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, args, file_type, file_ext.lower()


class GenerateNoise(Resource):
    def post(self):
        try:
            parser = RequestParser()
            parser.add_argument('threshold')
            img, args, file_type, file_ext = getImageFromPayload(parser)
            threshold = int(args['threshold'])
            g_img_np = cv2.blur(img, (threshold, threshold))
            g_img = Image.fromarray(g_img_np.astype('uint8'))
            # https://stackoverflow.com/questions/56946969/how-to-send-an-image-directly-from-flask-server-to-html
            f_object = io.BytesIO()
            file_ext = 'jpeg' if file_ext == 'jpg' else file_ext
            g_img.save(f_object, file_ext)
            f_object.seek(0)
            return send_file(f_object, mimetype=file_type)
        except Exception as e:
            return None

    def get(self):
        return 'generateNoise'


class Home(Resource):
    def get(self):
        return 'hello'


class DetectFace(Resource):
    def post(self):
        try:
            parser = RequestParser()
            parser.add_argument('detector')
            img, args, _, _ = getImageFromPayload(parser)
            t = str(args['detector'])
            res = detectFaces(img, t)
            print(res)
            return res
        except Exception as e:
            return str(e)


api.add_resource(Home, '/')
api.add_resource(DetectFace, '/detect')
api.add_resource(GenerateNoise, '/generate')

if __name__ == '__main__':
    app.run(debug=True)
