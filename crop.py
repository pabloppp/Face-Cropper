import dlib
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np

padding = 0.3 # percentage of width/height to add padding to

face_detector = dlib.get_frontal_face_detector()
Image.MAX_IMAGE_PIXELS = 999999999

def extract_padded_features(x):
	padd = int((x.right() - x.left()) * padding)
	return (x.left()-padd, x.top()-padd, x.right()+padd, x.bottom()+padd)

def detect_faces(image):
	detected_faces = face_detector(image, 1)
	face_frames = [extract_padded_features(x) for x in detected_faces]
	return face_frames

images_processed = 0
faces_generated = 0

for path in tqdm(glob.glob('./images/*.jpg')):
	try:
		images_processed += 1
		file_name = path.split('\\')[-1].replace('.jpg', '')

		image = Image.open(path) # io.imread(path)
		detected_faces = detect_faces(np.array(image))

		for n, face_rect in enumerate(detected_faces):
			faces_generated += 1
			face = image.crop(face_rect).resize((512, 512))
			face.save(f'./output/{file_name}_f{n+1}.jpg')
	except: 
		print(f"An exception occurred with image {path}")

print(f"Successfully generated {faces_generated} faces from {images_processed} images!!!")