from face_features import extract_face_features, angle_between, point_after_rotation
import glob
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import cv2

images_processed = 0
faces_generated = 0
outer_spacing = 0.38

for path in tqdm(glob.glob('./images/*.jpg')):
	try:
		images_processed += 1
		file_name = path.split('\\')[-1].replace('.jpg', '')
	
		image = Image.open(path) # io.imread(path)
		face_features = extract_face_features(np.array(image))

		for n, features in enumerate(face_features):
			# face = image.crop(face_rect).resize((512, 512))
			# face.save(f'./output/{file_name}_f{n+1}.jpg')
			le = features['left_eye']
			lec = le[:,0].mean(), le[:,1].mean()

			re = features['right_eye']
			rec = re[:,0].mean(), re[:,1].mean()

			m  = features['mouth']
			mc = m[:,0].mean(), m[:,1].mean()

			angle = angle_between(lec, rec)

			lec = point_after_rotation(lec, angle, image.width, image.height)
			rec = point_after_rotation(rec, angle, image.width, image.height)

			face = image.copy()
			face = face.rotate(angle)
			draw = ImageDraw.Draw(face)

			# draw.ellipse([(lec[0]-8, lec[1]-8), (lec[0]+8, lec[1]+8)], fill=(255,0,0,128))
			# draw.ellipse([(rec[0]-8, rec[1]-8), (rec[0]+8, rec[1]+8)], fill=(255,0,0,128))

			cy = (lec[1] + rec[1]) / 2
			cx = (lec[0] + rec[0]) / 2
			dx = abs(rec[0] - lec[0])
			sx = ((dx * (outer_spacing*2) / (1-outer_spacing*2)) / 2) + dx / 2
			syu = sx*2 * outer_spacing
			syd = sx*2 * (1-outer_spacing)

			if sx*2 > 256:
				face = face.crop([cx-sx, cy-syu, cx+sx, cy+syd])
				face = face.resize((512, 512))
				face.save(f'./output/{file_name}_f{n+1}.jpg')

				faces_generated += 1

	except Exception as e:
		print(f"An exception occurred with image {path}")
		print(e)

print(f"Successfully generated {faces_generated} faces from {images_processed} images!!!")