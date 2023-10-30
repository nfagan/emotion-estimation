from mtcnn import MTCNN
import cv2
import numpy as np
import os
import glob

IM_PATH = 'D:\\data\\changlab\\ilker_collab\\flickr-faces'
# IM_PATH = 'D:\\data\\changlab\\ilker_collab\\flickr-faces-subset\\valid'
EXT = '.png'

if __name__ == '__main__':
  dst_p = os.path.join(IM_PATH, 'detections')
  os.makedirs(dst_p, exist_ok=True)

  detector = MTCNN()
  kp_names = ['nose', 'mouth_right', 'right_eye', 'left_eye', 'mouth_left']
  
  files = glob.glob(os.path.join(IM_PATH, f'*{EXT}'))
  for i in range(len(files)):
    print(f'{i + 1} of {len(files)}')
    f = files[i]
    dst_f = os.path.join(dst_p, os.path.split(f)[1].replace(EXT, '.txt'))

    if os.path.isfile(dst_f):
      continue

    img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    detects = detector.detect_faces(img)
    if len(detects) != 1:
      print('Expected 1 face in image.')
      continue

    kps = np.array([detects[0]['keypoints'][x] for x in kp_names], dtype=np.float64)
    np.savetxt(dst_f, kps)