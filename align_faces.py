from os import listdir, makedirs
from tqdm import tqdm
import cv2
import face_alignment
from skimage import transform, io
from shutil import copy
import numpy as np

IMG_SIZE = (160, 160)
SRC = np.array([
            [54.706573, 73.85186],
            [105.045425, 73.573425],
            [80.036, 102.48086],
            [59.356144, 131.95071],
            [101.04271, 131.72014]], dtype=np.float32)

def align(img, dst):
    trans = transform.SimilarityTransform()
    trans.estimate(dst, SRC)
    M = trans.params[0:2, :]
    aligned = cv2.warpAffine(img, M, IMG_SIZE, borderValue=0.0)
    return aligned

if __name__ == '__main__':
    aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:0')

    for ROOT, TARGET in [('./data/test/', './data/test_aligned/'), ('./data/train/', './data/train_aligned')]:
        makedirs(TARGET, exist_ok=True)
        s = 0
        skip = 0

        for c in tqdm(listdir(ROOT)):
            makedirs(TARGET + c, exist_ok=True)

            for i in tqdm(listdir(ROOT + c)):
                img = io.imread(ROOT + c + '/' + i)

                try:
                    landmarks = aligner.get_landmarks(img)
                    points = landmarks[0]
                    p1 = np.mean(points[36:42, :], axis=0)
                    p2 = np.mean(points[42:48, :], axis=0)
                    p3 = points[33, :]
                    p4 = points[48, :]
                    p5 = points[54, :]
                    dst = np.array([p1, p2, p3, p4, p5], dtype=np.float32)
                    src = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    aligned = align(src, dst)
                    cv2.imwrite(TARGET + c + '/' + i, aligned)

                except Exception as e:
                    print(e)
                    copy(ROOT + c + '/' + i, TARGET + c)
                    skip += 1

                s += 1

        print('Failed: %d/%d'%(skip, s))