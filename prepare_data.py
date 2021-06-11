from icrawler.builtin import GoogleImageCrawler
from tqdm import tqdm
from os import listdir, makedirs, rename
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from random import sample, randint
from shutil import move

DETECTED = './detected/'
RAW = './raw/'
LABELS_LIST = 'original_labels.txt'
TEST = './test/'
IMG_SIZE = (160, 160)

def download_raw(labels, quantity_each=25):
    for i in tqdm(range(len(labels))):
        google_crawler = GoogleImageCrawler(storage={'root_dir': ROOT + str(i)}, feeder_threads=1, parser_threads=2, downloader_threads=4)
        google_crawler.crawl(keyword=labels[i], max_num=quantity_each)

def detect(labels):
    detector = MTCNN()

    for i in tqdm(range(len(labels))):
        j = 0
        makedirs(DETECTED + str(i), exist_ok=True)

        for img in tqdm(listdir(RAW + str(i))):
            image = Image.open(RAW + '%d/'%i + img)
            image = image.convert('RGB')
            pixels = np.asarray(image)
            results = detector.detect_faces(pixels)

            for result in results:
                x1, y1, w, h = result['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + w, y1 + h
                box = pixels[y1:y2, x1:x2]
                box = Image.fromarray(box)
                box = box.resize(IMG_SIZE)
                box.save(DETECTED + '%d/%d.jpg'%(i, j))
                j += 1

def create_test(labels):
    for i in tqdm(range(len(labels))):
        imgs = listdir(DETECTED + str(i))
        choices = sample(range(len(imgs)), randint(4, 5))

        makedirs(TEST + str(i), exist_ok=True)
        for choice in choices:
            move(DETECTED + '%d/%s'%(i, imgs[choice]), TEST + str(i))

def rename_files(labels):
    j = 0
    for p in [DETECTED, TEST]:
        print(p)
        for i in tqdm(range(len(labels))):
            for img in listdir(p + str(i)):
                rename(p + '%d/%s'%(i, img), p + '%d/img_%d.jpg'%(i, j))
                j += 1

if __name__ == '__main__':
    labels = [line.strip() for line in open(LABELS_LIST, 'r', encoding='UTF-8')]
    
    #download_raw(labels)
    
    #detect(labels)
    # + Eliminate images not belong to one person or abnormal, by hand

    #create_test(labels)

    #rename_files(labels)