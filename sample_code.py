from detector import detector, get_time
import cv2
import os


if __name__ =="__main__":
    outDir = 'out'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    d = detector()
    test_img = cv2.imread('test.jpg')
    total_cars, img = d(test_img.copy())            #calling the detector class with the image, and getting the total cars and the image with the bounding boxes
    print(f'Total cars: {total_cars}')
    cv2.imwrite(os.path.join(outDir, f'out {get_time()}.jpg'), img)