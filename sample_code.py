from detector import detector, get_time
import cv2
import os


if __name__ =="__main__":
    outDir = 'out'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    d = detector()
    test_img = cv2.imread('test/images/4 (13)_1649859983.jpg')
    total_cars, img = d(test_img.copy())
    print(f'Total cars: {total_cars}')
    cv2.imwrite(os.path.join(outDir, f'out {get_time()}.jpg'), img)