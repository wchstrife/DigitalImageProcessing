import cv2
import numpy as np
import skin_detector

def my_detector():
    #Open a simple image
    img=cv2.imread("./data/0007_01.jpg")

    #converting from gbr to hsv color space
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    #skin color range for hsv color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))


    HSV_result = cv2.bitwise_not(HSV_mask)
    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
    global_result=cv2.bitwise_not(global_mask)


    #show results
    cv2.imshow("1_HSV.jpg",HSV_result)
    cv2.imshow("2_YCbCr.jpg",YCrCb_result)
    cv2.imshow("3_global_result.jpg",global_result)
    cv2.imshow("Image.jpg",img)
    # cv2.imwrite("1_HSV.jpg",HSV_result)
    # cv2.imwrite("2_YCbCr.jpg",YCrCb_result)
    # cv2.imwrite("3_global_result.jpg",global_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def whitening(image, rate=0.3):
    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_HSV[:,:,-1] = np.minimum(image_HSV[:,:,-1] +  image_HSV[:,:,-1] * rate, 255).astype('uint8')
    image_RGB =  cv2.cvtColor(image_HSV, cv2.COLOR_HSV2BGR)

    return image_RGB

def get_skin_mask(image):
    return skin_detector.process(image)


if __name__ == "__main__":
    img_path = "./data/0015_01.jpg"
    image = cv2.imread(img_path)
    
    cv2.imshow("input", image)

    skin_mask = get_skin_mask(image)
    non_skin_mask = cv2.bitwise_not(skin_mask)
    #print(skin_mask.shape)
    
    image_light = whitening(image)
    #cv2.imshow("light", image_light)

    image_light = cv2.bitwise_and(image_light, image_light, mask=skin_mask)
    image_non_light = cv2.bitwise_and(image, image, mask=non_skin_mask)

    result = cv2.addWeighted(image_light, 1, image_non_light, 1, 0)

    #cv2.imshow("light", image_light)

    cv2.imshow("result", result)
    
    
    # result = whitening(result)
    # cv2.imshow("result2", result)


    

    #mask = skin_detector.process(image)
    #result = cv2.bitwise_and(image, mask)
    # cv2.imshow("input", image)
    # cv2.imshow("HSV", image_HSV)
    # cv2.imshow("RGB", image_RGB)
    #cv2.imshow("mask", result)


    cv2.waitKey(0)

 