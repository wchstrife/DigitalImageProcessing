import cv2
import numpy

def gaussian(x, sigma):
    return (1.0/(2*numpy.pi*(sigma**2))) * numpy.exp(-(x**2) / (2*(sigma**2)))

def distance(x1, y1, x2, y2):
    return numpy.sqrt(numpy.abs((x1-x2)**2 - (y1-y2)**2))

def bilateral_filter(image, diameter, sigma_i, sigma_s, face_pos):
    #new_image = numpy.zeros(image.shape)
    new_image = image

    for (x, y, w, h) in face_pos: # 对图像中每个人脸做双边滤波
        for chanel in range(3):
            for row in range(x, x + w):
                for col in range(y, y + h):
                    wp_total = 0            # 模板的总权值
                    filtered_image = 0      # 中心点的权值
                    for k in range(diameter):
                        for l in range(diameter):
                            n_x = row - (k - diameter/2)
                            n_y = col - (l - diameter/2)
                            # if n_x >= len(image):
                            #     n_x -= len(image)
                            # if n_y >= len(image[0]):
                            #     n_y -= len(image[0])
                            if n_x < len(image)  and n_x > 0 and  n_y < len(image[0]) and n_y > 0 : # 忽略边缘
                                gi = gaussian(image[int(n_x)][int(n_y)][chanel] - image[row][col][chanel], sigma_i)
                                gs = gaussian(distance(n_x, n_y, row, col), sigma_s)
                                wp = gi * gs
                                filtered_image = (filtered_image) + (image[int(n_x)][int(n_y)][chanel] * wp)
                                wp_total = wp_total + wp
                    filtered_image = filtered_image // wp_total
                    new_image[row][col][chanel] = int(numpy.round(filtered_image))

    return new_image

# 检测人脸
def detect(filename):
    
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 进行人脸检测
    # 返回一个列表，列表里边每一项是一个框起人脸的矩形(x, y, w, h)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(faces)
    
    return faces


if __name__ == "__main__":
    filename = "./data/lena512color.tiff"
    image = cv2.imread(filename, 1) # 读取图像
    image_old = image

    face_pos = detect(filename)    # 检测人脸位置

    cv2.imshow("original image", image) # 显示原始图像
    cv2.imwrite("original_image.jpg", image)

    image_new = bilateral_filter(image_old, 7, 20.0, 20.0, face_pos)

    # 给人脸加框
    for x, y, w, h in face_pos:
        image_new = cv2.rectangle(image_new, (x, y), (x+w, y+h), (255, 0, 0), 2)  

    cv2.imshow("process image", image_new) # 显示处理后图像
    cv2.imwrite("process_image.jpg", image_new)

    cv2.waitKey()

    # filtered_image_OpenCV = cv2.bilateralFilter(image, 7, 20.0, 20.0)
    # cv2.imwrite("filtered_image_OpenCV.png", filtered_image_OpenCV)
    # image_own = bilateral_filter(image, 7, 20.0, 20.0)
    # cv2.imwrite("filtered_image_own.png", image_own)