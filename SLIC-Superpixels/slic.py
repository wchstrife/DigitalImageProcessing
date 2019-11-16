import math
from skimage import io, color
from tqdm import trange
import numpy as np

class Cluster(object):
    cluster_index = 1       # 类变量，所有的实例共享

    def __init__(self, h, w, l=0, a=0, b=0):   # 初始化聚类中心
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b
        self.pixels = []
        self.no = self.cluster_index  
        Cluster.cluster_index += 1
    
    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()

class SLIC(object):

    '''
    @filename : 输入图像
    @K : 超像素个数
    @M : 
    '''
    def __init__(self, filepath, K, M):
        self.K = K                  # 超像素个数    
        self.M = M                  # LAB颜色差异常数，通常取10-40
        self.data = self.open_image(filepath)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width   # 像素总个数
        self.S = int(math.sqrt(self.N / self.K))        # 超像素间的距离

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)


    def open_image(self, path):
        img_rgb = io.imread(path)
        img_lab = color.rgb2lab(img_rgb)
        return img_lab
    
    def save_image(self, path, img_lab):
        img_rgb = color.lab2rgb(img_lab)
        io.imsave(path, img_rgb)

    # 将(h, w)的位置作为聚类中心
    def generate_cluster(self, h, w):
        h = int(h)
        w = int(w)
        return Cluster(h, w, self.data[h][w][0], self.data[h][w][1], self.data[h][w][2])

    # 初始化聚类中心
    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.generate_cluster(h, w))
                w += self.S
            w = self.S / 2
            h += self.S

    # 计算(h, w)处的梯度
    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + self.data[h + 1][w + 1][1] - self.data[h][w][1] + self.data[h + 1][w + 1][2] - self.data[h][w][2]
        return gradient

    # 计算3*3邻域的最小梯度
    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):     
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient

    # 计算聚类中心2S周围区域的距离，选择聚类中心
    def assignment(self):
        for cluster in self.clusters:
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if h < 0 or h >= self.image_height or w < 0 or w >= self.image_width:   # 排除边界
                        continue
                    L, A, B = self.data[h][w]
                    dc = math.sqrt(math.pow(L - cluster.l, 2) + math.pow(A - cluster.a, 2) + math.pow(B - cluster.b, 2))
                    ds = math.sqrt(math.pow(h - cluster.h, 2) + math.pow(w - cluster.w, 2))
                    #D = math.sqrt(math.pow(dc / self.M, 2) + math.pow(ds / self.S, 2))
                    D = math.sqrt(math.pow(dc, 2) + math.pow(ds / self.S, 2) * math.pow(self.M, 2))           # 用更新后的公式
                    # 更新当前点的聚类中心
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D
    
    # 更新聚类中心
    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
            _h = int(sum_h / number)
            _w = int(sum_w / number)
            cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    # 晶格化：聚类块内均值
    def average_cluster_lab(self):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            sum_L = 0
            sum_A = 0
            sum_B = 0
            count = 0
            for p in cluster.pixels:
                sum_L += image_arr[p[0]][p[1]][0]
                sum_A += image_arr[p[0]][p[1]][1]
                sum_B += image_arr[p[0]][p[1]][2]
                count += 1
            for p in cluster.pixels:    # 更新每个聚类中点的LAB为平均值
                image_arr[p[0]][p[1]][0] = sum_L / count
                image_arr[p[0]][p[1]][1] = sum_A / count
                image_arr[p[0]][p[1]][2] = sum_B / count
            # 更新聚类中心的值
            image_arr[cluster.h][cluster.w][0] = sum_L / count
            image_arr[cluster.h][cluster.w][1] = sum_A / count
            image_arr[cluster.h][cluster.w][2] = sum_B / count

        return image_arr

    # 迭代, 默认为10次
    def run_SLIL(self, save_path, n = 10):
        self.init_clusters()    # 生成聚类
        self.move_clusters()    # 调整聚类中心
        for i in trange(n):
            self.assignment()
            self.update_cluster()
        

        img_lab = self.average_cluster_lab()   # 晶格化
        self.save_image(save_path, img_lab)  

if __name__ == "__main__":
    input_path = './Lenna.png'

    iterations = 10 # 迭代次数

    output_path = './result_K_1000_M30.png'
    K = 1000        # 超像素个数
    M = 30          # 颜色的距离[1-40]
    p = SLIC(input_path, K, M)
    p.run_SLIL(output_path, iterations)

    output_path = './result_K_2000_M30.png'
    K = 2000         
    M = 30          
    p = SLIC(input_path, K, M)
    p.run_SLIL(output_path, iterations)

    output_path = './result_K_5000_M30.png'
    K = 5000         
    M = 30          
    p = SLIC(input_path, K, M)
    p.run_SLIL(output_path, iterations)

    # output_path = './result_K_1000_M10.png'
    # K = 1000        
    # M = 10         
    # p = SLIC(input_path, K, M)
    # p.run_SLIL(output_path, iterations)

    # output_path = './result_K_2000_M10.png'
    # K = 2000         
    # M = 10          
    # p = SLIC(input_path, K, M)
    # p.run_SLIL(output_path, iterations)

    # output_path = './result_K_5000_M10.png'
    # K = 5000         
    # M = 10        
    # p = SLIC(input_path, K, M)
    # p.run_SLIL(output_path, iterations)
    


        
