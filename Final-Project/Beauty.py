import cv2
import dlib
import numpy as np


# 五官父类
class Organ():

	def __init__(self, img, img_hsv, temp_img, temp_hsv, landmarks, name, ksize=None):
		self.img = img
		self.img_hsv = img_hsv
		self.landmarks = landmarks
		self.name = name
		self.get_rect()
		self.shape = (int(self.bottom-self.top), int(self.right-self.left))
		self.size = self.shape[0] * self.shape[1] * 3
		self.move = int(np.sqrt(self.size/3)/20)
		self.ksize = self.get_ksize()
		self.patch_img, self.patch_hsv = self.get_patch(self.img), self.get_patch(self.img_hsv)
		self.set_temp(temp_img, temp_hsv)
		self.patch_mask = self.get_mask_relative()

	# 获取定位方框
	def get_rect(self):
		y, x = self.landmarks[:, 1], self.landmarks[:, 0]
		self.top, self.bottom, self.left, self.right = np.min(y), np.max(y), np.min(x), np.max(x)

	# 空间滤波的核函数
	def get_ksize(self, rate=15):
		size = max([int(np.sqrt(self.size/3)/rate), 1])
		size = (size if size%2==1 else size+1)
		return(size, size)

	
	def get_patch(self, img):
		shape = img.shape
		return img[np.max([self.top-self.move, 0]): np.min([self.bottom+self.move, shape[0]]), np.max([self.left-self.move, 0]): np.min([self.right+self.move, shape[1]])]

	def set_temp(self, temp_img, temp_hsv):
		self.img_temp, self.hsv_temp = temp_img, temp_hsv
		self.patch_img_temp, self.patch_hsv_temp = self.get_patch(self.img_temp), self.get_patch(self.hsv_temp)

	
	def confirm(self):
		self.img[:], self.img_hsv[:] = self.img_temp[:], self.hsv_temp[:]

	
	def update_temp(self):
		self.img_temp[:], self.hsv_temp[:] = self.img[:], self.img_hsv[:]

	# 勾画凸多边形
	def _draw_convex_hull(self, img, points, color):
		points = cv2.convexHull(points)
		cv2.fillConvexPoly(img, points, color=color)

	# 获得局部相对坐标遮盖
	def get_mask_relative(self, ksize=None):
		if ksize == None:
			ksize = self.ksize
		landmarks_re = self.landmarks.copy()
		landmarks_re[:, 1] -= np.max([self.top-self.move, 0])
		landmarks_re[:, 0] -= np.max([self.left-self.move, 0])
		mask = np.zeros(self.patch_img.shape[:2], dtype=np.float64)
		self._draw_convex_hull(mask, landmarks_re, color=1)
		mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
		mask = (cv2.GaussianBlur(mask, ksize, 0) > 0) * 1.0
		return cv2.GaussianBlur(mask, ksize, 0)[:]

	# 获得全局绝对坐标遮盖
	def get_mask_abs(self, ksize=None):
		if ksize == None:
			ksize = self.ksize
		mask = np.zeros(self.img.shape, dtype=np.float64)
		patch = self.get_patch(mask)
		patch[:] = self.patch_mask[:]
		return mask

	# 美白
	def whitening(self, rate=0.15, confirm=True):
		if confirm:
			self.confirm()
			self.patch_hsv[:, :, -1] = np.minimum(self.patch_hsv[:, :, -1]+self.patch_hsv[:, :, -1]*self.patch_mask[:, :, -1]*rate, 255).astype('uint8')
			self.img[:]=cv2.cvtColor(self.img_hsv, cv2.COLOR_HSV2BGR)[:]
			self.update_temp()
		else:
			self.patch_hsv_temp[:] = cv2.cvtColor(self.patch_img_temp, cv2.COLOR_BGR2HSV)[:]
			self.patch_hsv_temp[:, :, -1] = np.minimum(self.patch_hsv_temp[:, :, -1]+self.patch_hsv_temp[:, :, -1]*self.patch_mask[:, :, -1]*rate, 255).astype('uint8')
			self.patch_img_temp[:] = cv2.cvtColor(self.patch_hsv_temp, cv2.COLOR_HSV2BGR)[:]

	# 提升鲜艳度
	def brightening(self, rate=0.3, confirm=True):
		patch_mask = self.get_mask_relative((1, 1))
		if confirm:
			self.confirm()
			patch_new = self.patch_hsv[:, :, 1]*patch_mask[:, :, 1]*rate
			patch_new = cv2.GaussianBlur(patch_new, (3, 3), 0)
			self.patch_hsv[:, :, 1] = np.minimum(self.patch_hsv[:, :, 1]+patch_new, 255).astype('uint8')
			self.img[:]=cv2.cvtColor(self.img_hsv, cv2.COLOR_HSV2BGR)[:]
			self.update_temp()
		else:
			self.patch_hsv_temp[:] = cv2.cvtColor(self.patch_img_temp, cv2.COLOR_BGR2HSV)[:]
			patch_new = self.patch_hsv_temp[:, :, 1]*patch_mask[:, :, 1]*rate
			patch_new = cv2.GaussianBlur(patch_new, (3, 3), 0)
			self.patch_hsv_temp[:, :, 1] = np.minimum(self.patch_hsv[:, :, 1]+patch_new, 255).astype('uint8')
			self.patch_img_temp[:] = cv2.cvtColor(self.patch_hsv_temp, cv2.COLOR_HSV2BGR)[:]

	# 磨皮
	def smooth(self, rate=0.6, ksize=None, confirm=True):
		if ksize == None:
			ksize=self.get_ksize(80)
		index = self.patch_mask > 0
		if confirm:
			self.confirm()
			patch_new = cv2.GaussianBlur(cv2.bilateralFilter(self.patch_img, 3, *ksize), ksize, 0)
			self.patch_img[index] = np.minimum(rate*patch_new[index]+(1-rate)*self.patch_img[index], 255).astype('uint8')
			self.img_hsv[:] = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)[:]
			self.update_temp()
		else:
			patch_new = cv2.GaussianBlur(cv2.bilateralFilter(self.patch_img_temp, 3, *ksize), ksize, 0)
			self.patch_img_temp[index] = np.minimum(rate*patch_new[index]+(1-rate)*self.patch_img_temp[index], 255).astype('uint8')
			self.patch_hsv_temp[:] = cv2.cvtColor(self.patch_img_temp, cv2.COLOR_BGR2HSV)[:]

	# 锐化
	def sharpen(self, rate=0.3, confirm=True):
		patch_mask = self.get_mask_relative((3, 3))
		kernel = np.zeros((9, 9), np.float32)
		kernel[4, 4] = 2.0
		boxFilter = np.ones((9, 9), np.float32) / 81.0
		kernel = kernel - boxFilter
		index = patch_mask > 0
		if confirm:
			self.confirm()
			sharp = cv2.filter2D(self.patch_img, -1, kernel)
			self.patch_img[index] = np.minimum(((1-rate)*self.patch_img)[index]+sharp[index]*rate, 255).astype('uint8')
			self.update_temp()
		else:
			sharp = cv2.filter2D(self.patch_img_temp, -1, kernel)
			self.patch_img_temp[:] = np.minimum(self.patch_img_temp+self.patch_mask*sharp*rate, 255).astype('uint8')
			self.patch_hsv_temp[:] = cv2.cvtColor(self.patch_img_temp, cv2.COLOR_BGR2HSV)[:]
            

# 额头
class ForeHead(Organ):
	def __init__(self, img, img_hsv, temp_img, temp_hsv, landmarks, mask_organs, name, ksize=None):
		self.mask_organs = mask_organs
		super(ForeHead, self).__init__(img, img_hsv, temp_img, temp_hsv, landmarks, name, ksize)

	# 获得局部mask
	def get_mask_relative(self, ksize=None):
		if ksize == None:
			ksize = self.ksize
		landmarks_re = self.landmarks.copy()
		landmarks_re[:, 1] -= np.max([self.top-self.move, 0])
		landmarks_re[:, 0] -= np.max([self.left-self.move, 0])
		mask = np.zeros(self.patch_img.shape[:2], dtype=np.float64)
		self._draw_convex_hull(mask, landmarks_re, color=1)
		mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
		mask = (cv2.GaussianBlur(mask, ksize, 0) > 0) * 1.0
		patch_organs = self.get_patch(self.mask_organs)
		mask= cv2.GaussianBlur(mask, ksize, 0)[:]
		mask[patch_organs>0] = (1-patch_organs[patch_organs>0])
		return mask


# 脸类
class Face(Organ):
	def __init__(self, img, img_hsv, temp_img, temp_hsv, landmarks, index):
		self.index = index
		self.organs_name = ['jaw', 'mouth', 'nose', 'left_eye', 'right_eye', 'left_brow', 'right_brow']	# 五官：下巴、嘴、鼻子、左右眼、左右耳
		# 五官标记点
		self.organs_point = [list(range(0, 17)), list(range(48, 61)), 
							 list(range(27, 35)), list(range(42, 48)), 
							 list(range(36, 42)), list(range(22, 27)),
							 list(range(17, 22))]
		self.organs = {name: Organ(img, img_hsv, temp_img, temp_hsv, landmarks[points], name) for name, points in zip(self.organs_name, self.organs_point)}
		# 额头
		mask_nose = self.organs['nose'].get_mask_abs()
		mask_organs = (self.organs['mouth'].get_mask_abs()+mask_nose+self.organs['left_eye'].get_mask_abs()+self.organs['right_eye'].get_mask_abs()+self.organs['left_brow'].get_mask_abs()+self.organs['right_brow'].get_mask_abs())
		forehead_landmark = self.get_forehead_landmark(img, landmarks, mask_organs, mask_nose)
		self.organs['forehead'] = ForeHead(img, img_hsv, temp_img, temp_hsv, forehead_landmark, mask_organs, 'forehead')
		mask_organs += self.organs['forehead'].get_mask_abs()
		# 人脸的完整标记点
		self.FACE_POINTS = np.concatenate([landmarks, forehead_landmark])
		super(Face, self).__init__(img, img_hsv, temp_img, temp_hsv, self.FACE_POINTS, 'face')
		mask_face = self.get_mask_abs() - mask_organs
		self.patch_mask = self.get_patch(mask_face)

	# 计算额头坐标
	def get_forehead_landmark(self, img, face_landmark, mask_organs, mask_nose):
		radius = (np.linalg.norm(face_landmark[0]-face_landmark[16])/2).astype('int32')
		center_abs = tuple(((face_landmark[0]+face_landmark[16])/2).astype('int32'))
		angle = np.degrees(np.arctan((lambda l:l[1]/l[0])(face_landmark[16]-face_landmark[0]))).astype('int32')
		mask = np.zeros(mask_organs.shape[:2], dtype=np.float64)
		cv2.ellipse(mask, center_abs, (radius, radius), angle, 180, 360, 1, -1)
		
		mask[mask_organs[:, :, 0]>0]=0
		
		index_bool = []
		for ch in range(3):
			mean, std = np.mean(img[:, :, ch][mask_nose[:, :, ch]>0]), np.std(img[:, :, ch][mask_nose[:, :, ch]>0])
			up, down = mean+0.5*std, mean-0.5*std
			index_bool.append((img[:, :, ch]<down)|(img[:, :, ch]>up))
		index_zero = ((mask>0)&index_bool[0]&index_bool[1]&index_bool[2])
		mask[index_zero] = 0
		index_abs = np.array(np.where(mask>0)[::-1]).transpose()
		landmark = cv2.convexHull(index_abs).squeeze()
		return landmark


# 化妆器
class Makeup():

	def __init__(self, predictor_path='./model/shape_predictor_68_face_landmarks.dat'):
		self.photo_path = []
		self.predictor_path = predictor_path
		self.faces = {}
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(self.predictor_path)

	# 使用dlib人脸定位
	def get_faces(self, img, img_hsv, temp_img, temp_hsv, name, n=1):
		rects = self.detector(img, 1)
		if len(rects) < 1:
			print('[Warning]:No face detected...')
			return None
		return {name: [Face(img, img_hsv, temp_img, temp_hsv, np.array([[p.x, p.y] for p in self.predictor(img, rect).parts()]), i) for i, rect in enumerate(rects)]}

	# 读取图片
	def read_img(self, fname, scale=1):
		img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), -1)
		if not type(img):
			print('[ERROR]:Fail to Read %s' % fname)
			return None
		return img

	# 标注关键点
	def read_and_mark(self, fname):
		img = self.read_img(fname)
		img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		temp_img, temp_hsv = img.copy(), img_hsv.copy()
		return img, temp_img, self.get_faces(img, img_hsv, temp_img, temp_hsv, fname)



if __name__ == '__main__':
	img_path = './data/test4.png'
	Mk = Makeup()
	img, temp_img, faces = Mk.read_and_mark(img_path)
	img_copy = img.copy()
	cv2.imshow('origin', img_copy)
	if faces:
		for face in faces[img_path]:
			face.whitening(0.3)
			face.smooth(0.4)
			face.organs['forehead'].whitening(0.4)
			face.organs['mouth'].whitening(0.4)
			face.organs['left_eye'].whitening(0.4)
			face.organs['right_eye'].whitening(0.4)
			face.organs['left_brow'].whitening(0.4)
			face.organs['right_brow'].whitening(0.4)
			face.organs['nose'].whitening(0.6)
			face.organs['mouth'].brightening(0.1)
			face.organs['forehead'].smooth(0.7)
			face.organs['mouth'].smooth(0.2)
			face.organs['right_eye'].smooth()
			face.organs['left_eye'].smooth()
			face.organs['nose'].smooth(0.3)
			face.organs['mouth'].smooth()
			face.organs['left_eye'].sharpen(0.2)
			face.organs['right_eye'].sharpen(0.2)
			face.organs['left_brow'].sharpen(0.2)
			face.organs['right_brow'].sharpen(0.2)
			face.organs['nose'].sharpen(0.3)
			face.sharpen(0.2)
			
		cv2.imshow('new', img.copy())
		cv2.waitKey()
	else:
		pass