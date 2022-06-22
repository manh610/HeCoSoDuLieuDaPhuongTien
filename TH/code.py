import cv2
from pyspark import *
from pyspark.sql import *
import os
import numpy as np
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans
import sys
import shutil

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def extract_sift(image):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	sift_extractor = cv2.xfeatures2d.SIFT_create()
	key, des = sift_extractor.detectAndCompute(gray_image, None)
	print("feature of keypoint: ")
	print(des)
	print("length: ")
	print(len(des))
	return des

def save_extracted_features(input_images_path, output_des_path):
	image_names = os.listdir(input_images_path)
	count = 0
	for image_name in image_names:
		image_path = os.path.join(input_images_path, image_name) 
		print(image_path)
		image = cv2.imread (image_path)
		des = extract_sift(image)
		# save the descriptor to file
		output_file_path = os.path.join(output_des_path, image_name)
		np.savez_compressed(output_file_path, des) 
		count += 1
		print(f"image {count} extracted done!")


def create_session(memory, warehouse):
	return SparkSession.builder.appName("My app")			\
		.config('spark.driver.memory', memory)				\
		.config('spark.sql.warehouse.dir', warehouse)		\
		.config('spark.rdd.compress', True)					\
		.config('spark.driver.bindAddress',	"127.0.0.1")	\
		.getOrCreate()


def save_to_parquet(spark, files, batch_size, parquet_name): 
	for i in range(0, len(files), batch_size):
		start = i
		end = i + batch_size 
		print(f"Loading {start}..{end-1}")
		arrs = [np.load(x, allow_pickle=True) for x in files[1:50]] 
		dataset = map(lambda x: (Vectors.dense(x), ), [x for arr in arrs if arr['arr_0'].ndim == 2 for x in arr["arr_0"]])
		df = spark.createDataFrame(dataset, schema=["features"], samplingRatio=1)
		df.show()
		# print(dataset)
		df.write.format('parquet').mode('append').saveAsTable('temporary')
		df.unpersist()
		df.cache()
	# * Compact files
	warehouse = spark.conf.get('spark.sql.warehouse.dir', 'spark-warehouse')

	tmp_parquet = os.path.join(warehouse, 'temporary') 
	df = spark.read.parquet(tmp_parquet)
	df.write.format('parquet').mode('overwrite').saveAsTable(parquet_name)

	# # Clean-up
	# # xoa di casi template cu
	shutil.rmtree(tmp_parquet)

def convert_parquet(memory, warehouse, path_input, name_path_output):
	spark = create_session(memory, warehouse)
	listdir = os.listdir(path_input)
	listfile = [os.path.join(path_input, filename) for filename in listdir]
	# print(listfile)
	save_to_parquet(spark, listfile, 50, name_path_output)
	print("DONE!")


def cluster(dataframe, k, maxIter=50):
	clf = KMeans(k=k, maxIter=maxIter)
	print(type(dataframe))
	# print(list(dataframe))
	# dataframe.show()
	model = clf.fit(dataframe)
	return model.clusterCenters()

def cal_centers(memory, warehouse, name_path_parquet, path_centroidfile, ncluster): 
	spark = create_session(memory, warehouse) 
	df = spark.read.parquet(os.path.join(warehouse, name_path_parquet))

	print("Read done! Begin clustering")

	centers = cluster(df, ncluster, 100) 
	np.save(path_centroidfile, centers)
	print("Done")

# extract bow:

# You a month ago cm
# Chuyen tu cac 
from sklearn.neighbors import KNeighborsClassifier
class extract_bow:
	def __init__(self, centroids):
		self.centroids = centroids 
		self.knn = KNeighborsClassifier(n_neighbors=1, weights="distance")
		self.knn.fit(centroids, range(len(centroids)))

	def extract(self, img):
		des = extract_sift(img)
		try:
			index = []
			for i, arr in enumerate(des): 
				# print("check_1", np.isnan(arr))
				# kiem tra xem vecter 128 co gia tri NAN ko neu co index++
				if np.any(np.isnan(arr)):						
					index.append(i)
					print("Vao")
			# Xoa nhung vecter NAN
			des = np.delete(des, index, axis = 0)
			# cai nay la mang 2 roi len ko can chuyen vao []
			# tim k lang gieng tham so truyen la tap du doan va no tra ve 1 array
			pred = self.knn.predict(des)
		except:
			# Neu gap exception tuc ko anh ko ton tai thi gan des la mang 2 chieu va gai tri cua no la 128 chieu
			length = 128
			des = np.zeros((1, length)) 
		pred = self.knn.predict(des)
		# Khoi tao vecter sau khi chuan hoa so chieu = so cum. mang 1 chieu co kich thuoc = so cum
		arr_count = np.zeros(len(self.centroids))
		# Lay tung gia tri trong tap ket qua phan doan tang gia tri tai cot do len
		for x in pred:
			arr_count[x] += 1
		# print(pred)
		# chia cac gia tri cho so luong cac diem do
		return arr_count / len(des)


import random
def split_data(dir_path, ratio):
	#constuct array of image paths [('path', 'Buom'),...]
	image_paths = []
	category_names = os.listdir(dir_path)
	# print(category_names);
	for category_name in category_names:
		image_names = os.listdir(os.path.join(dir_path, category_name))
		for image_name in image_names: 
			image_path = os.path.join(dir_path, category_name, image_name)
			image_paths.append((image_path, category_name))
	#split image into trainset and testset
	random.shuffle(image_paths)
	# print(len(image_paths), "\n\n\n")
	partition = int(len(image_paths)*ratio)
	# print("sasass: ", partition)
	train_set = image_paths[:partition]
	test_set = image_paths[partition:] 
	# print(len(train_set), "\n\n\n")
	return (image_paths, test_set)


def extract_encode_image(img_path, centroids): 
	extract_encode = extract_bow(centroids)
	print(img_path)
	img = cv2.imread(img_path)
	img_encode = extract_encode.extract(img)
	return img_encode

# chuan hoa cac mau thanh vecter index cua no la path
def cal_encodes_tranningset(train_set, centroids):
	encode_label_of_images = {} 
	print("1.", len(train_set))
	if not os.path.exists("file_encode_label.npy"):
		num = 0

		for (img_path, catagory) in train_set:
			img_encode = extract_encode_image(img_path, centroids) 
			encode_label_of_images[img_path] = (img_encode, catagory)
			num += 1
			print("encoding image:", num, "completed") 
		np.save('file_encode_label.npy', encode_label_of_images)
	else: 
		encode_label_of_images = np.load('file_encode_label.npy',encoding='latin1', allow_pickle=True).item() 
	# print("2.", len(encode_label_of_images), " -- ", num)
	return encode_label_of_images


def construct_model_knn(encode_label_of_images): 
	print("Begin training")
	clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
	# encode_label_of_images[img_path] = (vecter, category)
	# mang vecter
	trainX = [encode_label_of_images[path_image][0] for path_image in encode_label_of_images.keys()] 
	# mang category
	trainY = [encode_label_of_images[path_image][1] for path_image in encode_label_of_images.keys()]
	# print(trainX, len(trainX), "-", len(encode_label_of_images))
	clf.fit(trainX, trainY)
	print("Training completed")
	return clf


def find_similar_images (img_path, clf, centroids): 
	image_encode = extract_encode_image(img_path, centroids) 
	print("feature of image: ")
	print(image_encode)
	pre = clf.predict([image_encode])
	print(img_path, "- Predict: ", pre)
	(distant, nearest_img_index) = clf.kneighbors([image_encode])
	# print("--------", distant)
	return (nearest_img_index, pre)


import matplotlib.pyplot as plt
def most_similar_image_demo_display(img_path, dir_path, path_centroids, ratio=1):
	print('begin:')
	centroids = np.load(path_centroids)
	(train_set, test_set) = split_data(dir_path, ratio) 
	encode_label_of_images = cal_encodes_tranningset(train_set, centroids)
	clf = construct_model_knn(encode_label_of_images) 
	nearest_img_index, pre = find_similar_images(img_path, clf, centroids)
	# print(nearest_img_index, "aaaaa")
	nearest_img_index = nearest_img_index[0] 
	#  lay ra tat ca path trong encode_label_of_images chi lay nhung vi tri index can lay
	nearest_img_paths = [list(encode_label_of_images.keys())[index] for index in nearest_img_index] 
	print("Similar image: Path -", nearest_img_paths)
	print("Real Catagory: ", [encode_label_of_images[nearest_img_path][1] for nearest_img_path in nearest_img_paths])
# display result

	fig = plt.figure()
	imgl = fig.add_subplot(1,2,1)
	imgplot = plt.imshow(cv2.imread(img_path)) 
	imgl.set_title("Input Image")
	imgl = fig.add_subplot(1,2,2)
	imgplot = plt.imshow(cv2.imread(nearest_img_paths[0])) 
	imgl.set_title("Host similar image: " + encode_label_of_images[nearest_img_paths[0]][1])
	plt.show()


