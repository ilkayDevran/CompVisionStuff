
import os
import shutil
import csv 
import cv2 
 

def csvOrganizer(path, n):
	''' .csv organazing '''
	row = []
	with open(path, 'wb') as f:
		writer = csv.writer(f, delimiter=',')
		head = None
		mid = None 
		tail = None
		for i in range(n+1):
			if i < 10:
				head = '000' + str(i) + '.ppm'
				#string = '000' + str(i) + '.png,class:,#' + str(i)
				
			else:
				head = '00' + str(i) + '.ppm'
				#string = '00' + str(i) + '.png,class:,#' + str(i)
			mid = 'class:'
			tail = '#' + str(i)
			row = [head,mid,tail]
			writer.writerow(row)
		f.close()

def queriesFolderOrganizer():
	src = '/Users/ilkay/Desktop/imagesPPM/testing'
	dest = 'queries'

	src_files = os.listdir(src)
	
	for i in range(len(src_files)):
		if src_files[i].startswith('.') or src_files[i].endswith('.txt'):
			pass
		else:
			sub_folder = os.listdir(src + '/' + src_files[i])
			for j in range(len(sub_folder)):
				full_file_name = os.path.join(src + '/' + src_files[i], sub_folder[j])
				if sub_folder[j].endswith(".ppm"):
					s = src_files[i]
					d = None
					if s[3] == 0:
						d = s[4:]
					else:
						d = s[3:]
					copied_file_name = 'query' + d + '.' + str(j) + '.ppm'
					shutil.copy(full_file_name, os.path.join(dest, copied_file_name))
					#print copied_file_name

def cropedImagesFolderOrganizer():
	src = '/Users/ilkay/Desktop/training'
	dest = '/Users/ilkay/Desktop/ROI_groundTrurth'

	src_files = os.listdir(src)
	
	for i in range(len(src_files)):
		if src_files[i].startswith('.') or src_files[i].endswith('.txt'):
			continue
		#print src_files[i]

		sub_folder = os.listdir(src + '/' + src_files[i])
		csvFile = None
		# initialize the database dictionary of groundTruth
		db = {}

		# get the csv file in subfolder
		for file_name in sub_folder:
			if file_name.endswith('.csv'):
				csvFile = os.path.join(src + '/' + src_files[i], file_name)
				# loop over the database
				for l in csv.reader(open(csvFile), delimiter=';'):
				    # update the database using the image ID as the key
				    db[l[0]] = l[1:]

		# start cropping images according to coordinates in the csv files
		for j in range(len(sub_folder)):
			full_file_name = os.path.join(src + '/' + src_files[i], sub_folder[j])
			if sub_folder[j].endswith(".ppm"):
				(w, h, x1, y1, x2, y2, classId) = db[sub_folder[j]]
				img = cv2.imread(full_file_name)
				crop_img = img[int(x1):int(x2), int(y1):int(y2)]
				#cv2.imwrite("cropped_" + full_file_name, crop_img)
				#cv2.imshow("cropped", crop_img)
				#cv2.waitKey(0) 
				s = src_files[i]
				d = None
				if s[3] == 0:
					d = s[4:]
				else:
					d = s[3:]
				new_name = src_files[i] + str(j) + '.ppm'
				#copied_file_name = 'query' + d + '.' + str(j) + '.ppm'
				cv2.imwrite(os.path.join(dest + '/' + new_name), crop_img)
				#shutil.copy(full_file_name, os.path.join(dest, copied_file_name))

def LBP_ROI_Organizer():
	src = '/Users/ilkay/Desktop/git/CompVisionStuff/local-binary-patterns/ROI_images/training'
	dest = 'queries'

	src_files = os.listdir(src)
	
	for i in range(len(src_files)):
		if src_files[i].startswith('.') or src_files[i].endswith('.txt'):
			continue

		sub_folder = os.listdir(src + '/' + src_files[i])
		csvFile = None

		# initialize the database dictionary of groundTruth
		db = {}

		# get the csv file in subfolder
		for file_name in sub_folder:
			if file_name.endswith('.csv'):
				csvFile = os.path.join(src + '/' + src_files[i], file_name)
				# loop over the database
				for l in csv.reader(open(csvFile), delimiter=';'):
				    # update the database using the image ID as the key
				    db[l[0]] = l[1:]

		# start cropping images according to coordinates in the csv files
		for j in range(len(sub_folder)):
			full_file_name = os.path.join(src + '/' + src_files[i], sub_folder[j])
			if sub_folder[j].endswith(".ppm"):
				(w, h, x1, y1, x2, y2, classId) = db[sub_folder[j]]
				img = cv2.imread(full_file_name)
				crop_img = img[int(x1):int(x2), int(y1):int(y2)]
				#cv2.imwrite("cropped_" + full_file_name, crop_img)
				#cv2.imshow("cropped", crop_img)
				#cv2.waitKey(0) 
				s = src_files[i]
				d = None
				if s[3] == 0:
					d = s[4:]
				else:
					d = s[3:]
				copied_file_name = 'query' + d + '.' + str(j) + '.ppm'
				cv2.imwrite(os.path.join(src + '/' + src_files[i], copied_file_name), crop_img)
				os.remove(full_file_name)
				#shutil.copy(full_file_name, os.path.join(dest, copied_file_name))

def main():

	#csvOrganizer('database.csv',61)	# configure the csv file
	#queriesFolderOrganizer()
	cropedImagesFolderOrganizer()
	#LBP_ROI_Organizer()

if __name__ == '__main__':
    main()
