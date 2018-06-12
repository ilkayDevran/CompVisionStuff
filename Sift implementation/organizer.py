
import os
import shutil
import csv  
 

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
	src = '/Users/ilkay/Desktop/Testing'
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

def main():

	csvOrganizer('database.csv',61)	# configure the csv file
	#queriesFolderOrganizer()

if __name__ == '__main__':
    main()
