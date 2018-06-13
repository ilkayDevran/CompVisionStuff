# USAGE
# python ppm2png.py --path aaa

# import the necessary packages
from PIL import Image
import argparse
import os

def main():

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True,
        help="path to the path images")
    args = vars(ap.parse_args())
    
    # Get the path by the user
    src = args["path"]
    src_files = os.listdir(src)

    #
    for i in range(len(src_files)): 
        if src_files[i].startswith('.') or src_files[i].endswith('.txt'):
            # pass unnecessary files
            pass
        else:
            sub_folder = os.listdir(src + '/' + src_files[i])
            for j in range(len(sub_folder)):
                imagePath = os.path.join(src + '/' + src_files[i], sub_folder[j])
                if imagePath.endswith(".ppm"):
                    im = Image.open(imagePath)
                    ext = imagePath.index('.')
                    newImagePath = imagePath[:ext]
                    os.remove(imagePath)
                    im.save(newImagePath + ".png")

if __name__ == '__main__':
    main()