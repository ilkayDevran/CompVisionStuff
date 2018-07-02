# USAGE
# python pca.py -t ../ROI_images/training  
# python pca.py -t ../ROI_images/training -m m || -m s
# https://sebastianraschka.com/Articles/2014_pca_step_by_step.html

# import the necessary packages
from localbinarypatterns import LocalBinaryPatterns
from matplotlib import pyplot as plt
from imutils import paths
import numpy as np
import argparse
import cv2


def compute_manually(all_samples, labels, samples_amount_of_classes):

    histLength, sampleLength = all_samples.shape

    # Computing the d-dimensional mean vector
    mean_vector = []
    for i in all_samples:
        mean = np.mean(i)
        mean_vector.append([mean])

    mean_vector = np.array(mean_vector)
    #print mean_vector

    # Computing eigenvectors and corresponding eigenvalues
    eig_val_sc, eig_vec_sc , eig_val_cov, eig_vec_cov= None, None, None, None


    # Computing the Scatter Matrix
    scatter_matrix = np.zeros((histLength,histLength))
    for i in range(all_samples.shape[1]):
        scatter_matrix += (all_samples[:,i].reshape(histLength,1) - mean_vector).dot((all_samples[:,i].reshape(histLength,1) - mean_vector).T)
    eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)

    # Computing the Covariance Matrix (alternatively to the scatter matrix)
    tmpList = []
    for i in all_samples:
        tmpList.append(i)
    cov_mat = np.cov(tmpList)
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

    for i in range(len(eig_val_sc)):
        eigvec_sc = eig_vec_sc[:,i].reshape(1,histLength).T
        eigvec_cov = eig_vec_cov[:,i].reshape(1,histLength).T
        assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'
        """
        print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))
        print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))
        print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
        print('Scaling factor: ', eig_val_sc[i]/eig_val_cov[i])
        print(40 * '-')
        """
    for i in range(len(eig_val_sc)):
        eigv = eig_vec_sc[:,i].reshape(1,histLength).T
        np.testing.assert_array_almost_equal(scatter_matrix.dot(eigv), eig_val_sc[i] * eigv,
            decimal=6, err_msg='', verbose=True)

    for ev in eig_vec_sc:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    """for i in eig_pairs:
        print(i[0])
    """

    # Choosing k eigenvectors with the largest eigenvalues
    matrix_w = np.hstack((eig_pairs[0][1].reshape(histLength,1), eig_pairs[1][1].reshape(histLength,1)))
    #print('Matrix W:\n', matrix_w)

    # Transforming the samples onto the new subspace
    transformed = matrix_w.T.dot(all_samples)
    assert transformed.shape == (2,sampleLength), "The matrix is not 2x4527 dimensional."
    max_Y = max(transformed[1])
    max_X = max(transformed[0])
    min_Y = min(transformed[1])
    min_X = min(transformed[0])
    print max_X ,max_Y, min_X, min_Y
    raw_input()
    tmp= []
    for i in range(len(transformed[0])):
        if 0.4 - transformed[0, i] < 0 and 0.6 - transformed[0, i] > 0:
            print transformed[1,i]
            tmp.append(transformed[1,i])
    tmp = np.array(tmp)
    tmp.sort()
    print len(tmp)
    
    temp = 0
    for i,val in enumerate(samples_amount_of_classes):
        #print i, val
        plt.plot(transformed[0,temp:val], transformed[1,temp:val], 'o', markersize=7, color=np.random.rand(3,), alpha=0.5, label=labels[i])
        temp = val
    plt.xlim([min_X,max_X])
    plt.ylim([min_Y, max_Y])
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples with class labels')

    plt.show()



def use_matplotlib(all_samples, labels, samples_amount_of_classes):
    print "mat"
    from matplotlib.mlab import PCA as mlabPCA

    mlab_pca = mlabPCA(all_samples.T)
    #print('PC axes in terms of the measurement axes scaled by the standard deviations:\n', mlab_pca.Wt)
    max_Y = max(mlab_pca.Y[:,1])
    max_X = max(mlab_pca.Y[:,0])
    min_Y = min(mlab_pca.Y[:,1])
    min_X = min(mlab_pca.Y[:,0])
    #print max_Y, max_X, min_Y, min_X
    #raw_input()
    temp = 0
    for i,val in enumerate(samples_amount_of_classes):
        plt.plot(mlab_pca.Y[temp:val,0], mlab_pca.Y[temp:val,1], 'o', markersize=7, color=np.random.rand(3,), alpha=0.5)
        temp = val + 1
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.xlim([min_X,max_X])
    plt.ylim([min_Y, max_Y])
    plt.legend()
    plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

    plt.show()



def use_sklearn(all_samples, labels, samples_amount_of_classes):
    from sklearn.decomposition import PCA as sklearnPCA

    sklearn_pca = sklearnPCA(n_components=2)
    sklearn_transf = sklearn_pca.fit_transform(all_samples.T)
    sklearn_transf = sklearn_transf.T

    max_Y = max(sklearn_transf[1])
    max_X = max(sklearn_transf[0])
    min_Y = min(sklearn_transf[1])
    min_X = min(sklearn_transf[0])
    
    tmp= []
    for i in range(len(sklearn_transf[0])):
        if 0.4 - sklearn_transf[0,i] < 0 and sklearn_transf[0,i] - 0.6 < 0:
            print sklearn_transf[1,i]
            tmp.append(sklearn_transf[1,i])
    tmp = np.array(tmp)
    tmp.sort()
    print len(tmp)

    temp = 0
    for i,val in enumerate(samples_amount_of_classes):
        plt.plot(sklearn_transf[0,temp:val],sklearn_transf[1,temp:val], 'o', markersize=7, color=np.random.rand(3,), alpha=0.5)
        temp = val

    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.xlim([min_X,max_X])
    plt.ylim([min_Y, max_Y])
    plt.legend()
    plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

    plt.show()



def get_all_samples(path):
    # initialize the local binary patterns descriptor along with the data and label lists
    desc = LocalBinaryPatterns(24, 8)
    data = []
    labels = []
    classSamplesList = []
    samples_amount_of_classes = []
    currentClass = None
    flag = False

    # loop over the training images
    for imagePath in paths.list_files(path, validExts=(".png",".ppm")):
        if (flag == False):
            currentClass = imagePath.split("/")[-2]
            labels.append(currentClass)
            flag = True
        else:
            if imagePath.split("/")[-2] != currentClass:
                currentClass = imagePath.split("/")[-2]
                classSamplesList.append(np.transpose(np.array(data)))
                samples_amount_of_classes.append(len(data))
                data = []
                labels.append(currentClass)

        # load the image, convert it to grayscale, and describe it
        image = cv2.imread(imagePath)
        gray = np.matrix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        resized_image = cv2.resize(gray, (32, 32))
        hist = desc.describe(resized_image)
        hist = hist / max(hist)
        
        # extract the label from the image path
        data.append(hist)
    
    all_samples =  tuple(classSamplesList)
    all_samples = np.concatenate(all_samples, axis=1)

    return all_samples, labels ,samples_amount_of_classes


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training", required=True,
        help="path to the training images")
    ap.add_argument("-m", "--Mod", type = str, default = 'man',
        help="Calculate Manually")
    ap.add_argument("-mat", "--matplot", type = str, default = '',
        help="Calculate with Matplotlib")
    ap.add_argument("-s", "--sklearn", type = str, default = '',
        help="Calculate with Sklearn")
    args = vars(ap.parse_args())

    # get all_samples list
    all_samples, labels, samples_amount_of_classes = get_all_samples(args["training"])

    # choose calculation mod 
    calculationMod = args["Mod"]
    if calculationMod == 's' or calculationMod == 'sklearn':
        use_sklearn(all_samples, labels, samples_amount_of_classes)
    elif calculationMod == 'm' or calculationMod == 'mat' or calculationMod == 'matplot':
        use_matplotlib(all_samples, labels, samples_amount_of_classes)
    else:
        compute_manually(all_samples, labels, samples_amount_of_classes)