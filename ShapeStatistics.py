#Purpose: To implement a suite of 3D shape statistics and to use them for point
#cloud classification
#TODO: Fill in all of this code for group assignment 2
import sys
sys.path.append("S3DGLPy")
from Primitives3D import *
from PolyMesh import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
#from sklearn.metrics.pairwise import chi2_kernel

POINTCLOUD_CLASSES = ['biplane', 'desk_chair', 'dining_chair', 'fighter_jet', 'fish', 'flying_bird', 'guitar', 'handgun', 'head', 'helicopter', 'human', 'human_arms_out', 'potted_plant', 'race_car', 'sedan', 'shelves', 'ship', 'sword', 'table', 'vase']
NUM_PER_CLASS = 10

#########################################################
##                UTILITY FUNCTIONS                    ##
#########################################################

#Purpose: Export a sampled point cloud into the JS interactive point cloud viewer
#Inputs: Ps (3 x N array of points), Ns (3 x N array of estimated normals),
#filename: Output filename
def exportPointCloud(Ps, Ns, filename):
    N = Ps.shape[1]
    fout = open(filename, "w")
    fmtstr = "%g" + " %g"*5 + "\n"
    for i in range(N):
        fields = np.zeros(6)
        fields[0:3] = Ps[:, i]
        fields[3:] = Ns[:, i]
        fout.write(fmtstr%tuple(fields.flatten().tolist()))
    fout.close()

#Purpose: To sample a point cloud, center it on its centroid, and
#then scale all of the points so that the RMS distance to the origin is 1
def samplePointCloud(mesh, N):
    (Ps, Ns) = mesh.randomlySamplePoints(N)
    ##TODO: Center the point cloud on its centroid and normalize
    #by its root mean square distance to the origin.  Note that this
    #does not change the normals at all, only the points, since it's a
    #uniform scale

    # Center the randomly distributed point cloud on its centroid.
    c  = np.asmatrix([list(np.mean(Ps, axis=1))]).T
    Ps = Ps - c
    # Calculate scale
    squares = list(np.einsum("ji,ji->i", Ps, Ps))
    sums    = np.sum(squares)
    scale   = math.sqrt(sums/len(squares))
    # Apply Scale
    Ps      = Ps / scale

    return (Ps, Ns)

#Purpose: To sample the unit sphere as evenly as possible.  The higher
#res is, the more samples are taken on the sphere (in an exponential
#relationship with res).  By default, samples 66 points
def getSphereSamples(res = 2):
    m = getSphereMesh(1, res)
    return m.VPos.T

#Purpose: To compute PCA on a point cloud
#Inputs: X (3 x N array representing a point cloud)
def doPCA(X):
    ##TODO: Fill this in for a useful helper function

    # Compute covariance matrix A = X.T * X
    A = X.dot(X.T)
    # Compute eigenvalues/eigenvectors of A, sorted in decreasing order

    eigenValues, eigenVectors = np.linalg.eig(A)

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]

    return (eigenValues, eigenVectors)

#########################################################
##                SHAPE DESCRIPTORS                    ##
#########################################################

#Purpose: To compute a shape histogram, counting points
#distributed in concentric spherical shells centered at the origin
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency)
#NShells (number of shells), RMax (maximum radius)
#Returns: hist (histogram of length NShells)
def getShapeHistogram(Ps, Ns, NShells, RMax):
    hist = np.zeros(NShells)                   #[0,0,0,0..NShells]
    bins = np.linspace(0, RMax, num = NShells+1) # np.linspace(2.0, 3.0, num=5)  // array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
    # Magnitudes / Euclidean Distance
    mags = np.sqrt(np.einsum("ji,ji->i", Ps, Ps))
    # Put em in the buckets / Return
    return np.histogram(mags, bins=bins, density=True)[0]

#Purpose: To create shape histogram with concentric spherical shells and
#sectors within each shell, sorted in decreasing order of number of points
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency), NShells (number of shells),
#RMax (maximum radius), SPoints: A 3 x S array of points sampled evenly on
#the unit sphere (get these with the function "getSphereSamples")
def between(bin1,bin2,pt):
    print pt.shape
    mag = np.linalg.norm(pt)
    print "norm",mag
    if  mag >= bin1 and mag < bin2:
        return True
    return False
def getShapeShellHistogram(Ps, Ns, NShells, RMax, SPoints):
    NSectors = SPoints.shape[1] #A number of sectors equal to the number of
    #points sampled on the sphere
    #Create a 2D histogram that is NShells x NSectors
    hist = np.zeros((NShells, NSectors))
    bins = np.linspace(0, RMax, num = NShells*NSectors)
    ##TODO: Finish this; fill in hist, then sort sectors in descending order

    # Find which ring its in
    mags = np.sqrt(np.einsum("ji,ji->i", Ps, Ps))

    #XSub = X[:, X[0, :] > 0] #Select all points with first coordinate greater than zero
    for i in xrange(len(bins)-1):
        ptsInSector = Ps[:, Ps[0, :] > 0]
        #ptsInSector = Ps[:, math.sqrt(Ps[0, :]**2 + Ps[0, :]**2 + Ps[0, :]**2) >= bins[i] and (math.sqrt(Ps[0, :]**2 + Ps[0, :]**2 + Ps[0, :]**2)) < bins[i+1]]
        #print Ps
        #print ptsInSector.shape
    # Find its sectors
    simil = Ps.T.dot(SPoints) # Each point is in the sector it is most similar to
    #np.histogram(mags, bins=bins, density=True)[0]
    return hist.flatten() #Flatten the 2D histogram to a 1D array

#Purpose: To create shape histogram with concentric spherical shells and to
#compute the PCA eigenvalues in each shell
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency), NShells (number of shells),
#RMax (maximum radius), sphereRes: An integer specifying points on the sphere
#to be used to cluster shells
def getShapeHistogramPCA(Ps, Ns, NShells, RMax):
    #Create a 2D histogram, with 3 eigenvalues for each shell
    hist = np.zeros((NShells, 3))
    ##TODO: Finish this; fill in hist
    bins = np.linspace(0, RMax, num = (NShells*3)+1) # np.linspace(2.0, 3.0, num=5)  // array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])
    # Magnitudes / Euclidean Distance
    mags = np.sqrt(np.einsum("ji,ji->i", Ps, Ps))
    numpy.digitize(mags, bins) # index of bin that mags is in

    PCA = doPCA(Ps)
    # Put em in the buckets / Return
    return np.histogram(mags, bins=bins, density=True)[0].flatten()


    #return hist #Flatten the 2D histogram to a 1D array

#Purpose: To create shape histogram of the pairwise Euclidean distances between
#randomly sampled points in the point cloud
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency), DMax (Maximum distance to consider),
#NBins (number of histogram bins), NSamples (number of pairs of points sample
#to compute distances)
def getD2Histogram(Ps, Ns, DMax, NBins, NSamples):
    ##TODO: Finish this; fill in hist
    bins = np.linspace(0, DMax, num = NBins+1) # np.linspace(2.0, 3.0, num=5)  // array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])

    rand = np.random.random_integers(Ps.shape[1]-1, size=(NSamples,2))

    r0 = rand[:,0]
    r1 = rand[:,1]

    Ps0 = Ps[:,r0] #np.take(Ps.T,r0)
    Ps1 = Ps[:,r1] #np.take(Ps.T,r1)

    vecs = Ps0 - Ps1
    mags = np.sqrt(np.einsum("ji,ji->i", vecs, vecs))

    hist = np.histogram(mags, bins=bins, density=True)[0] #np.histogram(mags, bins=bins)[0]
    return hist
#Purpose: To create shape histogram of the angles between randomly sampled
#triples of points
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals) (not needed here
#but passed along for consistency), NBins (number of histogram bins),
#NSamples (number of triples of points sample to compute angles)
def getA3Histogram(Ps, Ns, NBins, NSamples):
    hist = np.zeros(NBins)
    ##TODO: Finish this; fill in hist
    bins = np.linspace(0, 3.141596, num = NBins+1)
    rand = np.random.random_integers(Ps.shape[1]-1, size=(NSamples,3))

    r0 = rand[:,0]
    r1 = rand[:,1]
    r2 = rand[:,2]

    Ps0 = Ps[:,r0] #np.take(Ps.T,r0)
    Ps1 = Ps[:,r1] #np.take(Ps.T,r1)
    Ps2 = Ps[:,r2] #np.take(Ps.T,r1)

    #Get appropriate vectos Ps0 -> Ps1 and Ps1 -> Ps2
    vec01 = Ps1 - Ps0
    vec12 = Ps2 - Ps1

    # Normalized Vectors
    norms01 = np.linalg.norm(vec01,axis=0)
    norms12 = np.linalg.norm(vec12, axis=0)

    vec01 = vec01.astype('float') / norms01
    vec12 = vec12.astype('float') / norms12

    ### theta = a dot b / ||a||||b|| ###

    # Dot product
    num = np.einsum("ji,ji->i", vec01, vec12) # 1xN array

    # Magnitudes of vectors
    mag01 = np.sqrt(np.einsum("ji,ji->i", vec01, vec01))
    mag12 = np.sqrt(np.einsum("ji,ji->i", vec12, vec12))

    den = mag01 * mag12 #1xN

    CosThetas = num / den
    ArcCosT = np.arccos(CosThetas)
    return np.histogram(ArcCosT, bins=bins, density=True)[0]

#Purpose: To create the Extended Gaussian Image by binning normals to
#sphere directions after rotating the point cloud to align with its principal axes
#Inputs: Ps (3 x N point cloud) (use to compute PCA), Ns (3 x N array of normals),
#SPoints: A 3 x S array of points sampled evenly on the unit sphere used to
#bin the normals
def getEGIHistogram(Ps, Ns, SPoints):
    S = SPoints.shape[1]
    hist = np.zeros(S)
    # TODO: Finish this; fill in hist
    # Project all points on PCA Axis
    eigVal, eigVec = doPCA(Ps)
    axis = eigVec[:,-1]

    projPS = ((Ps.T.dot(axis)) / axis.T.dot(axis)) * axis
    return hist

#Purpose: To create an image which stores the amalgamation of rotating
#a bunch of planes around the largest principal axis of a point cloud and
#projecting the points on the minor axes onto the image.
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals, not needed here),
#NAngles: The number of angles between 0 and 2*pi through which to rotate
#the plane, Extent: The extent of each axis, Dim: The number of pixels along
#each minor axis
def getSpinImage(Ps, Ns, NAngles, Extent, Dim):
    #Create an image
    hist = np.zeros((Dim, Dim))
    # TODO: Finish this
    # Project all points on PCA Axis
    bins = np.linspace(0, Extent, num = Dim+1)
    eigVal, eigVec = doPCA(Ps)
    pAxis  = eigVec[:,0]
    mAxis1 = eigVec[:,1]
    mAxis2 = eigVec[:,2]
    projPs = np.asarray((Ps.T.dot(pAxis)) / pAxis.T.dot(pAxis))[:,0]

    angle = 2 * math.pi/ NAngles
    for i in xrange(NAngles):
        ang = angle * i
        vec = mAxis1 * math.cos(ang) + mAxis2 * math.sin(ang)
        # plane is now defined by pAxis and vec
        projVec = np.asarray((Ps.T.dot(vec)) / vec.T.dot(vec))[:,0]
        tmpHist = np.histogram2d(projVec,projPs,bins=(bins,bins),normed=True)[0]
        hist += tmpHist / NAngles

    return hist.flatten()


def getSpinImageFast(Ps, Ns, NAngles, Extent, Dim):
    #Create an image
    hist = np.zeros((Dim, Dim))
    # TODO: Finish this
    # Project all points on PCA Axis
    bins = np.linspace(0, Extent, num = Dim+1)
    eigVal, eigVec = doPCA(Ps)
    pAxis  = eigVec[:,0]
    projPs = np.asarray((Ps.T.dot(pAxis)) / pAxis.T.dot(pAxis))[:,0]

    perpProj = Ps - pAxis.dot(pAxis.T).dot(Ps)
    mags     = np.sqrt(np.einsum("ji,ji->i", perpProj, perpProj))

    return np.histogram2d(mags,projPs,bins=(bins,bins),normed=True)[0].flatten()


#Purpose: To create a histogram of spherical harmonic magnitudes in concentric
#spheres after rasterizing the point cloud to a voxel grid
#Inputs: Ps (3 x N point cloud), Ns (3 x N array of normals, not used here),
#VoxelRes: The number of voxels along each axis (for instance, if 30, then rasterize
#to 30x30x30 voxels), Extent: The number of units along each axis (if 2, then
#rasterize in the box [-1, 1] x [-1, 1] x [-1, 1]), NHarmonics: The number of spherical
#harmonics, NSpheres, the number of concentric spheres to take
def getSphericalHarmonicMagnitudes(Ps, Ns, VoxelRes, Extent, NHarmonics, NSpheres):
    hist = np.zeros((NSpheres, NHarmonics))
    #TODO: Finish this

    return hist.flatten()

#Purpose: Utility function for wrapping around the statistics functions.
#Inputs: PointClouds (a python list of N point clouds), Normals (a python
#list of the N corresponding normals), histFunction (a function
#handle for one of the above functions), *args (addditional arguments
#that the descriptor function needs)
#Returns: AllHists (A KxN matrix of all descriptors, where K is the length
#of each descriptor)
def makeAllHistograms(PointClouds, Normals, histFunction, *args):
    N = len(PointClouds)
    #Call on first mesh to figure out the dimensions of the histogram
    h0 = histFunction(PointClouds[0], Normals[0], *args)
    K = h0.size
    AllHists = np.zeros((K, N))
    AllHists[:, 0] = h0
    for i in range(1, N):
        print "Computing histogram %i of %i..."%(i+1, N)
        AllHists[:, i] = histFunction(PointClouds[i], Normals[i], *args)
    return AllHists

#########################################################
##              HISTOGRAM COMPARISONS                  ##
#########################################################

#Purpose: To compute the euclidean distance between a set
#of histograms
#Inputs: AllHists (K x N matrix of histograms, where K is the length
#of each histogram and N is the number of point clouds)
#Returns: D (An N x N matrix, where the ij entry is the Euclidean
#distance between the histogram for point cloud i and point cloud j)
def compareHistsEuclidean(AllHists):
    N = AllHists.shape[1]
    D = np.zeros((N, N))
    #TODO: Finish this, fill in D
    dotX = np.sum(AllHists**2, 0)[:, None]
    dotY = np.sum(AllHists**2, 0)[None, :]
    D = dotX + dotY - 2*AllHists.T.dot(AllHists)

    return np.sqrt(np.around(D, decimals=10))

#Purpose: To compute the cosine distance between a set
#of histograms
#Inputs: AllHists (K x N matrix of histograms, where K is the length
#of each histogram and N is the number of point clouds)
#Returns: D (An N x N matrix, where the ij entry is the cosine
#distance between the histogram for point cloud i and point cloud j)
def compareHistsCosine(AllHists):
    N = AllHists.shape[1]
    D = np.zeros((N, N))
    #TODO: Finish this, fill in D
    num = AllHists.T.dot(AllHists)
    mag = np.asmatrix([list(np.sqrt(np.einsum("ji,ji->i", AllHists, AllHists)))])
    den = mag.T.dot(mag)

    return np.arccos(num/den)

#Purpose: To compute the chi squared distance between a set
#of histograms
#Inputs: AllHists (K x N matrix of histograms, where K is the length
#of each histogram and N is the number of point clouds)
#Returns: D (An N x N matrix, where the ij entry is the chi squared
#distance between the histogram for point cloud i and point cloud j)
def compareHistsChiSquared(AllHists):
    N = AllHists.shape[1]
    D = np.zeros((N, N))
    #TODO: Finish this, fill in D
    histSums = np.sum(AllHists, 0)[:, None] # creates 1 column of vectors
    res = histSums[np.newaxis, :] - histSums[:, np.newaxis]
    return D

#Purpose: To compute the 1D Earth mover's distance between a set
#of histograms (note that this only makes sense for 1D histograms)
#Inputs: AllHists (K x N matrix of histograms, where K is the length
#of each histogram and N is the number of point clouds)
#Returns: D (An N x N matrix, where the ij entry is the earth mover's
#distance between the histogram for point cloud i and point cloud j)
def compareHistsEMD1D(AllHists):
    N = AllHists.shape[1]
    D = np.zeros((N, N))
    #TODO: Finish this, fill in D
    cdfs = np.sum(AllHists, axis=0)

    return D


#########################################################
##              CLASSIFICATION CONTEST                 ##
#########################################################

#Purpose: To implement your own custom distance matrix between all point
#clouds for the point cloud clasification contest
#Inputs: PointClouds, an array of point cloud matrices, Normals: an array
#of normal matrices
#Returns: D: A N x N matrix of distances between point clouds based
#on your metric, where Dij is the distnace between point cloud i and point cloud j
def getMyShapeDistances(PointClouds, Normals):
    #TODO: Finish this
    #This is just an example, but you should experiment to find which features
    #work the best, and possibly come up with a weighted combination of
    #different features
    HistsD2 = makeAllHistograms(PointClouds, Normals, getD2Histogram, 3.0, 30, 100000)
    DEuc = compareHistsEuclidean(HistsD2)
    return DEuc

#########################################################
##                     EVALUATION                      ##
#########################################################

#Purpose: To return an average precision recall graph for a collection of
#shapes given the similarity scores of all pairs of histograms.
#Inputs: D (An N x N matrix, where the ij entry is the earth mover's distance
#between the histogram for point cloud i and point cloud j).  It is assumed
#that the point clouds are presented in contiguous chunks of classes, and that
#there are "NPerClass" point clouds per each class (for the dataset provided
#there are 10 per class so that's the default argument).  So the program should
#return a precision recall graph that has 9 elements
#Returns PR, an (NPerClass-1) length array of average precision values for all
#recalls
def getPrecisionRecall(D, NPerClass = 10):
    PR = np.zeros(NPerClass-1) # [0,0,0...]
    #TODO: Finish this, compute average precision recall graph using all point clouds as queries
    # This is the average precision recall for every shape not just 1 shape
    # Sort rows of D
    s = np.argsort(D,axis=1)
    # Walk through
    for i in xrange(s.shape[1]): # On every row we calculate the precision and recall
        iClass = i/NPerClass # the row indicates the shape
        numP = 0
        denP = 0
        numR = 0
        for j in xrange(s.shape[1]):
            jClass = s[i, j]/NPerClass
            if i != s[i, j]:
                if iClass == jClass:

                    numR = numR + 1
                    numP = numP + 1
                    denP = denP + 1

                    PR[numR-1] += (1.0*numP/denP) * (1.0/D.shape[1])

                    if numR == NPerClass - 1:
                        break
                else:
                    denP += 1

    return PR

#########################################################
##                     MAIN TESTS                      ##
#########################################################

if __name__ == '__main__':
    NRandSamples = 10000 #You can tweak this number
    np.random.seed(100) #For repeatable results randomly sampling
    #Load in and sample all meshes
    PointClouds = []
    Normals = []
    for i in range(len(POINTCLOUD_CLASSES)):

        print "LOADING CLASS %i of %i..."%(i, len(POINTCLOUD_CLASSES))
        PCClass = []
        for j in range(NUM_PER_CLASS):
            m = PolyMesh()
            filename = "models_off/%s%i.off"%(POINTCLOUD_CLASSES[i], j)
            print "Loading ", filename
            m.loadOffFileExternal(filename)
            (Ps, Ns) = samplePointCloud(m, NRandSamples)
            PointClouds.append(Ps)
            Normals.append(Ps)


    #TODO: Finish this, run experiments.  Also in the above code, you might
    #just want to load one point cloud and test your histograms on that first
    #so you don't have to wait for all point clouds to load when making
    #minor tweaks
    #HistsShape1    = makeAllHistograms(PointClouds, Normals, getShapeShellHistogram, 1, 5, getSphereSamples())
    HistsSpin = makeAllHistograms(PointClouds, Normals, getSpinImage,100, 2, 40)
    HistsSpinFast = makeAllHistograms(PointClouds, Normals, getSpinImageFast,100, 2, 40)

    DSpin = compareHistsEuclidean(HistsSpin)
    DSpinF = compareHistsEuclidean(HistsSpinFast)
    PRSpin = getPrecisionRecall(DSpin)
    PRSpinF = getPrecisionRecall(DSpinF)

    recalls = np.linspace(1.0/9.0, 1.0, 9)
    plt.hold(True)
    plt.plot(recalls, PRSpin, 'b', label='Spin')
    plt.plot(recalls, PRSpinF, 'g', label='Spin Fast')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()
    '''HistsShape1    = makeAllHistograms(PointClouds, Normals, getShapeHistogram, 1, 5)
    HistsShape10   = makeAllHistograms(PointClouds, Normals, getShapeHistogram, 10, 5)
    HistsShape20   = makeAllHistograms(PointClouds, Normals, getShapeHistogram, 20, 5)
    HistsShape30   = makeAllHistograms(PointClouds, Normals, getShapeHistogram, 30, 5)

    HistsD2100     = makeAllHistograms(PointClouds, Normals, getD2Histogram, 3.0, 30, 100)
    HistsD21000    = makeAllHistograms(PointClouds, Normals, getD2Histogram, 3.0, 30, 1000)
    HistsD210000   = makeAllHistograms(PointClouds, Normals, getD2Histogram, 3.0, 30, 10000)
    HistsD2100000  = makeAllHistograms(PointClouds, Normals, getD2Histogram, 3.0, 30, 100000)
    HistsD21000000 = makeAllHistograms(PointClouds, Normals, getD2Histogram, 3.0, 30, 1000000)

    HistsA3        = makeAllHistograms(PointClouds, Normals, getA3Histogram, 30, 100000)

    DShapeE1  = compareHistsEuclidean(HistsShape1)
    DShapeE10 = compareHistsEuclidean(HistsShape10)
    DShapeE20 = compareHistsEuclidean(HistsShape20)
    DShapeE30 = compareHistsEuclidean(HistsShape30)

    DShapeC  = compareHistsCosine(HistsShape1)
    #DShapeCS = compareHistsChiSquared(HistsShape1)

    DD2E100     = compareHistsEuclidean(HistsD2100)
    DD2E1000    = compareHistsEuclidean(HistsD21000)
    DD2E10000   = compareHistsEuclidean(HistsD210000)
    DD2E100000  = compareHistsEuclidean(HistsD2100000)
    DD2E1000000 = compareHistsEuclidean(HistsD21000000)
    DD2C        = compareHistsCosine(HistsD21000000)
    #DD2CS = compareHistsChiSquared(HistsD2)

    DA3E    = compareHistsEuclidean(HistsA3)
    DA3C    = compareHistsCosine(HistsA3)
    #DA3CHS = compareHistsChiSquared(HistsA3)

    PRShapeE1  = getPrecisionRecall(DShapeE1)
    PRShapeE10 = getPrecisionRecall(DShapeE10)
    PRShapeE20 = getPrecisionRecall(DShapeE20)
    PRShapeE30 = getPrecisionRecall(DShapeE30)
    PRShapeC   = getPrecisionRecall(DShapeC)
    #PRShapeCS = getPrecisionRecall(DShapeCS)

    PRD2E100     = getPrecisionRecall(DD2E100)
    PRD2E1000    = getPrecisionRecall(DD2E1000)
    PRD2E10000   = getPrecisionRecall(DD2E10000)
    PRD2E100000  = getPrecisionRecall(DD2E100000)
    PRD2E1000000 = getPrecisionRecall(DD2E1000000)

    PRD2C    = getPrecisionRecall(DD2C)
    #PRD2CS   = getPrecisionRecall(DD2CS)

    PRA3E    = getPrecisionRecall(DA3E)
    PRA3C    = getPrecisionRecall(DA3C)
    #PRA3CS   = getPrecisionRecall(DA3CS)

    recalls = np.linspace(1.0/9.0, 1.0, 9)

    plt.figure(0)
    plt.hold(True)
    plt.title('Histogram Comparison by Cosine Distance')
    plt.plot(recalls, PRA3C,    'y',    label='A3')
    plt.plot(recalls, PRD2C,    'r',    label='D2')
    plt.plot(recalls, PRShapeC, 'black',label='Shape')
    #ax1.plot(recalls, PRShapeCS, 'black', label='Shape')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


    plt.figure(1)
    plt.hold(True)
    plt.title('Histogram Comparison by Euclidean Distance')
    plt.plot(recalls, PRA3E,     'b',    label='A3')
    plt.plot(recalls, PRD2E1000000,     'g',    label='D2')
    plt.plot(recalls, PRShapeE30,'cyan', label='Shape')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


    plt.figure(2)
    plt.hold(True)
    plt.title('Shell Histogram Comparison by Euclidean Dist. w/diff. # of bins')
    plt.plot(recalls, PRShapeE1,  'g',    label='1 bin')
    plt.plot(recalls, PRShapeE10, 'cyan', label='10 bins')
    plt.plot(recalls, PRShapeE20, 'red',  label='20 bins')
    plt.plot(recalls, PRShapeE30, 'b',    label='30 bins')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()


    plt.figure(3)
    plt.hold(True)
    plt.title('D2 Histogram Comparison by Euclidean Dist. w/diff. # of Samples')
    plt.plot(recalls, PRD2E100,     'g',    label='100 Samples')
    plt.plot(recalls, PRD2E1000,    'g',    label='1000 Samples')
    plt.plot(recalls, PRD2E10000,   'cyan', label='10000 Samples')
    plt.plot(recalls, PRD2E100000,  'red',  label='100000 Samples')
    plt.plot(recalls, PRD2E1000000, 'b',    label='1000000 Samples')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()'''
