#Purpose: To implement a suite of 3D shape statistics and to use them for point
#cloud classification
#TODO: Fill in all of this code for group assignment 2
import sys
sys.path.append("S3DGLPy")
from Primitives3D import *
from PolyMesh import *
from random import randint


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

    heatmap, xedges, yedges = np.histogram2d(mags,projPs,bins=(bins,bins),normed=True)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap, extent=extent)
    plt.show()

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
##              SYMMETRY COMPARISONS                  ##
#########################################################

#Takes in point cloud Ps and a line from p0 to p1, centers Ps on the midpoint
#of p0 and p1. It then computes the standard deviation of the distance from
#points in Ps to the line. Higher SD = less rotational symmetry around line.
#Function returns a tuple (StandardDeviation, CylinderRadius)
#IMPORTANT: Ps, p0, and p1 must use the same coodinate system.
def getCylindricalSymmetry(Ps, p0, p1):

    origin = (p0 + p1) / 2
    Ps = Ps - origin
    p0 = p0 - origin
    p1 = p1 - origin
    v  = p1 - p0

    perpProj = Ps - v.dot(v.T).dot(Ps)
    mags     = np.sqrt(np.einsum("ji,ji->i", perpProj, perpProj))

    SD = np.std(mags)
    R  = np.mean(mags)

    return SD, R

#### Initial code for PRST - currently doesnt work just a framework for later.
def w(point1, point2, y):
    return 1 / f(point1).dot(f(point2))
def reflectionPlane(point1, point2):
    n = point2 - point1
    mP  = (point2 + point1) /2
    return n.dot(mP)

def SD2(f, y):
    return 0
def PRST2(f, y):
    # 1 - SD2(f,y) / ||f||^2
    SD2 = SD2(f,y)
# Planar Reflective Symmetry Transform (Monte Carlo Algorithm)
def PRST(Ps, Ns):
    # Align with Principle Component Axis
    for i in xrange(Ps.shape[1]):
        point1 = Ps[:,i]
        for j in xrange(Ps.shape[1]):
            point2 = Ps[:,j]
            y = reflectionPlane(point1, point2) # This is going to be the reflection plane
            PRST2 += w(point1, point2, plane) * f(point1) * f(point2)


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
    D[D < 0] = 0
    return np.sqrt(D)

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
    shape = (AllHists.shape[1], AllHists.shape[1])
    def chiSquaredDist(a,b):
        h1 = AllHists[:,a]
        h2 = AllHists[:,b]
        f = np.vectorize(indvChiSquared)
        return np.sum(f(h1.flatten(), h2.flatten()), dtype=float)
    def indvChiSquared(a, b):
        n = 2 * np.square(a - b)
        d = a + b
        if n ==0:
            return 0
        return (n / float(d))

    f = np.vectorize(chiSquaredDist)
    x = np.fromfunction(lambda i, j: f(i, j), shape, dtype=int)
    return x

#########################################################
##                     MAIN TESTS                      ##
#########################################################

if __name__ == '__main__':
    NRandSamples = 10000 #You can tweak this number
    np.random.seed(100) #For repeatable results randomly sampling
    #Load in and sample all meshes
    PointClouds = []
    Normals = []
    '''for i in range(len(POINTCLOUD_CLASSES)):

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
    m = PolyMesh()
    filename = "models_off/biplane0.off"
    print "Loading ", filename
    m.loadOffFileExternal(filename)
    (Ps, Ns) = samplePointCloud(m, 10000)'''

    Ps = np.load("../data_0.npy") # Load data from the numpy parsed files
    x = [randint(0,1000000) for p in range(0,Ps.shape[1]-1)]
    Ps = Ps[:,:100000]               # Limit the number of points
    Ps = Ps[1:4,:]             # Only look at the vocel data as points

    # Recenter and scale points taken from voxel data
    c  = np.asmatrix([list(np.mean(Ps, axis=1))]).T
    Ps = Ps - c
    # Calculate scale
    squares = list(np.einsum("ji,ji->i", Ps, Ps))
    sums    = np.sum(squares)
    scale   = math.sqrt(sums/len(squares))
    # Apply Scale
    Ps      = Ps / scale

    PointClouds.append(Ps)
    Normals.append(Ps)

    HistsSpin = makeAllHistograms(PointClouds, Normals, getSpinImageFast,100, 2, 40)
