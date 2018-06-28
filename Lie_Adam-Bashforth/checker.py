import numpy as np
import os

# averaging over grid with variable square-bin sizes. For testing of result with different grids resolutions.
def gridBinAvg(grid,binSize):
    # get size of the averaged grid (half of grid size)
    avgGridSize = grid.shape[0]/2

    # initialize averaged grid.
    avgGrid = np.zeros((avgGridSize,avgGridSize))

    # bin size in 1D (Binsize=4 is 2 x 2)
    bin1DSize = binSize/2

    # get top-left coordinate of each bin.
    Ls = range(0,grid.shape[0],bin1DSize)

    # get 2 x 2 bins and average over them. Store this in the averaged grid
    yAvgGrid = 0
    for yGrid in Ls:
        xAvgGrid = 0
        for xGrid in Ls:
            avg = (grid[xGrid,yGrid]+grid[xGrid+1,yGrid]+grid[xGrid,yGrid+1]+grid[xGrid+1,yGrid+1])/float(binSize)
            avgGrid[xAvgGrid,yAvgGrid] = avg
            xAvgGrid += 1
        yAvgGrid += 1
    return avgGrid


# check if energy has increased within specified tolerance. Default tolerance: 1e-3.
def energyCheck(energy,tol=1e-3):
    for i in range(len(energy)-1):
        if (energy[i+1] - energy[i] >= tol):
            print('Energy increased from %.10f to %.10f at index %1.i.' %(energy[i],energy[i+1],i))
            break


# check if concentration stays within specified tolerance. Default tolerance: 5e-3.
def concCheck(avgConc,m,tol=5e-3):
    if (np.allclose(avgConc,np.ones(avgConc.shape)*m,rtol=tol,atol=tol)) != True:
        print "Average concentration is not conserved within tolerance."


# check if directory for output of arrays and digrams exists, otherwise create them.
def checkDir(imgDir,folder):
    path = imgDir + folder

    if os.path.isdir(imgDir) != True:
        os.mkdir(imgDir)

    if os.path.isdir(path) != True:
        os.mkdir(path)
