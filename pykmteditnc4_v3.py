#!/usr/bin/python

import pygame,os,sys
from pygame.locals import *
from gameobjects.vector2 import Vector2
import sys,os
from pylab import *
import math
import numpy as np
import collections
import time as gettime
from optparse import OptionParser
from scipy import signal
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.ndimage.interpolation import spline_filter
from scipy.ndimage.morphology import binary_erosion
import scipy.ndimage.measurements as meas

try:
    # for Python2
    import Tkinter as tk
    import tkSimpleDialog as tksd
    import tkMessageBox as tkmb
except:
    # for Python3
    import tkinter as tk
    import tkinter.simpledialog as tksd
    

try:
    import netCDF4 as netcdf
    print "netCDF4 IO constructor loaded..."
except ImportError:
	try:
#	    from netCDF3 import Datatset
	    import netCDF3 as netcdf
	    print "netCDF3 IO constructor Dataset loaded..."
	except IOError:
		print 'This version of pykmtedit requires netcdf4-python (i.e. netCDF3 or netCDF4)'
		sys.exit(1)    


def getUserName():
    try:
	import os, pwd, string
    except ImportError:
	return 'unknown user'
    pwd_entry = pwd.getpwuid(os.getuid())
    name = string.strip(string.splitfields(pwd_entry[4], ',')[0])
    if name == '':
	name = pwd_entry[0]
    return name


def argin():
    usage = """usage: %prog [options]
    This program will allow you to change the KMT file manually.
    The left mouse button will change a land cell to ocean or smooth the ocean value if it is already ocean.
    The right mouse botton will change an ocean cell to land or do nothing if it is already land.
    The enter key will allow you to enter a KMT value from 0 to maximum of kmt in the cursor location
    The arrow keys will allow you to move a cursor (green box) around the KMT map
    The escape key will save the new kmt file and quit.\n
    Added a gaussian fiter to smooth the ocean bottom topography (Press the S key)
    Added a gaussian fiter to smooth the ocean bottom topography with mask retained  (Press the M key): Use sigma = 1 seems to be good
    Added a box fiter to smooth the ocean bottom topography with mask actually working (Press the B key): This option works better than the S and M key because it maintains edge detection)

    Added a command to set minumum ocean bottom topography to depth level = 3  (Press the 3 key):
    Added a command to widen channels where to land points are one ocean space apart (Press the W key): This needs more work... not working properly
    Things todo:add caspian sea outline
		-Still working on masked arrays (like used in CLM files)
		-Allow for undo (ctrl-z), need to develop the buffer
		-Allow for zooming in and out (using scroll wheel)
		-Map grid to a cartesian or spherical system
		
    e.g. python pykmtedit.py -g -s 8 -f gridkmt.nc -o newgridkmt.nc
    This command will turn on the black grid using grid cells of size 8 pixels and use the respective input and output files.
    """
    
    parser = OptionParser(usage)
    parser.add_option("-f", "--infile",
		      dest="infile",
		      help="input KMT FILENAME")
    parser.add_option("-o", "--outfile",
		      dest="outfile",
		      help="output KMT FILENAME")
    parser.add_option("-s", "--gridsize",
                      action="store", # optional because action defaults to "store"
                      dest="sqrsz",
                      default=8,
                      help="Specify -s for the grid cell edge in pixels",)
    parser.add_option("-k", "--varmkt",
                      action="store", # optional because action defaults to "store"
                      dest="varkmt",
                      default="kmt",
                      help="Specify -k for netCDF kmt variable name (e.g. KMT)",)
    parser.add_option("-x", "--varlon",
                      action="store", # optional because action defaults to "store"
                      dest="varlon",
                      default="ULON",
                      help="Specify -x for netCDF lat variable name (e.g. ULONG)",)
    parser.add_option("-y", "--varlat",
                      action="store", # optional because action defaults to "store"
                      dest="varlat",
                      default="ULAT",
                      help="Specify -x for netCDF lon variable name (e.g. ULAT)",)
    parser.add_option("-g", "--grid",
		      action="store_true",
		      dest="grid",
                      help="Turn mesh on or off",)
    parser.add_option("-v", "--verbose",
		      action="store_true",
		      dest="verbose")
    parser.add_option("-q", "--quiet",
		      action="store_false",
		      dest="verbose")
    (options, args) = parser.parse_args()

    if options.infile == None:
	print "You must specify an input file with -f"
	sys.exit(1)
    if options.outfile == None:
	print "You must specify an output file with -o"
	sys.exit(1)
    if options.sqrsz == 0:
	print "You must specify a cell size > 0"
	sys.exit(1)

    return options, args


# A Python Library to create a Progress Bar.
# Copyright (C) 2008  BJ Dierkes <wdierkes@5dollarwhitebox.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
# This class is an improvement from the original found at:
#
#   http://code.activestate.com/recipes/168639/
#
 
 
class ProgressBar:
    def __init__(self, min_value = 0, max_value = 100, width=77,**kwargs):
        self.char = kwargs.get('char', '#')
        self.mode = kwargs.get('mode', 'dynamic') # fixed or dynamic
        if not self.mode in ['fixed', 'dynamic']:
            self.mode = 'fixed'
 
        self.bar = ''
        self.min = min_value
        self.max = max_value
        self.span = max_value - min_value
        self.width = width
        self.amount = 0       # When amount == max, we are 100% done 
        self.update_amount(0) 
 
 
    def increment_amount(self, add_amount = 1):
        """
        Increment self.amount by 'add_ammount' or default to incrementing
        by 1, and then rebuild the bar string. 
        """
        new_amount = self.amount + add_amount
        if new_amount < self.min: new_amount = self.min
        if new_amount > self.max: new_amount = self.max
        self.amount = new_amount
        self.build_bar()
 
 
    def update_amount(self, new_amount = None):
        """
        Update self.amount with 'new_amount', and then rebuild the bar 
        string.
        """
        if not new_amount: new_amount = self.amount
        if new_amount < self.min: new_amount = self.min
        if new_amount > self.max: new_amount = self.max
        self.amount = new_amount
        self.build_bar()
 
 
    def build_bar(self):
        """
        Figure new percent complete, and rebuild the bar string base on 
        self.amount.
        """
        diff = float(self.amount - self.min)
        percent_done = int(round((diff / float(self.span)) * 100.0))
 
        # figure the proper number of 'character' make up the bar 
        all_full = self.width - 2
        num_hashes = int(round((percent_done * all_full) / 100))
 
        if self.mode == 'dynamic':
            # build a progress bar with self.char (to create a dynamic bar
            # where the percent string moves along with the bar progress.
            self.bar = self.char * num_hashes
        else:
            # build a progress bar with self.char and spaces (to create a 
            # fixe bar (the percent string doesn't move)
            self.bar = self.char * num_hashes + ' ' * (all_full-num_hashes)
 
        percent_str = str(percent_done) + "%"
        self.bar = '[ ' + self.bar + ' ] ' + percent_str
 
 
    def __str__(self):
        return str(self.bar)
 
 



# UndoBuffer Not implemented yet, see http://stackoverflow.com/questions/3384643/how-to-build-an-undo-storage-with-limit
class UndoBuffer(object):
    def __init__(self,value,max_length=100):
        self.max_length=max_length
        self._buffer=collections.deque([value],max_length)
    @property
    def data(self):
        return self._buffer[-1]
    @data.setter
    def data(self,value):
        self._buffer.append(value)
    def restore(self,index):
        self.data=self._buffer[index]


def floatRgb(mag, cmin, cmax):
	"""
	Return a tuple of floats between 0 and 255 for the red, green and
	blue amplitudes.
	"""
	val = 255.0
# Make white (for land)
	if mag == 0.0:
	    (red, green, blue) = (255.,255.,255.)
# Make black (for grid border)
	elif mag == 256.0:
	    (red, green, blue) = (0.,0.,0.)
	elif mag == -1:
	    (red, green, blue) = (0.,255.,0.)
	else:	    
	    try:
		# normalize to [0,1]
		x = float(mag-cmin)/float(cmax-cmin)
	    except:
		# cmax = cmin
		x = 0.5
	    blue = min((max((4*(0.75-x), 0.)), 1.))*val
	    red  = min((max((4*(x-0.25), 0.)), 1.))*val
	    green= min((max((4*math.fabs(x-0.5)-1., 0.)), 1.))*val
	    
	return (red, green, blue)
	
def colorsquare(x,y,sqrsz,colorindx,grid,screen, vwidth):

    if grid:
	bsp = 1
    else:
	bsp = 0

    if vwidth == 0:
        xypos = (x+bsp,y+bsp)
	xysize = (sqrsz-bsp,sqrsz-bsp)
    else:
        xypos = (x,y)
	xysize = (sqrsz+1,sqrsz+1)

    cmapcolor = floatRgb(colorindx, 0, 255)
    pygame.draw.rect(screen, cmapcolor, Rect(xypos, xysize), vwidth)

def lnd2ocn(kmtxy,j,i,nlat,nlon):

# This function creates a box of 9 grid cells and uses the average to change a land cell to ocean
# The case of corner grid cells (poles will always be land but include anyway)
    gridbuff = np.zeros((3,3))
    nlonm1 = nlon-1
    nlatm1 = nlat-1
# Fill the grid buffer with values
#top left
    if j == 0 and i == 0:
	gridbuff[0,0] = kmtxy[nlatm1,nlonm1]
	gridbuff[0,1:3] = kmtxy[nlatm1,i:i+2]
	gridbuff[1,0] = kmtxy[j,nlonm1]
	gridbuff[1,1:3] = kmtxy[j,i:i+2]
	gridbuff[2,0] = kmtxy[j+1,nlonm1]
	gridbuff[2,1:3] = kmtxy[j+1,i:i+2]

#top right
    elif j == 0 and i == nlon:
	gridbuff[0,0:2] = kmtxy[nlatm1,i-1:i+1]
	gridbuff[0,2] = kmtxy[nlatm1,0]
	gridbuff[1,0:2] = kmtxy[j,i-1:i+1]
	gridbuff[1,2] = kmtxy[j,0]
	gridbuff[2,0:2] = kmtxy[j+1,i-1:i+1]
	gridbuff[2,2] = kmtxy[j+1,0]

#bottom left
    elif j == nlat and i == 0:
	gridbuff[0,0] = kmtxy[j-1,nlonm1]
	gridbuff[0,1:3] = kmtxy[j-1,i:i+2]
	gridbuff[1,0] = kmtxy[j,nlonm1]
	gridbuff[1,1:3] = kmtxy[j,i:i+2]
	gridbuff[2,0] = kmtxy[0,nlonm1]
	gridbuff[2,1:3] = kmtxy[0,i:i+2]

#bottom right
    elif j == nlat and i == nlon:
	gridbuff[0,0:2] = kmtxy[j-1,i-1:i+1]
	gridbuff[0,2] = kmtxy[j-1,0]
	gridbuff[1,0:2] = kmtxy[j,i-1:i+1]
	gridbuff[1,2] = kmtxy[j,0]
	gridbuff[2,0:2] = kmtxy[j+1,i-1:i+1]
	gridbuff[2,2] = kmtxy[j+1,0]

# top
    elif j == 0 and (i != 0 and i != nlonm1):
	gridbuff[0,:] = kmtxy[nlatm1,i-1:i+2]
	gridbuff[1,:] = kmtxy[j,i-1:i+2]
	gridbuff[2,:] = kmtxy[j+1,i-1:i+2]
# bottom
    elif j == nlat and (i != 0 and i != nlonm1):
	gridbuff[0,:] = kmtxy[j-1,i-1:i+2]
	gridbuff[1,:] = kmtxy[j,i-1:i+2]
	gridbuff[2,:] = kmtxy[0,i-1:i+2]

#left
    elif i == 0 and (j != 0 and j != nlatm1):
	gridbuff[:,0] = kmtxy[j-1:j+2,nlonm1]
	gridbuff[:,1] = kmtxy[j-1:j+2,i]
	gridbuff[:,2] = kmtxy[j-1:j+2,i+1]
#right
    elif i == nlon and (j != 0 and j != nlatm1):
	gridbuff[:,0] = kmtxy[j-1:j+2,i-1]
	gridbuff[:,1] = kmtxy[j-1:j+2,i]
	gridbuff[:,2] = kmtxy[j-1:j+2,0]
	
    else:
	gridbuff = kmtxy[j-1:j+2,i-1:i+2]
#	print gridbuff, j, i
	
#now calculate average of gridbuff box
    gridbuffavg = int(sum(sum(gridbuff))/9)

    return gridbuffavg



def enterkmtval(kmtmax):

    root = tk.Tk()
    root.withdraw()
    root["padx"] = 30
    root["pady"] = 20       
    kmtstr = str(kmtmax)+") :"
    kmt = tksd.askinteger("KMT Modify Value", "Enter the KMT Value (0-"+kmtstr, parent=root, minvalue=0, maxvalue=kmtmax)
    if kmt == None:
	root.destroy()
    else:
        return kmt

#    root.mainloop()

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size) + y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)

    improc = signal.convolve(im, g, mode='same') # valid,same,full    
    return(improc)

def entersmoothinfo(screen,kmtxy,xr,yr,sqr,g,w,kmax,lats,lons):

    root = tk.Tk()
    root.withdraw()

    LMChooser = tkmb.askyesnocancel("Land Mask Filter","Replace original land mask after the gaussian filter is applied?")
    print LMChooser
    if LMChooser == None:
	root.destroy()
	return None, None, None
    else:
	root["padx"] = 30
	root["pady"] = 20       

	nwindow = tksd.askinteger("Gaussian Blur", "Enter the window radius (min 1)", parent=root, minvalue=1, maxvalue=500)

	if nwindow == None:
	    root.destroy()
	    return None, None, None
	else:
# Need to pad earthgrid in a 3x3 array to take away edge effects with convolve same option
	    kmtpad = np.zeros((kmtxy.shape[0]*3,kmtxy.shape[1]*3))
	    for i in range(0,3):
		for j in range(0,3):
		    kmtpad[lats*i:lats*i+lats,lons*j:lons*j+lons] = kmtxy[:,:] 		
#	    knew = np.ceil(blur_image(kmtxy,nwindow))	
	    print "applying gaussian filter (window size = %i) to kmt grid... Please wait..." %nwindow

	    kpadnew = blur_image(kmtpad,nwindow)	
#	    knew = kmtxy.copy()
	    knew = kpadnew[1*lats:2*lats,1*lons:2*lons]
	    return knew, nwindow, LMChooser

#    root.mainloop()

def isodd(n):
    return bool(n%2)


def nested_change(item, func):
    if isinstance(item, list):
        return [nested_change(x, func) for x in item]
    return func(item)


def maskedboxavg(screen,kmtxy,xr,yr,sqr,g,w,kmax,lats,lons):

    root = tk.Tk()
    root.withdraw()

    nwindow = tksd.askinteger("Smooth", "Enter the box dimension (3,5,7,9)", parent=root, minvalue=3, maxvalue=9)

    lndmask=np.ma.masked_equal(kmtxy,0)
    ocnmask=np.ma.masked_greater(kmtxy,0)
    ocnmask.mask=np.logical_not(lndmask.mask)

    if nwindow == None or isodd(nwindow) == False:
	    root.destroy()
	    return None, None, None
    else:
# Need to pad earthgrid in a 3x3 stencil to take away edge effects
	    kmtpad = np.zeros((kmtxy.shape[0]*3,kmtxy.shape[1]*3))
	    kmtpadnew = np.zeros((kmtxy.shape[0]*3,kmtxy.shape[1]*3))
	    lndmaskpad = np.ma.zeros((lndmask.shape[0]*3,lndmask.shape[1]*3))
	    ocnmaskpad = np.ma.zeros((ocnmask.shape[0]*3,ocnmask.shape[1]*3))

	    for i in range(0,3):
		for j in range(0,3):
		    kmtpad[lats*i:lats*i+lats,lons*j:lons*j+lons] = kmtxy[:,:] 		
		    lndmaskpad[lats*i:lats*i+lats,lons*j:lons*j+lons] = lndmask[:,:] 		
		    lndmaskpad.mask[lats*i:lats*i+lats,lons*j:lons*j+lons] = lndmask.mask[:,:] 		
		    ocnmaskpad[lats*i:lats*i+lats,lons*j:lons*j+lons] = ocnmask[:,:] 		
		    ocnmaskpad.mask[lats*i:lats*i+lats,lons*j:lons*j+lons] = ocnmask.mask[:,:] 		

	    print "applying masked nxn stencil smoother (window n size = %i) to kmt grid... Please wait..." %nwindow


	    halfint=nwindow/2
	    kmtbox = np.ma.zeros((nwindow,nwindow))
	    kmtbox.mask = False

            # Make weights (Note: These Weights have to conform with the nwindow x nwindow mask)

            prog = ProgressBar(lats-halfint, 2*lats+halfint, 77, mode='fixed', char='#')

            for i in range(lats-halfint,2*lats+halfint):
		prog.increment_amount()
                print prog, '\r',
                sys.stdout.flush()
	        for j in range(lons-halfint,2*lons+halfint):

		    for ii in range(0,nwindow):
                       for jj in range(0,nwindow):
             		    kmtbox[ii,jj] = kmtpad[i+ii,j+jj]
		            kmtbox.mask[ii,jj] = lndmaskpad.mask[i+ii,j+jj]

		    if ocnmaskpad.mask[i+halfint,j+halfint]:
      	    	    	kmtpadnew[i+halfint,j+halfint] = np.ma.average(kmtbox)	
		    else:
			kmtpadnew[i+halfint,j+halfint] = 0.0
	    print "\n"

            # Convert NaN to 0
#            kmtpadnew[np.isnan(kmtpadnew)] = 0.0
	    knew = np.round(kmtpadnew[1*lats:2*lats,1*lons:2*lons],decimals=0)

	    return np.int32(knew)

#    root.mainloop()


def maskedfuncavg(screen,kmtxy,xr,yr,sqr,g,w,kmax,lats,lons):

    root = tk.Tk()
    root.withdraw()

    nwindow = tksd.askinteger("Smooth", "Enter the box width (3,5,7,9)", parent=root, minvalue=3, maxvalue=9)
    igaussbox = tksd.askinteger("Box or Gaussian Smooth", "0=Gaussian,1=Box", parent=root, minvalue=0, maxvalue=1)

    lndmask=np.ma.masked_equal(kmtxy,0)
    ocnmask=np.ma.masked_greater(kmtxy,0)
#    print lndmask[0,:]
#    print lndmask.mask[0,:]
    ocnmask.mask=np.logical_not(lndmask.mask)
#    print ocnmask[0,:]
#    print ocnmask.mask[0,:]

    if nwindow == None or isodd(nwindow) == False:
	    root.destroy()
	    return None, None, None
    else:
# Need to pad earthgrid in a 3x3 array to take away edge effects with convolve same option
	    kmtpad = np.zeros((kmtxy.shape[0]*3,kmtxy.shape[1]*3))
	    kmtpadnew = np.zeros((kmtxy.shape[0]*3,kmtxy.shape[1]*3))
	    lndmaskpad = np.ma.zeros((lndmask.shape[0]*3,lndmask.shape[1]*3))
	    ocnmaskpad = np.ma.zeros((ocnmask.shape[0]*3,ocnmask.shape[1]*3))

	    for i in range(0,3):
		for j in range(0,3):
		    kmtpad[lats*i:lats*i+lats,lons*j:lons*j+lons] = kmtxy[:,:] 		
		    lndmaskpad[lats*i:lats*i+lats,lons*j:lons*j+lons] = lndmask[:,:] 		
		    lndmaskpad.mask[lats*i:lats*i+lats,lons*j:lons*j+lons] = lndmask.mask[:,:] 		
		    ocnmaskpad[lats*i:lats*i+lats,lons*j:lons*j+lons] = ocnmask[:,:] 		
		    ocnmaskpad.mask[lats*i:lats*i+lats,lons*j:lons*j+lons] = ocnmask.mask[:,:] 		

	    print "applying masked smoother (window size = %i) to kmt grid... Please wait..." %nwindow


	    halfint=nwindow/2
	    kmtbox = np.ma.zeros((nwindow,nwindow))
	    kmtbox.mask = False

            # Make weights (Note: These Weights have to conform with the nwindow x nwindow mask)

	    if igaussbox == 0:
        	    boxgaussweights = gauss_kern(halfint)
	    else:
		    boxgaussweights = np.ones((nwindow,nwindow))
		    boxgaussweights = boxgaussweights/(nwindow**2)

	    print "Box weights are:", boxgaussweights


            prog = ProgressBar(lats-halfint, 2*lats+halfint, 77, mode='fixed', char='#')

            for i in range(lats-halfint,2*lats+halfint):
		prog.increment_amount()
                print prog, '\r',
                sys.stdout.flush()
	        for j in range(lons-halfint,2*lons+halfint):

		    for ii in range(0,nwindow):
                       for jj in range(0,nwindow):
             		    kmtbox[ii,jj] = kmtpad[i+ii,j+jj]*boxgaussweights[ii,jj]
		            kmtbox.mask[ii,jj] = lndmaskpad.mask[i+ii,j+jj]

	            print np.isnan(kmtbox)

# Need to average over non-masked values only
		    if ocnmaskpad.mask[i+halfint,j+halfint]:
      	    	    	kmtpadnew[i+halfint,j+halfint] = np.ma.sum(kmtbox)	
		    else:
			kmtpadnew[i+halfint,j+halfint] = 0.0
	    print "\n"

            # Convert NaN to 0
#            kmtpadnew[np.isnan(kmtpadnew)] = 0.0
	    knew = np.round(kmtpadnew[1*lats:2*lats,1*lons:2*lons],decimals=0)

	    return np.int32(knew)

#    root.mainloop()



def smooth_with_function_and_mask(image, function, mask):
    """Smooth an image with a linear function, ignoring masked pixels

    Parameters
    ----------
    image : array
      The image to smooth
    
    function : callable
      A function that takes an image and returns a smoothed image
    
    mask : array
      Mask with True for significant pixels, False for masked pixels

    Notes
    ------
    This function calculates the fractional contribution of masked pixels
    by applying the function to the mask (which gets you the fraction of
    the pixel data that's due to significant points). We then mask the image
    and apply the function. The resulting values will be lower by the bleed-over
    fraction, so you can recalibrate by dividing by the function on the mask
    to recover the effect of smoothing from just the significant pixels.
    """

    bleed_over = function(mask.astype(float))
    masked_image = np.zeros(image.shape, image.dtype)
    masked_image[mask] = image[mask]
    smoothed_image = function(masked_image)
    output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
    return output_image


def entersmoothmask(screen,kmtxy,xr,yr,sqr,g,w,kmax,lats,lons):

    root = tk.Tk()
    root.withdraw()

    filttype = tksd.askinteger("Smoothing Filter", "0=Uniform, 1=Gaussian, 2=Spline, 3=Binary Erosion", parent=root, minvalue=0, maxvalue=3)


    sigmasize = tksd.askfloat("Filter Size", "Enter the size (for Uniform) or standard deviation (for Gaussian) or order (for Spline)", parent=root, minvalue=0, maxvalue=32)
#    modeval = tksd.askinteger("Filter Mode", "Enter the filter edge effect mode (0=reflect),(1=constant),(2=nearest),(3=mirror),(4=wrap)", parent=root, minvalue=0, maxvalue=4)

# "wrap" is the best boundary condtion mode (at least for longitude). The poles should be missing values anyway  
    modelist = ['reflect','constant','nearest','mirror', 'wrap']
    print "applying gaussian filter to kmt grid... Please wait..."
    modeval = 4	

    lndmask=np.ma.masked_equal(kmtxy,0)
    ocnmask=np.logical_not(lndmask.mask)


    if filttype == 0:
#        fsmooth = lambda x: uniform_filter(x, sigmasize, mode=modelist[modeval])
	smoothed = uniform_filter(kmtxy, sigmasize, mode=modelist[modeval])
    elif filttype == 1:
        fsmooth = lambda x: gaussian_filter(x, sigmasize, mode=modelist[modeval])
        smoothed = smooth_with_function_and_mask(kmtxy, fsmooth, ocnmask)
    elif filttype == 2:
#        fsmooth = lambda x: spline_filter(x, order=sigmasize)
	smoothed = spline_filter(kmtxy, order=sigmasize)
    else:
#        fsmooth = lambda x: binary_erosion(x, interations=sigmasize)
	smoothed = binary_erosion(kmtxy, iterations=sigmasize, mask=ocnmask)

#    smoothed = smooth_with_function_and_mask(kmtxy, fsmooth, ocnmask)

    return smoothed

#    root.mainloop()



def printposxy(spy,spx,lats,lons,sqrsz,ulat,ulon):
	ycoord = lats - spy/sqrsz
	xcoord = spx/sqrsz
	if xcoord < 0: xcoord = 0
	if xcoord > lons-1: xcoord = lons-1
	if ycoord < 0: ycoord = 0
	if ycoord > lats-1: ycoord = lats-1
	print "y: %i, x: %i, latitude: %f, longitude: %f" % (ycoord, xcoord, ulat[int(ycoord),int(xcoord)],ulon[int(ycoord),int(xcoord)])


def widenchannels(kmtxy,lats,lons):

	print "applying channel widening for one grid wide channels"
	kmtpad = np.zeros((3,3))
	kmtnew = kmtxy
	for i in range(1,lats-3,3):
	  for j in range(1,lons-3,3):
	    kmtpad = kmtxy[i-1:i+2,j-1:j+2] == 0
            ksum = sum(sum(kmtpad))
	    if ksum >= 2 and ksum <=8:
#                print i,j,sum(sum(kmtpad)),kmtpad
		#corners
		if kmtpad[0,0] == True and kmtpad[2,2] == True and kmtpad[1,1] == False: kmtnew[i-1,j-1] = kmtnew[i,j] 
		if kmtpad[2,0] == True and kmtpad[0,2] == True and kmtpad[1,1] == False: kmtnew[i-1,j+1] = kmtnew[i,j] 
		#across vertical
		if kmtpad[0,0] == True and kmtpad[0,2] == True and kmtpad[0,1] == False: kmtnew[i-1,j-1] = kmtnew[i-1,j] 
		if kmtpad[1,0] == True and kmtpad[1,2] == True and kmtpad[1,1] == False: kmtnew[i,j-1] = kmtnew[i,j] 
		if kmtpad[2,0] == True and kmtpad[2,2] == True and kmtpad[2,1] == False: kmtnew[i+1,j-1] = kmtnew[i+1,j] 
		#across horizontal
		if kmtpad[0,0] == True and kmtpad[2,0] == True and kmtpad[1,0] == False: kmtnew[i-1,j-1] = kmtnew[i,j-1] 
		if kmtpad[0,1] == True and kmtpad[2,1] == True and kmtpad[1,1] == False: kmtnew[i-1,j] = kmtnew[i,j] 
		if kmtpad[0,2] == True and kmtpad[2,2] == True and kmtpad[1,2] == False: kmtnew[i-1,j+1] = kmtnew[i,j+1] 


	return kmtnew



#####---------------------------MAIN    

def main():


    opts, args = argin()

    grid = opts.grid
    sqrsz = int(opts.sqrsz)
    infile = opts.infile
    outfile = opts.outfile

    strkmt = opts.varkmt
    strlon = opts.varlon
    strlat = opts.varlat

    print "Reading variable: %s with num longitude: %s and latitude: %s" %(strkmt,strlon,strlat)

#ncpath = Nio.open_file(infile, 'r')
    ncpath = netcdf.Dataset(infile, 'r')

    try:
    	kmt = ncpath.variables[strkmt][:,:]
    except IOError as (errno, strerror):
    	print "I/O error({0}): {1}".format(errno, strerror)

    try:
    	ulon = ncpath.variables[strlon][:,:]
   	lons = len(ulon[0,:])
    	curvlin = True
    except ValueError:
    	ulon = ncpath.variables[strlon][:]
   	lons = len(ulon[:])
    	curvlin = False
    except IOError as (errno, strerror):
    	print "I/O error({0}): {1}".format(errno, strerror)
    
    try:
    	ulat = ncpath.variables[strlat][:,:]
  	lats = len(ulat[:,0])
    except ValueError:
    	ulat = ncpath.variables[strlat][:]
   	lats = len(ulat[:])
    except IOError as (errno, strerror):
    	print "I/O error({0}): {1}".format(errno, strerror)

    lon = ulon
    lat = ulat
    print "#longitudes: %i, #latitudes: %i" %(lons,lats)

    kmtmax = np.max(kmt)
#    print kmt.shape

# if there is a time dimension take it out
# amd reverse the field to print it out visially correct (must re-reverse at end)

    if kmt.ndim == 3:
    	modkmt = kmt[0,:,:].copy()
    	modkmt = modkmt[::-1,:]
    else:
    	modkmt = kmt.copy()
    	modkmt = modkmt[::-1,:]

#   print modkmt.ndim
#   print modkmt.shape
#    print modkmt.dtype

# To do: need to rework code for masked arrays.

#   print modkmt.dtype
    if np.ma.count(modkmt) != modkmt.shape[0]*modkmt.shape[1]:
    	print "Working with masked array"
   	maskedarray = True
    else:
  	print "Working with unmasked array"
   	maskedarray = False

    print "%s grid has the dimensions (row: %i, col: %i)" %(strkmt,modkmt.shape[0],modkmt.shape[1])
    print "%s has a min val: %f and a max val: %f" %(strkmt, np.min(kmt), np.max(kmt))

    xres = lons*sqrsz
    yres = lats*sqrsz

# New file open
    os.system('rm '+outfile)

#   f = netcdf.Dataset(outfile, 'w', None, 'Created ' + gettime.ctime(gettime.time()) + ' by ' + getUserName())
    f = netcdf.Dataset(outfile, 'w', None)

#   f.data = "Modified KMT file for CCSM3: old kmtfile = %s, new kmtfile = %s" % (infile, outfile)
#   f.create_dimension("longitude", lons)
#   f.create_dimension("latitude", lats)
    f.createDimension("longitude", lons)
    f.createDimension("latitude", lats)

#    lon = f.create_variable("ULON", "f", ("latitude","longitude"))
#    lat = f.create_variable("ULAT", "f", ("latitude","longitude"))
    if curvlin:
   	 lon = f.createVariable("ULON", "f8", ("latitude","longitude"))
   	 lat = f.createVariable("ULAT", "f8", ("latitude","longitude"))
    else:
   	 lon = f.createVariable("LON", "f8", ("longitude",))
   	 lat = f.createVariable("LAT", "f8", ("latitude",))
    
#    kmtnew = f.create_variable("kmt","f",("latitude","longitude"))
    kmtnew = f.createVariable(strkmt,"i4",("latitude","longitude"))
    f.description = "Modified %s file for CCSM3: old kmtfile = %s, new kmtfile = %s" % (strkmt, infile, outfile)
    f.history = 'Created '+ gettime.ctime(gettime.time()) + ' by ' + getUserName()

# start editor code
    pygame.init()
    screen = pygame.display.set_mode((xres, yres), 0, 32)

#    screen.fill((255,255,255))
    screen.fill((0,0,0))


# initial values
    x = sqrsz
    y = sqrsz
    width = 0
    ii = 1 # cursor start at x
    ij = 1 # cursor start at y
    sprite_pos = Vector2(x*ii, y*ij)
    arrowpress = False
    enterval = False
#    if grid:
#	bsp = 1
#    else:
#	bsp = 0

#   xyipos = (x*ii+bsp,y*ij+bsp)
#   xyisize = (sqrsz-bsp,sqrsz-bsp)
#    pygame.draw.rect(screen, (255,0,0), Rect(xyipos, xyisize))
    colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,0,grid,screen, width)

    pygame.display.set_caption("kmt Editor for CCSM3")

#    screen.lock()
    for x in range(0,xres,sqrsz):
        i=x/sqrsz
        for y in range(0,yres,sqrsz):
	    j=y/sqrsz
# check for masked or missing value
	    if np.ma.count(modkmt[j,i]) != 1:
		cindx = 256
	    else:
		cindx = int(255*modkmt[j,i]/kmtmax)
	    colorsquare(x,y,sqrsz,cindx,grid, screen, width)

    if grid:
#vertical black lines
	print "setting grid..."
#        pygame.draw.line(screen,(0,0,0),(x,0),(x,yres),1)
    
#horizontal black lines
#        pygame.draw.line(screen,(0,0,0),(0,sqrsz),(xres,y),1)

        for x in range(0,xres,sqrsz):
            x = x + sqrsz
            pygame.draw.line(screen,(0,0,0),(x,0),(x,yres),1)

        for y in range(0,yres,sqrsz):
            y = y + sqrsz
            pygame.draw.line(screen,(0,0,0),(0,y), (xres,y),1)
        
	pygame.display.flip()

# Add sprite arrow control object
#    pygame.draw.rect(screen, (255,0,0), Rect(xyipos, xyisize))


#add Caspian Sea in red grid lines
    


 #   screen.unlock()
    pygame.display.update()

    done = False
    while ~done:
        for e in pygame.event.get():

            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                done = True
                break 

	    if e.type == MOUSEBUTTONDOWN:
		if e.button == 1:
#		    left button clicked (change land to ocean)
		    xp = (e.pos[0]/sqrsz)*sqrsz  #0 is the x position
		    yp = (e.pos[1]/sqrsz)*sqrsz  #1 is the y position
		    idx = int(xp/sqrsz)		    
		    jdx = int(yp/sqrsz)
		    jmdx = lats - int(yp/sqrsz)
		    print "x index = %i, y index = %i (%i), kmt = %i" % (idx, jdx, jmdx, modkmt[jdx,idx])
		    kmtchange = lnd2ocn(modkmt, jdx, idx, lats, lons)
		    cindx = int(255*kmtchange/kmtmax)
		    colorsquare(xp, yp, sqrsz, cindx, grid, screen, width)
		    modkmt[jdx,idx] = kmtchange
		    print "The value has been changed to ocean using a average of 9 points: ", kmtchange
		    printposxy(e.pos[1],e.pos[0],lats,lons,sqrsz,ulat,ulon)

		    pygame.display.flip()
# move cursor to mouse position
		    if grid:
			cindx = int(255*modkmt[sprite_pos[1]/sqrsz,sprite_pos[0]/sqrsz]/kmtmax)
		        colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,cindx,grid,screen, 0)
		        colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,256,grid, screen ,1)
# -1 = green outline to cursor
			sprite_pos = (idx*sqrsz,jdx*sqrsz)
		        colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,-1,grid,screen,1)
		        pygame.display.flip()
		    else:
			cindx = int(255*modkmt[sprite_pos[1]/sqrsz,sprite_pos[0]/sqrsz]/kmtmax)
		        colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,cindx,grid,screen,1)
# -1 = green outline to cursor
			sprite_pos = (idx*sqrsz,jdx*sqrsz)
		        colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,-1,grid,screen, 1)
		        pygame.display.flip()
		elif e.button == 2:
		    print "middle button clicked: not used"
		elif e.button == 3:
#		    right button clicked (change ocean to land)
		    xp = (e.pos[0]/sqrsz)*sqrsz  #0 is the x position
		    yp = (e.pos[1]/sqrsz)*sqrsz  #1 is the y position
		    idx = int(e.pos[0]/sqrsz)    
		    jdx = int(e.pos[1]/sqrsz)
		    print "The cell has been changed to land: lon=%i, lat=%i, kmt old=%i" % (idx, jdx, modkmt[jdx,idx])
		    width = 0
		    colorsquare(xp, yp, sqrsz, 0, grid, screen, width)
		    modkmt[jdx,idx] = 0
		    print "kmt new=%i" % (modkmt[jdx,idx])
		    pygame.display.flip()
# move cursor to mouse position
		    if grid:
			cindx = int(255*modkmt[sprite_pos[1]/sqrsz,sprite_pos[0]/sqrsz]/kmtmax)
		        colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,cindx,grid, screen,0)
		        colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,256,grid, screen,1)
# -1 = green outline to cursor
			sprite_pos = (idx*sqrsz,jdx*sqrsz)
		        colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,-1,grid, screen,1)
		        pygame.display.flip()
		    else:
			cindx = int(255*modkmt[sprite_pos[1]/sqrsz,sprite_pos[0]/sqrsz]/kmtmax)
		        colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,cindx,grid, screen,1)
# -1 = green outline to cursor
			sprite_pos = (idx*sqrsz,jdx*sqrsz)
		        colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,-1,grid, screen,1)
		        pygame.display.flip()

		elif e.button == 4:
		    print "scrolling forward: not used"
		elif e.button == 5:
		    print "scrolling backward: not used"
		else:
		    print "some cool button: not used"
#		print e.pos
#		print e.button

# Move the cursor to input values at a specific location
	    if e.type == 2:
		print "key pressed =", e.key

		key_direction = Vector2(0, 0)
		if e.key == 273:
		    print "up arrow"
		    key_direction.y = -1
		    arrowpress = True
		elif e.key == 276:
		    print "left arrow"
		    key_direction.x = -1
		    arrowpress = True
		elif e.key == 275:
		    print "right arrow"
		    key_direction.x = +1
		    arrowpress = True
		elif e.key == 274:
		    print "down arrow"
		    key_direction.y = +1
		    arrowpress = True
		elif e.key == 13:
		    print "enter key"
		    enterval = True
		elif e.key == 115:
		    print "S key: Gaussian smooth"
		    enterval = True
		elif e.key == 119:
		    print "W key: Widen channels"
		    enterval = True
		elif e.key == 109:
		    print "M key: Masked smooth"
		    enterval = True
		elif e.key == 98:
		    print "B key: Masked Box average"
		    enterval = True
		elif e.key == 102:
		    print "F key: Masked Function average"
		    enterval = True
		elif e.key == 51:
		    print "3 key: Min Ocean Depth=3"
		    enterval = True
		elif e.key == 57:
		    print "9 key: Pan Up"
		    enterval = True
		elif e.key == 112:
		    print "P key: Pan Right"
		    enterval = True
		elif e.key == 105:
		    print "I key: Pan Left"
		    enterval = True
		elif e.key == 108:
		    print "L key: Pan Down"
		    enterval = True
		elif e.key == 27:
		    print "Finished Editing... Exiting"
		    enterval = True
		else:
		    print "some key: %i ----not used " %e.key
		#  move the cursor
		if arrowpress:
		#    print "y: %i, x: %i, latitude: %f, longitude: %f" % (ycoord, xcoord, ulat[int(ycoord),int(xcoord)],ulon[int(ycoord),int(xcoord)])
		    key_direction.normalize()
		    cindx = int(255*modkmt[sprite_pos[1]/sqrsz,sprite_pos[0]/sqrsz]/kmtmax)
		    arrowpress = False
		    if grid:
		        colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,cindx,grid, screen,0)
		        colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,256,grid, screen,1)
		        sprite_pos += key_direction * sqrsz
# -1 = green outline to cursor
		        colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,-1,grid, screen,1)
		        pygame.display.flip()
		    else:
		        colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,cindx,grid, screen,1)
		        sprite_pos += key_direction * sqrsz
# -1 = green outline to cursor
		        colorsquare(sprite_pos[0],sprite_pos[1],sqrsz,-1,grid, screen,1)
		        pygame.display.flip()

		    printposxy(sprite_pos[1],sprite_pos[0],lats,lons,sqrsz,ulat,ulon)

# Enter integer in box to add specific kmt value
		if enterval and e.key == 13:
		    kmtent = enterkmtval(kmtmax)
		    if kmtent != None:
			idx = int(sprite_pos[0]/sqrsz)		    
			jdx = int(sprite_pos[1]/sqrsz)
			modkmt[jdx,idx] = kmtent
			cindx = int(255*kmtent/kmtmax)
			colorsquare(sprite_pos[0],sprite_pos[1], sqrsz, cindx, grid, screen, width)
			print "The grid cell has been changed to kmt = ", kmtent
			pygame.display.flip()
			enterval = False

		if enterval and e.key == 119:
		    modkmt = widenchannels(modkmt,lats,lons)
		    for x in range(0,xres,sqrsz):
			i=x/sqrsz
			for y in range(0,yres,sqrsz):
			    j=y/sqrsz
			    cindx = int(255*modkmt[j,i]/kmtmax)
			    colorsquare(x, y,sqrsz, cindx, grid, screen, width)
		    pygame.display.flip()
		    print "replotting kmt grid with channels widened..."

# Set minimum ocean depth to KMT=3


		if enterval and e.key == 51:

		    root = tk.Tk()
		    root.withdraw()
		    choose3 = tkmb.askyesnocancel("Min Ocean Depth","Set minimum ocean depth to KMT=3?")
		    print choose3
		    if choose3 == False or choose3 == None:
		  	root.destroy()
   		    else:
       		        for i in range(0,lats):
		          for j in range(0,lons):

			     if modkmt[i,j] > 0 and modkmt[i,j] <= 2:
				modkmt[i,j] = 3

		        for x in range(0,xres,sqrsz):
			  i=x/sqrsz
			  for y in range(0,yres,sqrsz):
			    j=y/sqrsz
			    cindx = int(255*modkmt[j,i]/kmtmax)
			    colorsquare(x, y,sqrsz, cindx, grid, screen, width)

		        pygame.display.flip()
		        print "replotting kmt grid with ocean values set to a minimum of 3..."
		        root.destroy()




# Smooth ocean bottom with a gaussian blur, but keep continental boundaries intact
		if enterval and e.key == 115:
		    kmtgauss, gwin, lmreplace = entersmoothinfo(screen, modkmt, xres, yres, sqrsz, grid, width, kmtmax, lats, lons)
		    if kmtgauss != None:

			# create a mask to leave land mask same for default below
			lm = np.ones(modkmt.shape,dtype = np.bool)
			if lmreplace:
			        print "Replacing land mask after gaussian blurr..."
				lm = modkmt[:,:] != 0
			# apply changes including reapplying original landmask if true
			modkmt[lm] = kmtgauss[lm]
#			print modkmt			

			for x in range(0,xres,sqrsz):
			    i=x/sqrsz
			    for y in range(0,yres,sqrsz):
			        j=y/sqrsz
#			        cindx = int(255*kmtgauss[j,i]/kmtmax)
			        cindx = int(255*modkmt[j,i]/kmtmax)
			        colorsquare(x, y,sqrsz, cindx, grid, screen, width)
			print "replotting kmt grid with gauss filter..."

		        if grid:
			    #vertical black lines
		            for x in range(0,xres,sqrsz):
		                x = x + sqrsz
		                pygame.draw.line(screen,(0,0,0),(x,0),(x,yres),1)

			    for y in range(0,yres,sqrsz):
		                y = y + sqrsz
		                pygame.draw.line(screen,(0,0,0),(0,y), (xres,y),1)

		        pygame.display.flip()
			enterval = False

# Smooth ocean bottom with a gaussian blur, but keep continental boundaries intact
		if enterval and e.key == 109:
		    kmtgaumask = entersmoothmask(screen, modkmt, xres, yres, sqrsz, grid, width, kmtmax, lats, lons)
		    if kmtgaumask != None:

			# create a mask to leave land mask same for default below
			print "Replacing land mask after gaussian blurr..."
			lm = modkmt[:,:] != 0
			# apply changes including reapplying original landmask if true
			modkmt[lm] = kmtgaumask[lm]

#			modkmt = kmtgaumask

			for x in range(0,xres,sqrsz):
			    i=x/sqrsz
			    for y in range(0,yres,sqrsz):
			        j=y/sqrsz
			        cindx = int(255*modkmt[j,i]/kmtmax)
			        colorsquare(x, y,sqrsz, cindx, grid, screen, width)
			print "replotting kmt grid with gauss filter..."

		        if grid:
			    #vertical black lines
		            for x in range(0,xres,sqrsz):
		                x = x + sqrsz
		                pygame.draw.line(screen,(0,0,0),(x,0),(x,yres),1)

			    for y in range(0,yres,sqrsz):
		                y = y + sqrsz
		                pygame.draw.line(screen,(0,0,0),(0,y), (xres,y),1)

		        pygame.display.flip()
			enterval = False

# Smooth ocean bottom with a masked nxn stencil average, to keep continental boundaries intact
		if enterval and e.key == 98:
		    kmtboxmask = maskedboxavg(screen, modkmt, xres, yres, sqrsz, grid, width, kmtmax, lats, lons)
		    if kmtboxmask != None:

#			# create a mask to leave land mask same for default below
#			print "Replacing land mask after gaussian blurr..."
#			lm = modkmt[:,:] != 0
#			# apply changes including reapplying original landmask if true
#			modkmt[lm] = kmtgaumask[lm]

			modkmt = kmtboxmask

			for x in range(0,xres,sqrsz):
			    i=x/sqrsz
			    for y in range(0,yres,sqrsz):
			        j=y/sqrsz
			        cindx = int(255*modkmt[j,i]/kmtmax)
			        colorsquare(x, y,sqrsz, cindx, grid, screen, width)
			print "replotting kmt grid with box filter..."

		        if grid:
			    #vertical black lines
		            for x in range(0,xres,sqrsz):
		                x = x + sqrsz
		                pygame.draw.line(screen,(0,0,0),(x,0),(x,yres),1)

			    for y in range(0,yres,sqrsz):
		                y = y + sqrsz
		                pygame.draw.line(screen,(0,0,0),(0,y), (xres,y),1)

		        pygame.display.flip()
			enterval = False

# Smooth ocean bottom with a masked nxn gaussian or weighted box average, to keep continental boundaries intact
		if enterval and e.key == 102:
		    kmtboxmask = maskedfuncavg(screen, modkmt, xres, yres, sqrsz, grid, width, kmtmax, lats, lons)
		    if kmtboxmask != None:

#			# create a mask to leave land mask same for default below
#			print "Replacing land mask after gaussian blurr..."
#			lm = modkmt[:,:] != 0
#			# apply changes including reapplying original landmask if true
#			modkmt[lm] = kmtgaumask[lm]

			modkmt = kmtboxmask

			for x in range(0,xres,sqrsz):
			    i=x/sqrsz
			    for y in range(0,yres,sqrsz):
			        j=y/sqrsz
			        cindx = int(255*modkmt[j,i]/kmtmax)
			        colorsquare(x, y,sqrsz, cindx, grid, screen, width)
			print "replotting kmt grid with box filter..."

		        if grid:
			    #vertical black lines
		            for x in range(0,xres,sqrsz):
		                x = x + sqrsz
		                pygame.draw.line(screen,(0,0,0),(x,0),(x,yres),1)

			    for y in range(0,yres,sqrsz):
		                y = y + sqrsz
		                pygame.draw.line(screen,(0,0,0),(0,y), (xres,y),1)

		        pygame.display.flip()
			enterval = False


		    
	if done:


#           Finish by making sure that the minimum ocean cell has a kmt value of 3 (This is in the default CCSM3 and CCSM4 ocean datasets)

#	    coastmask1 = np.ma.masked_equal(modkmt,1)
#	    coastmask2 = np.ma.masked_equal(modkmt,2)
#            np.putmask(modkmt,coastmask1.mask,3)	
#            np.putmask(modkmt,coastmask2.mask,3)	

	    print >>sys.stdout, 'writing data for', outfile
#           first reverse it before we write
	    kmtnew[:,:] = modkmt[::-1,:] 
	    if curvlin:
	   	 lon[:,:] = ulon
	   	 lat[:,:] = ulat
	    else:
	   	 lon[:] = ulon
	   	 lat[:] = ulat
	    f.close()
#	    same screen image in file
            pygame.image.save(screen,os.path.splitext(outfile)[0]+".png")

            lndmask=np.ma.masked_equal(modkmt,0)
            ocnmask=np.logical_not(lndmask.mask)

            print "Bathymetry min: max: mean: stdev: cmass: \n", meas.minimum(modkmt[ocnmask]),meas.maximum(modkmt[ocnmask]),meas.mean(modkmt[ocnmask]),meas.standard_deviation(modkmt[ocnmask]),meas.center_of_mass(modkmt[ocnmask])

	    break


if __name__ == "__main__":
    main()

