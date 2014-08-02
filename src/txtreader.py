#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os, re
import numpy as np
import scipy.io, scipy.sparse
import logging

logger = logging.getLogger(__name__)


class Txtreader():
  """ Class to read files and put data in arrays or other objects
  """
  def __init__(self, infile=None):
    
    self.infile = infile
        
    
  def readcols(self, skipline=1, readupto=0, dtype='float64'):
    'read file with data arranged in columns'
    infile = self.infile
    logger.debug("File:-> %s", infile)
    try: 
        INPF = open(infile, 'r')
    except IOError:
        print "file", infile, " does not exist"
        raise SystemExit("Now Exiting")
    cols = []
    lineno, lines_read = 0, 0
    for line in INPF:
        if re.match("@|#|&", line):
            continue
        if (lineno % skipline == 0):
            line = line.split()
    	# line=map(float,line.split())
            if len(line) < 1:
                logger.warn('empty line detected ... skipping')
                continue 
                
            if (len(line) > 0) and (lines_read == 0):
        	logger.debug("Detecting %s columns",len(line))
        	ncols=len(line)
        
            lines_read += 1
            
    	    if (len(line) != ncols):
    		
    		logger.warn("%s\n %s",infile,line)
    		logger.warn("Detecting %s column, expecting %s ...skipping line %s",len(line), ncols, lines_read)
        	continue
        	
            if ((readupto > 0) and (readupto == lines_read)): break
            cols.append(line)
        lineno += 1
    self.cols = np.array(cols, dtype=dtype)
    logger.debug('File %s, Lines read: %d', infile, lines_read)

  def readformat(self, infile, cols_type, read_nline=1, readupto=0):
    """ read files with format specified in cols_type and return numpy array
    """
    print "file is", infile
    try: 
        INPF = open(infile, 'r')
    except IOError:
        print "file", infile, " does not exist"
        exit(2)
    tcols = []
    lineno, lines_read = 0, 0
    for line in INPF:
        if re.match("@|#|&", line):
            continue
        # line=map(float,line.split())
        if (lineno % read_nline == 0):
            line = line.split()
            if len(line) < 1:
                print 'empty line detected ... skipping'
                continue 
            lines_read += 1
            if ((readupto > 0) and (readupto == lines_read)): break
            tcols.append(line)
        lineno += 1
    cols = np.zeros(len(tcols), cols_type)
    i = 0
    for entr in tcols:
      if (len(entr) > 1):
        entr = tuple(entr)
      else: entr = entr[0]
      cols[i] = entr
      i = i + 1
    self.cols = cols
    # return cols
    print 'Lines read: ', lines_read
    
  def readfiles(self, infiles, colno=1, skipline=1, readupto=0):
    ''' Read all the files in infiles and present them as list'''
    row1, col1 = 0, 0
    for i, file in enumerate(infiles):
      self.infile = file
      self.readcols(skipline=skipline, readupto=readupto)
      row, col = self.cols.shape
      if i == 0: 
          row1, col1 = row, col
          data = np.zeros((row, len(infiles)))
          x=self.cols[:,0]
      if (row1 == row) and (col1 == col): 
          data[:, i] = self.cols[:, colno]
      else:
        print 'shape mismatch from 1st file !!'
        print 'BEFORE', row1, col1, 'now ', row, col, 'skipping file', file
    return x,data
  

  def join_ncols(self, infiles, col, outp):
    ''' take the specified  col from each file and put it in the output
    in same sequence as of infiles'''
    try: OF = open(outp, 'w')
    except IOError: print "Cannot open file for output"
    inp1 = readF()
    inp1.read_files(infiles)
    rows, cols = inp1.files[0].cols.shape
    for i in range(rows):
      OF.write("%g\t" % inp1.files[0].cols[i, 0])
      for j in range(len(infiles)):
          OF.write("%g\t" % inp1.files[j].cols[i, col])
      OF.write("\n")
    OF.close()
	
  def read_matrices(self, infiles):
    ''' Read all the matrices in infiles and present them as list'''
    mats = []
    for i, file in enumerate(infiles):
      a = scipy.io.mmread(file)
      mats.append(a)
    self.mats = mats
    

