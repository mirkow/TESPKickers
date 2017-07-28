#!/usr/bin/env python
from calibkinect import depth2xyzuv

from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import Tkinter as Tk

plot_figures = 0
filter_const = 0.15 # filter time constant for speed est filtering [s]

# run_filter - filter parameter
# old_val - previous value of estimate
# val - new estimate
# dt - time difference between estimates
def run_filter(old_val, val, dt):
  global filter_const
  filter_factor = dt / filter_const
  new_val = old_val + (val - old_val) * filter_factor
  return new_val
  
# pick_color - returns color of pixel selected by mouse press
def pick_color(event, x, y, flags, param):
	# grab references to the global variables
	global hsv
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		print x, y, hsv[y,x]

# main - Main processing loop
def main():
  global depth, rgb, hsv, plot_figures
  prev_frame_time = time.time()
  prev_pos = [0, 0, 0]
  prev_speed = [0, 0, 0]
  
  X_pred = 0
  
  pred_hist = []
  
  # define range of red color in HSV
  lower_red0 = np.array([0,140,60])
  upper_red0 = np.array([15,255,255])
  
  lower_red1 = np.array([165,140,60])
  upper_red1 = np.array([180,255,255])

  # define range of blue color in HSV
  lower_blue = np.array([105,120,20])
  upper_blue = np.array([135,255,255])
  
  plt.ion()
  fig = plt.figure()
  
  while True:
      ball_found = 0
      frame_time = time.time()
      dt = frame_time - prev_frame_time
      prev_frame_time = frame_time
      
      #print "dt: ", dt
#    try:
      # Get a fresh frame
      (depth,_), (bgr,_) = get_depth(), get_video()
      
      # get depth in meters
      xyz, uv = depth2xyzuv(depth)
      
      frame_time = time.time()
      
      # Build a two panel color image
      d3 = np.dstack((depth,depth,depth)).astype(np.uint8)
      da = np.hstack((d3,bgr))
      # Simple Downsample
      if plot_figures:
        cv2.imshow('both',np.array(da[::2,::2,::-1]))

      # convert inmage to HSV space for color filtering
      hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
      
      # filter colors
      mask0 = cv2.inRange(hsv, lower_red0, upper_red0)
      mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
      mask2 = cv2.inRange(hsv, lower_blue, upper_blue)

      filt_im = mask0 + mask1 + mask2
      
      filt_im = cv2.erode(filt_im, None, iterations=2)
      filt_im = cv2.dilate(filt_im, None, iterations=2)

      # blur image
      filt_im = cv2.medianBlur(filt_im, 11)
      
      if plot_figures:
        cv2.imshow('filt',filt_im)
      
      # find circles
      circles = cv2.HoughCircles(filt_im,cv2.cv.CV_HOUGH_GRADIENT,1,50,
                        param1=50,param2=14,minRadius=1,maxRadius=100)

      circles_im =  cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
      if circles is not None:
        circles = np.uint16(np.around(circles))

        # filter out identified balls outside of predescribed cuboid
        idx = []
        x = 0
        for i in circles[0,:]:
          if np.count_nonzero((xyz[i[1], i[0],1:2] < 1.5)==False) | np.count_nonzero((xyz[i[1],i[0],1:2] > -1.5)==False) or xyz[i[1], i[0], 2] > 0 or xyz[i[1], i[0], 2] < -3.5:
            idx.append(x)
          x = x + 1
        circles = np.delete(circles, idx, axis=1)
        if np.shape(circles)[1] > 0:
          ball_found = 1
          for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(circles_im,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(circles_im,(i[0],i[1]),2,(0,0,255),3)
          
          # if multiple balls identified, assume that largest is correct
          biggest_circle = np.argmax(circles[0,:,2::3])  
          
          #depth_size = np.shape(depth)
          #l = 1#circles[0,biggest_circle,2]/8
          #if circles[0,biggest_circle,0]-l>0 and circles[0,biggest_circle,0]+l<depth_size[0] and circles[0,biggest_circle,1]-l>0 and circles[0,biggest_circle,1]+l<depth_size[1]:
          #  pos = np.median(xyz[circles[0,biggest_circle,1]-l:circles[0,biggest_circle,1]+l, circles[0,biggest_circle,0]-l:circles[0,biggest_circle,0]+l,2])
          #else:
          
          # get position of identified ball
          pos = xyz[circles[0,biggest_circle,1], circles[0,biggest_circle,0],]
          
          speed = (pos - prev_pos) / dt
          speed[0] = run_filter(prev_speed[0], speed[0], dt)
          speed[1] = run_filter(prev_speed[1], speed[1], dt)
          speed[2] = run_filter(prev_speed[2], speed[2], dt)
          
          prev_pos = list(pos)
          prev_speed = list(speed)
        
          print "pos, speed: ", pos, speed
          
          # if ball is going towards the camera
          if speed[2] > 0:
            X_pred = pos[1] - pos[2] * speed[1]/speed[2]
            print "X pred: ", X_pred
      
      if plot_figures:
        cv2.imshow('blob',circles_im)

      pred_hist.append(X_pred)
      t = np.arange(0, np.shape(pred_hist)[0], 1)
      plt.plot(t, pred_hist)

      plt.xlabel('time step [-]')
      plt.ylabel('x intercept [m]')
      plt.title('Estiamted ball location at goal post')
      plt.grid(True)
      plt.show()
      plt.pause(0.0001)
      
      im_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

      # optical flow
      if 'old_gray' not in locals():
        # Take first frame and find corners in it
        old_gray = im_gray.copy()
        flow_hsv = np.zeros_like(bgr)
        flow_hsv[...,1] = 255
      else:
        # calculate optical flow
        #flow = cv2.calcOpticalFlowFarneback(old_gray, im_gray, 0.5, 3, 15, 3, 5, 1.2, 0)

        #mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        #flow_hsv[...,0] = ang*180/np.pi/2
        #flow_hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        #flow_bgr = cv2.cvtColor(flow_hsv,cv2.COLOR_HSV2BGR)
        #cv2.imshow('frame2', flow_bgr)

        # Now update the previous frame and previous points
        old_gray = im_gray.copy()
      
      cv2.waitKey(1)
#    except:
#      print("Finding cicles failed")
#      time.sleep(1)

# open window for color picker and initialise callback
cv2.namedWindow("filt")
cv2.setMouseCallback("filt", pick_color)

main()

 
"""
IPython usage:
 ipython
 [1]: run -i demo_freenect
 #<ctrl -c>  (to interrupt the loop)
 [2]: %timeit -n100 get_depth(), get_rgb() # profile the kinect capture

"""

