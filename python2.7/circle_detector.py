#!/usr/bin/env python
from calibkinect import depth2xyzuv

from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import freenect
import cv2
import numpy as np
import time

# freenect.freenect_init
plot_figures = 1
plot_traj = 0
use_hough = 0
filter_const = 0.1 # filter time constant for speed est filtering [s]
filter_speed = 1
filter_pos = 1

if plot_traj:
  import matplotlib.pyplot as plt

# run_filter - filter parameter
# old_val - previous value of estimate
# val - new estimate
# dt - time difference between estimates
def run_filter(old_val, val, dt):
  global filter_const
  filter_factor = dt / filter_const
  if filter_factor > 1:
    filter_factor = 1
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

class circle_detector(object):
  def __init__(self):
      self.prev_pos = [0, 0, 0]
      self.prev_speed = [0, 0, 0]
      self.prev_frame_time = time.time()
      self.X_pred = 0
      print "initialising detector"



  # main - Main processing loop
  def detect(self):
    global depth, rgb, hsv, plot_figures, use_hough

    pred_hist = []

    # define range of red color in HSV
    lower_red0 = np.array([0,120,80])
    upper_red0 = np.array([15,255,255])

    lower_red1 = np.array([165,120,80])
    upper_red1 = np.array([180,255,255])

    # define range of blue color in HSV
    lower_blue = np.array([105,125,20])
    upper_blue = np.array([135,255,255])

    xcal = 28
    xcalz = -8
    ycal = -15
    pos = None
    if plot_traj:
      plt.ion()
      fig = plt.figure()

    ball_found = 0

    #print "dt: ", dt
#    try:
    # Get a fresh frame
    (depth,_), (bgr,_) = get_depth(), get_video()

    # Simple Downsample
    if plot_figures:
      # Build a two panel color image
      d3 = np.dstack((depth,depth,depth)).astype(np.uint8)
      da = np.hstack((d3,bgr))
      d3[:,320+xcal:320+xcal+2,0:3] = 0
      #cv2.imshow('both',np.array(da[::2,::2,::-1]))

    # convert inmage to HSV space for color filtering
    hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)

    # filter colors
    mask0 = cv2.inRange(hsv, lower_red0, upper_red0)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)

    filt_im = mask0 + mask1# + mask2

    filt_im = cv2.erode(filt_im, None, iterations=2)
    filt_im = cv2.dilate(filt_im, None, iterations=2)

    # blur image
    filt_im = cv2.medianBlur(filt_im, 11)
    filt_im = cv2.Canny(filt_im,100,200)

    if 0:#plot_figures:
      cv2.imshow('filt',filt_im)

    circles = None
    # find circles
    if use_hough:
      circles = cv2.HoughCircles(filt_im,
                     cv2.cv.CV_HOUGH_GRADIENT,1,25,
                     param1=200,param2=10,minRadius=1,maxRadius=50)
    else:
      # find contours in the mask and initialize the current
      # (x, y) center of the ball
      cnts = cv2.findContours(filt_im.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]

      # only proceed if at least one contour was found
      if len(cnts) > 0:
        for c in cnts:
          ((x, y), radius) = cv2.minEnclosingCircle(c)

          # only proceed if the radius meets a minimum size
          if radius > 5 and radius < 80:
            M = cv2.moments(c)
            if M["m00"] > 0:
              center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
              temp = [[[center[0], center[1], int(radius)]]]
            else:
              temp = [[[int(x), int(y), int(radius)]]]

            if circles is None:
              circles = temp
            else:
              circles = np.concatenate((circles, temp), axis=1)
    if plot_figures:
      circles_im =  cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if circles is not None:
      circles = np.uint16(np.around(circles))
      # filter out identified balls outside of predescribed cuboid
      # idx = []
      # x = 0
      # for i in circles[0,:]:
      #   if np.count_nonzero((xyz[i[1], i[0],1:2] < 1.5)==False) | np.count_nonzero((xyz[i[1],i[0],1:2] > -1.5)==False) or xyz[i[1], i[0], 2] > 0 or xyz[i[1], i[0], 2] < -3.5:
      #     idx.append(x)
      #     x = x + 1
      #   circles = np.delete(circles, idx, axis=1)

      if np.shape(circles)[1] > 0:
        num_circles = (np.shape(circles))[1]
        for circle_n in np.arange(num_circles):
            # if multiple balls identified, assume that largest is correct
            biggest_circle = np.argmax(circles[0,:,2::3])
            circle = circles[0, biggest_circle]

            #depth_size = np.shape(depth)
            #l = 1#circles[0,biggest_circle,2]/8
            #if circles[0,biggest_circle,0]-l>0 and circles[0,biggest_circle,0]+l<depth_size[0] and circles[0,biggest_circle,1]-l>0 and circles[0,biggest_circle,1]+l<depth_size[1]:
            #  pos = np.median(xyz[circles[0,biggest_circle,1]-l:circles[0,biggest_circle,1]+l, circles[0,biggest_circle,0]-l:circles[0,biggest_circle,0]+l,2])
            #else:

            #print circles[0,biggest_circle,0], circles[0,biggest_circle,1]
            x, y = circles[0,biggest_circle,0] + xcal, circles[0,biggest_circle,1] + ycal

            x = np.uint16(np.around(x  + (x - 640/2 - 1) * 0.1) + (0.8 + self.prev_pos[2]) * xcalz)
            y = np.uint16(np.around(y + (y - 480/2 - 1) * 0.07))

            # average distance over area of half the radius of the ball
            w = 5
            if w >= circles[0,biggest_circle,2] / 5:
              w = circles[0,biggest_circle,2] / 5 + 1

            # get position of identified ball in meters
            if x >= w and x < 640-w and y >= w and y < 480-w:
              u,v = np.mgrid[y-w:y+w+1,x-w:x+w+1]
              xyz = depth2xyzuv(depth[y-w:y+w+1,x-w:x+w+1], u, v)

              if plot_figures:
                cv2.circle(d3,(x,y),2,(0,0,255),3)

              # get average values of x,y,z
              sum0 = 0
              sum1 = 0
              sum2 = 0
              num = 0
              for i in np.arange(0,w+1):
                for j in np.arange(0,w+1):
                  if xyz[i,j,2] < 0:
                    #print xyz[i,j,:]
                    sum0 = sum0 + xyz[i,j,0]
                    sum1 = sum1 + xyz[i,j,1]
                    sum2 = sum2 + xyz[i,j,2]
                    num = num + 1
              if num > 0:
                pos = np.array([sum0/num, sum1/num,sum2/num])
                if pos[1] < -0.6 or pos[1] > 0.6 or pos[0] > 0.3 or pos[0] < -0.35 or pos[2] > -0.5 or pos[2] < -2.2:
                    pos = None
                    circles = np.delete(circles, biggest_circle, axis=1)
                else:
                    if plot_figures:
                        # draw the outer circle
                        cv2.circle(circles_im,(circle[0],circle[1]),circle[2],(0,255,0),2)
                        # draw the center of the circle
                        cv2.circle(circles_im,(circle[0],circle[1]),2,(0,0,255),3)
                    break

          # we need at least one good value to estimate the position
        if pos is not None:
            # bookkeeping for time
            frame_time = time.time()
            dt = frame_time - self.prev_frame_time

            if filter_pos:
              pos[0] = run_filter(self.prev_pos[0], pos[0], dt)
              pos[1] = run_filter(self.prev_pos[1], pos[1], dt)
              pos[2] = run_filter(self.prev_pos[2], pos[2], dt)

            speed = (pos - self.prev_pos) / dt
            if filter_speed:
              speed[0] = run_filter(self.prev_speed[0], speed[0], dt)
              speed[1] = run_filter(self.prev_speed[1], speed[1], dt)
              speed[2] = run_filter(self.prev_speed[2], speed[2], dt)

            # if ball is going towards the camera
            if speed[2] >= 0.05:
              self.X_pred = pos[1] - pos[2] * speed[1]/speed[2]
              print "X pred: ", self.X_pred, speed, dt, pos
            else:
                self.X_pred = 0
            self.prev_pos = np.array(pos)
            self.prev_speed = np.array(speed)
            self.prev_frame_time = frame_time

        if plot_figures:
          circles_im[:,320,0:2] = 0
          cv2.imshow('blob',circles_im)
          cv2.imshow('both',np.array(d3[:,:,::-1]))

        if plot_traj:
          pred_hist.append(self.X_pred)
          t = np.arange(0, np.shape(pred_hist)[0], 1)
          plt.plot(t, pred_hist)

          plt.xlabel('time step [-]')
          plt.ylabel('x intercept [m]')
          plt.title('Estiamted ball location at goal post')
          plt.grid(True)
          plt.show()
          plt.pause(0.0001)

        if plot_figures:
          cv2.waitKey(1)
    return (self.X_pred, pos)

  #    except:
  #      print("Finding cicles failed")
  #      time.sleep(1)

  # open window for color picker and initialise callback
  # cv2.namedWindow("filt")
  # cv2.setMouseCallback("filt", pick_color)

  # def

# main()

 
"""
IPython usage:
 ipython
 [1]: run -i demo_freenect
 #<ctrl -c>  (to interrupt the loop)
 [2]: %timeit -n100 get_depth(), get_rgb() # profile the kinect capture

"""

