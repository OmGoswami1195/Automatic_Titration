#& C:/ProgramData/anaconda3/python.exe c:/Users/user1/titrator_pub.py -s file -f "C:\\Users\\user1\\Downloads\\June 25, 2021.mp4"
# python3 testing_frames.py -s file -f "/home/soumoroy/Downloads/June 25, 2021.mp4"
import cv2 #OpenCV - main computer vision engine
import math
import numpy as np
import pafy
import argparse
import sys
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tkinter import *
import logging
import smtplib
from email.mime.text import MIMEText

#Calibration of the phenolphthalein solution color at high pH (NaOH solution at 0.1M):

CYAN_D = (106, 21, 70)  #Inverted HSV color code of "the darkest pixel" of the fuchsia spot
CYAN_B = (182, 98, 100) #Inverted HSV color code of "the brightest pixel" of the fuchsia spot

#Marker at the minimal dispensed volume (top of the burette):
T_MARKER_D = (84, 11, 20) #"the darkest pixel"
T_MARKER_B = (99, 99, 99) #"the brightest pixel"
VOL_MIN = 31.0 #Minimal volume in mL as read from the burette

#Moving free-floating marker "meniscus" (in the middle of the burette)
M_MARKER_D = (50, 55, 60) #"the darkest pixel"
M_MARKER_B = (60, 99, 99) #"the brightest pixel"

#Marker at the maximal dispensed volume (bottom of the burette):
B_MARKER_D = (217, 11, 20) #"the darkest pixel"
B_MARKER_B = (242, 99, 99) #"the brightest pixel"
VOL_MAX = 50.0 #Maximal volume in mL as read from the burette

#Volumetric calibration equation (from EXCEL 'trend line'):
#V = 0.9960 Va + 0.0377
#Calibration run (Calibration-1):
CALIB_V_A = 0.9960
CALIB_V_B = 0.0377

#Measured area of the fuchsia spot (indicator)
AREA_THRESHOLD = 1688  #Area of the fuchsia spot in sq. pixels
#===============================================================================

#Scale frames 
FRAME_SCALE = 1.0 #Scale = 0.0...1.0 of frames (smaller value accelerates calculations and reduces resolution of the frame image)


#Calibration and production modes:
MODE_C = "c" #Calibration mode: grab a frame from the source and save it as a ".jpeg" file
MODE_P = "p" #Production run: based on detected marks determine the volume of dispensed titrant and area of the fuchsia spot 

#Source of video/image
SOURCE_URL  = "url"    #read frames from YouTube URL stream
SOURCE_WCAM = "webcam" #read frames from video streams over Wi-Fi (IP and path of the webcam are needed), e.g. #DroidApp
SOURCE_FILE = "file"   # read frames from a local video file
#Name of the data file to store volume and area data (complete name with the data-time stamp will be specified #below automatically):
DATA_FILE_PRE = "V_A_" #Data file prefix
DATA_FILE_EXT = ".txt" #Data file extension
#Output image (complete name with the data-time stamp will be specified below automatically):
IMG_FILE_PRE = "IMG_" #Output image file prefix
IMG_FILE_EXT = ".jpg" #Output image file extension

#Colors of outlining rectangles (RGB) - for visualization only:
REC_VU =  (0,    0,255) #Minimal dispensed volume marker (top marker)
REC_M  =  (255,  0,  0) #Moving free-floating marker "meniscus" 
REC_VL =  (  0,255,  0) #Maximal dispensed volume marker (bottom marker)
REC_IND = (255,255,  0) #the fuchsia indicator spot

#Information panel on the image (coordinates in pixels and RGB colors)
L_H = 30 #Line height
L_M = 30 #Left margin of the text

MESSAGE_COLOR = (0,0,0) #Color of the font

LINE_1 = (L_M,  L_H) #Position on the image of the 1st line 
LINE_2 = (L_M,2*L_H) # ---  2nd line
LINE_3 = (L_M,3*L_H) # ---  3rd line
LINE_4 = (L_M,4*L_H) # --   4th line
LINE_5 = (L_M,5*L_H) # --   5th line
LINE_6 = (L_M,6*L_H) # --   6th line
LINE_7 = (L_M,7*L_H) # --   7th line
LINE_8 = (L_M,8*L_H) # --   8th line
LINE_9 = (L_M,9*L_H) # --   9th line
LINE_10 = (L_M,10*L_H) # --   10th line
LINE_11 = (L_M,11*L_H) # --   11th line
LINE_12 = (L_M,12*L_H) # --   12th line
log_file = "titrator_log.txt"
logging.basicConfig(filename=log_file, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

#===============================================================================
def HSV_HSV(HSV):
	"""HSV color code conversion from GIMP/Photoshop to CV2 format"""
	(H, S, V) = HSV
	return np.array([int(180*H/360),int(255*S/100),int(255*V/100)])
#===============================================================================
class MRectangle(object):
	"""Rectangle coordinates"""
	def __init__(self, xg, yg, wg, hg):
		self.xg = xg
		self.yg = yg
		self.wg = wg
		self.hg = hg
#===============================================================================
class Titrator(object):
	"""Class for computer vision based automatic titrator"""
#-------------------------------------------------------------------------------
	def __init__(self):
		"""Initialize with current configuration"""

		print("Python:")
		print (sys.version)
		print (sys.version_info)
		
		#Generate unique file name for the volume and area date
		dt = datetime.now() # current date and time
		#Data-time stamp
		DT = dt.strftime("%m%d%Y_%H%M%S")
		self.logger = logging.getLogger("titrator")

		#Script arguments:
		ap = argparse.ArgumentParser()
		ap.add_argument("-m", "--mode", help = "mode of the run: calibration or production", default=MODE_P)
		ap.add_argument("-s", "--source", help = "input video stream source", default=SOURCE_URL)
		ap.add_argument("-i", "--input",  help = "URL or the video, index, or IP of the webcam")
		ap.add_argument("-v", "--volume", help = "input initial volume", default=VOL_MIN)
		ap.add_argument("-f", "--file", help="path to the video file")
		args = vars(ap.parse_args())

		#Determine mode of the run:
		Mode = args["mode"]

		self.CalibrationMode = False
		if ( Mode == MODE_C ):
			self.CalibrationMode = True

		#Name of the output image file
		self.ImageFile = IMG_FILE_PRE+DT+IMG_FILE_EXT
		#File to save volume and area (if it is not a calibration run)
		
		if ( not self.CalibrationMode):
			DataFileName  = DATA_FILE_PRE+DT+DATA_FILE_EXT
			self.DataFile = open(DataFileName,"w")

		self.Cap = None

		#Determine source of the video stream
		Source = args["source"]

		#Source is YouTube


		if  (Source == SOURCE_URL):
			self.URL = args["input"]
			video_url = pafy.new(self.URL)
			#Get the highest resolution available
			video_stream = video_url.getbest(preftype="mp4")
			self.Cap = cv2.VideoCapture()
			self.Cap.open(video_stream.url)
			print ("Source: YouTube URL: ",self.URL)

		#Source is a stream from a remote webcam (e.g. identified by the IP in the Wi-Fi network)
		#For example, the cam can be addressed as: e.g. "http://192.168.0.120:8080/videofeed" or #"http://10.78.2.14:4747/video"
		if ( Source == SOURCE_WCAM ):
			Cam = str(args["input"])
			self.Cap = cv2.VideoCapture()
			self.Cap.open(Cam)
			print ("Source: IP/remote webcam[",Cam,"]")
		if Source == SOURCE_FILE:
			video_file_path = args["file"]
			self.Cap = cv2.VideoCapture(video_file_path)
			print("Source: Local Video File: ", video_file_path)


		fps = self.Cap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
        
		#Set desired FPS rate that does not exceeds the stream's fps. Negative fps is not accepted:
		#self.SkipFrames = int(fps * 5)
		print("Video stream: ",fps," FPS") 

		#Calibration data
		print("Calibration Data:")
		print("Threshold area:", AREA_THRESHOLD, " px^2")
		
		#Initial volume of the titrant
		self.IniVolume = float(args["volume"])
		print("Initial volume of the titrant: ",self.IniVolume," mL")
				
		#Volumetric calibration equation
		print("Volumetric calibration coefficients:")
		print("CALIB_V_A: ",CALIB_V_A ," mL ")
		print("CALIB_V_A: ",CALIB_V_B ," mL ")
		self.fig, self.ax = plt.subplots()
		self.line_volume, = self.ax.plot([], [], label="Volume (mL)")
		self.line_area, = self.ax.plot([], [], label="Area (px^2)")
		self.ax.legend()
		self.ani = FuncAnimation(self.fig, self.update_plot, frames=range(10),interval=1000)  # Update every 1000 milliseconds
		self.x_data = []
		self.volume_data = []
		self.area_data = []

		
		#Volume of titration solution 
		self.Volume = 0.0
		#Area of the indicator spot
		self.Area   = 0.0
		
		#Is End-Point Reached?
		self.EndPointReached = False
		#Initialize parameters of the end-point
		self.EndPointArea = -1
		self.EndPointVolume = -1

		#Convertion of HSV colors to CV2 format:
		self.CyanD = HSV_HSV(CYAN_D)
		self.CyanB = HSV_HSV(CYAN_B)
		
		self.TmD = HSV_HSV(T_MARKER_D)
		self.TmB = HSV_HSV(T_MARKER_B)
		
		self.MmD = HSV_HSV(M_MARKER_D)
		self.MmB = HSV_HSV(M_MARKER_B)
		
		self.BmD = HSV_HSV(B_MARKER_D)
		self.BmB = HSV_HSV(B_MARKER_B)

		#Specify minimal and maximal volume marks:
		self.VU = VOL_MIN
		self.VL = VOL_MAX
		return
#-------------------------------------------------------------------------------
	def Frame(self):
		"""Obtain frame Cam - index of webcams: Cam== 0 - built-in web cam, Cam == 1 - Droid cam"""
		# Webcam frames:

		frame = None
		ret = True

		ret, frame = self.Cap.read()
		if (not ret):
			print("Can't receive the next frame from the video stream...")
			return ret, frame

		if (FRAME_SCALE != 1.0):
			(h, w) = frame.shape[:2]
			frame = cv2.resize(frame, ( int(w*FRAME_SCALE), int(h*FRAME_SCALE)))

		return ret, frame
#-------------------------------------------------------------------------------
	def Calibration(self, Frame):
		"""Calibration mode: just store one image to calibrate colors of labels/markers"""
		
		if (self.CalibrationMode):
			cv2.imwrite(self.ImageFile, Frame)
			self.logger.info("Calibration image saved.")
			return True
		return False
#-------------------------------------------------------------------------------
	def SendEmailNotification(self, subject, message):
		sender_email = "r.rohanraj.2001@gmail.com"  # Replace with your email
		receiver_email = "rohan.raj2021@vitstudent.ac.in"  # Replace with recipient's email
		password = "Jaihind@17"  # Replace with your email password

        # Create the email message
		msg = MIMEText(message)
		msg["Subject"] = subject
		msg["From"] = sender_email
		msg["To"] = receiver_email
		try:
			# Connect to the SMTP server
			server = smtplib.SMTP("smtp.gmail.com", 587)
			server.starttls()

            # Login to your email account
			server.login(sender_email, password)

            # Send the email
			server.sendmail(sender_email, receiver_email, msg.as_string())
			print("Email notification sent successfully!")
		except Exception as e:
			print("Error sending email notification:", str(e))
		finally:
            # Disconnect from the SMTP server
			server.quit()


	
	def	GetCalibratedVolume(self,VU, VL, VM):
		
		#Minimal dispensed volume (top) mark:
		Ax = VU.xg+VU.wg/2
		Ay = VU.yg+VU.hg
		
		#Moving (free-floating) mark:
		Mx = VM.xg+VM.wg/2
		#Bottom of the free-floating marker:
		My = VM.yg+VM.hg

		#Maximal dispensed volume (bottom) mark:
		Bx = VL.xg+VL.wg/2
		By = VL.yg

		Abs_AB = math.sqrt( (Bx-Ax)**2+(By-Ay)**2 )
		Abs_AM = math.sqrt( (Mx-Ax)**2+(My-Ay)**2 )
		if ( (Abs_AB == 0.0) or (Abs_AM == 0.0) ):
			#If labels are incorrectly identified - just proceed to the next frame with 0 volume
			Warning = "Warning: Can't see marker(s)!"
			Volume = 0.0
			return (Warning, Volume)
		else:
			Warning = ""
			#Compute volume based on vectors AM and AB:
			Volume = (self.VL-self.VU)*(Abs_AM/Abs_AB)
			#Apply the volumetric calibration equation
			Volume = Volume*CALIB_V_A+CALIB_V_B
			return (Warning, Volume)
#-------------------------------------------------------------------------------
	def	IdentifyMarker(self, Frame, lower_color, upper_color):
		"""Identify the largest marker of the specified color range with the rectangle"""
		# Convert BGR to HSV
		ImageHSV = cv2.cvtColor(Frame, cv2.COLOR_BGR2HSV)
		Mask_Marker = cv2.inRange (ImageHSV, lower_color, upper_color)
		Cnts_Marker = cv2.findContours(Mask_Marker.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
		if len(Cnts_Marker) > 0:
			Max_cnts_marker = max(Cnts_Marker, key=cv2.contourArea)
			(xg,yg,wg,hg) = cv2.boundingRect(Max_cnts_marker)
		else: #Markers were not found, we don't want to crash just proceeed with zeroes
			return MRectangle(0,0,0,0)
			
		return MRectangle(xg,yg,wg,hg)
#-------------------------------------------------------------------------------
	def GetVolume(self, Frame):
		"""Get volume from the frame analysis"""

		#Minimal volume (top) marker (VU)
		VU = self. IdentifyMarker(Frame, self.TmD, self.TmB)
		cv2.rectangle(Frame,(VU.xg,VU.yg),(VU.xg+VU.wg, VU.yg+VU.hg),REC_VU,1)

		#Maximal volume (bottom) marker (VL)
		VL = self. IdentifyMarker(Frame, self.BmD, self.BmB)
		cv2.rectangle(Frame,(VL.xg,VL.yg),(VL.xg+VL.wg, VL.yg+VL.hg),REC_VL,1)

		#Moving (free-floating) marker "meniscus"
		VM = self. IdentifyMarker(Frame, self.MmD, self.MmB)
		cv2.rectangle(Frame,(VM.xg,VM.yg),(VM.xg+VM.wg, VM.yg+VM.hg),REC_M,1)

		(Warning, Volume) = self.GetCalibratedVolume(VU, VL, VM)

		return (Warning, Volume)
#-------------------------------------------------------------------------------
	def GetArea(self, Frame):
		"""Get the area of the indicator spot from the frame analysis"""

		#Invert color fuchsia->cyan (malachite green)
		ImageInvRGB = ~Frame
		#Convert RGB to HSV
		ImageHSV = cv2.cvtColor(ImageInvRGB, cv2.COLOR_BGR2HSV)
		#Find the colors within the specified boundaries and apply the mask
		Mask = cv2.inRange (ImageHSV, self.CyanD, self.CyanB)
		#Find contours that have required color (cyanuchsia) 
		Cnts = cv2.findContours(Mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
		Flag = False
		#Find contours of spots with fuchsia color 
		if len(Cnts)>0:
			#Find the largest contour based on its area
			Max_Cnt = max(Cnts, key=cv2.contourArea)
			Area = cv2.contourArea(Max_Cnt)
			(xg,yg,wg,hg) = cv2.boundingRect(Max_Cnt)
			cv2.rectangle(Frame,(xg,yg),(xg+wg, yg+hg),REC_IND,1)
		else:
			Area = 0.0

		return Area
#-------------------------------------------------------------------------------
	def IsEndPoint(self, Frame, Volume, Area):
		"""Determine if the endpoint of titration is reached"""

		if ( Area >= AREA_THRESHOLD ):
			if (self.EndPointReached == False):
				self.EndPointArea    = Area
				self.EndPointVolume  = Volume
				self.EndPointReached = True
				# Send email notification when endpoint is reached
				subject = "Titrator: Endpoint Reached"
				message = f"The titration is complete. Endpoint reached with volume: {Volume} mL and area: {Area} px^2."
				self.SendEmailNotification(subject, message)
				

		if (self.EndPointReached):
			self.WriteEndPoint(Frame, self.EndPointVolume, self.EndPointArea)

		return self.EndPointReached
#-------------------------------------------------------------------------------
	def ImageText(self, Frame, Message, POSITION):
		TypeFace = cv2.FONT_HERSHEY_SIMPLEX
		TopLeftCornerOfText = POSITION
		FontColor = MESSAGE_COLOR
		FontScale = 1
		LineType  = 2
		cv2.putText(Frame,Message, TopLeftCornerOfText, TypeFace, FontScale, FontColor, LineType)
#-------------------------------------------------------------------------------
	def WriteWelcome(self,Frame):
		"""Print welcome message"""
		Message = "Press ESC to exit"
		self.ImageText(Frame, Message, LINE_1)
#-------------------------------------------------------------------------------
	def WriteWarning(self, Frame, Warning):
		"""Print warning message"""
		Message_original= "Original soln:10 ml and molarity:0.1 "
		self.ImageText(Frame, Message_original, LINE_2)
#-------------------------------------------------------------------------------
	def WriteVolume(self, Frame, Volume):
		"""Display volume"""
		#Display volume of the dispensed titrant (if negative due to noise then set to zero):
		Vol_Dispensed = max(Volume+VOL_MIN-self.IniVolume, 0.0)
		MessageVDisp = "Vol. Disp.: "+'{:5.3f}'.format(Vol_Dispensed)+" mL"
		self.DataFile.write(str(Vol_Dispensed))
		self.DataFile.write("\t")
		self.ImageText(Frame, "", LINE_3) #Empty line
		self.ImageText(Frame, MessageVDisp, LINE_4)
#-------------------------------------------------------------------------------
	def WriteArea(self, Frame, Area):
		"""Display area"""
		Message = "Area:   "+'{:5d}'.format(int(Area))+" px^2"
		self.DataFile.write(str(Area))
		self.DataFile.write("\n")
		self.ImageText(Frame, Message, LINE_5)
#-------------------------------------------------------------------------------
	def WriteEndPoint(self, Frame, Volume, Area):
		"""Display titration completed message"""
		Vol_Dispensed = Volume+VOL_MIN-self.IniVolume
		Message_1 = "Vol. Disp.: "+'{:5.3f}'.format(Vol_Dispensed)+" mL"
		Message_2 = "Area:    "+'{:5d}'.format(int(Area))+" px^2"
		self.ImageText(Frame, "", LINE_6) #Empty line
		self.ImageText(Frame, "Endpoint:", LINE_7)
		self.ImageText(Frame, Message_1, LINE_8)
		self.ImageText(Frame, Message_2, LINE_9)
		
		"""Calculation of the molarity"""
		N1=0.1  # change as per soln
		V1=10	# change as per soln
		V2=Volume+VOL_MIN-self.IniVolume
		N2=(N1*V1)/V2
		Message_mol="Molarity of soln:"+'{:5.3f}'.format(N2)
		self.ImageText(Frame, "", LINE_10) #Empty line
		self.ImageText(Frame,"Calculation:",LINE_11)
		self.ImageText(Frame,Message_mol,LINE_12)
		cv2.imwrite(self.ImageFile, Frame)
#--------------------------------------------------------------------------------
	def DisplayImage(self, Frame):
		cv2.imshow('Frame',Frame)
		self.x_data.append(datetime.now())
		self.volume_data.append(self.Volume)
		self.area_data.append(self.Area)
		plt.pause(0.001)
		plt.ion()  # Turn on interactive mode
		plt.show()  # Show the plot
		if (cv2.waitKey(1) == 27):
			self.logger.info("User pressed ESC to exit.")
			return True
		else:
			return False
#-------------------------------------------------------------------------------
	def update_plot(self, frame):
		# Measure volume of the titration solution
		(warning, volume) = self.GetVolume(self.Frame()[1])
		# Measure area of the colored indicator spot
		area = self.GetArea(self.Frame()[1])

        # Append data to arrays
		self.x_data.append(datetime.now())
		self.volume_data.append(volume)
		self.area_data.append(area)

        # Update the plot
		self.line_volume.set_data(self.x_data, self.volume_data)
		self.line_area.set_data(self.x_data, self.area_data)
		self.ax.relim()
		self.ax.autoscale_view()
		return self.line_volume, self.line_area


	
	
	
	
	def Finalize(self):
		if self.Cap:
			self.Cap.release()
		cv2.destroyAllWindows()
		plt.close(self.fig)
		if ( not self.CalibrationMode):
			self.DataFile.close()
			self.logger.info("Data file closed.")
		self.logger.info("Titrator finalized.")

		print("Done.")
		return 0
#===============================================================================
def main():
	#Initialize the titrator
	T = Titrator()
	T.logger.info("Titrator initialized.")

	while True:
		#Scan video input or read an image from the file
		ret, Frame = T.Frame()

		#Stop and wait for user input if no frames
		if (not ret):
			#Release all resources
			cv2.waitKey(0)
			T.Finalize()
			T.logger.info("Exiting due to no frames.")
			break

		#Calibration mode (no measurements is performed)
		if T.Calibration(Frame):
			T.logger.info("Exiting calibration mode.")
			break

		T.WriteWelcome(Frame)

		#Measure volume of the titration solution
		(Warning, Volume) = T.GetVolume(Frame)
		#Measure area of the colored indicator spot 
		Area = T.GetArea(Frame)

		#Display parameters
		T.WriteWarning(Frame, Warning)
		T.WriteVolume(Frame, Volume)
		T.WriteArea(Frame, Area)

		#Is the end-point reached?
		T.IsEndPoint(Frame, Volume, Area)

		#Display the titrator with recognized objects
		if T.DisplayImage(Frame):
			T.logger.info("User exited the program.")
			break

	return 0
#===============================================================================
#Call the main program only if the script is called directly. Not through the 'import' statement
if __name__ == "__main__":
	main()
