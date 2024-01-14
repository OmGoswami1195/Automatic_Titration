import cv2 #OpenCV - main computer vision engine
import math
import numpy as np
import pafy
import argparse
import sys
import glob
from datetime import datetime

CYAN_D = (106, 21, 70) #Inverted HSV color code of "the darkest pixel" of the fuchsia spot 
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
AREA_THRESHOLD = 1688 #Area of the fuchsia spot in sq. pixels

#Scale frames
FRAME_SCALE = 1.0 #Scale = 0.0...1.0 of frames (smaller value accelerates calculations and reduces resolution of the frame image)
#Calibration
MODE_C = "c" 
MODE_P = "p"
#Calibration mode: grab a frame from the source and save it as a ".jpeg" file
#Production run: based on detected marks determine the volume of dispensed titrant and area of the
#Source of video/image
SOURCE_URL = "url" 
SOURCE_WCAM = "webcam"
#Name of the data file

DATA_FILE_PRE = "V_A_"
DATA_FILE_EXT = ".txt"

#Output image (complete name with the data-time stamp will be specified below automatically): 
IMG_FILE_PRE = "IMG_" #Output image file prefix
IMG_FILE_EXT = ".jpg" #Output image file extension
#Colors of outlining rectangles (RGB) - for visualization only:
REC_VU = (0, 0,255) #Minimal dispensed volume marker (top marker) 
REC_M = (255, 0, 0) #Moving free-floating marker "meniscus"
REC_VL = ( 0,255, 0) #Maximal dispensed volume marker (bottom marker) 
REC_IND = (255,255, 0) #the fuchsia indicator spot
#Information panel on the image (coordinates in pixels and RGB colors) 
L_H = 30 #Line height
L_M = 30 #Left margin of the text
MESSAGE_COLOR = (0,0,0) #Color of the font
LINE_1 = (L_M, L_H) #Position on the image of the 1st line
LINE_2 = (L_M,2*L_H) # --- 
LINE_3 = (L_M,3*L_H) # --- 
LINE_4 = (L_M,4*L_H) # -- 
LINE_5 = (L_M,5*L_H) # -- 
LINE_6 = (L_M,6*L_H) # -- 
LINE_7 = (L_M,7*L_H) # -- 
LINE_8 = (L_M,8*L_H) # -- 
LINE_9 = (L_M,9*L_H) # --

def HSV_HSV(HSV):
    """HSV color code conversion from GIMP/Photoshop to CV2 format""" 
    (H, S, V) = HSV
    return np.array([int(180*H/360),int(255*S/100),int(255*V/100)])
class MRectangle(object):
    """Rectangle coordinates"""
    def __init__(self, xg, yg, wg, hg): 
        self.xg = xg
        self.yg = yg
        self.wg = wg
        self.hg = hg

class Titrator(object):
    def __init__(self):
        """Initialize with current configuration"""
        print("Python:")
        print (sys.version) 
        print (sys.version_info)
        #Generate unique file name for the volume and area date 
        dt = datetime.now() # current date and time
        #Data-time stamp
        DT = dt.strftime("%m%d%Y_%H%M%S")
        #Script arguments:
        ap = argparse.ArgumentParser()
        ap.add_argument("-m", "--mode", help = "mode of the run: calibration or production",default=MODE_P)
        ap.add_argument("-s", "--source", help = "input video stream source", default=SOURCE_URL)
        ap.add_argument("-i", "--input", help = "URL or the video, index, or IP of the webcam") 
        ap.add_argument("-v", "--volume", help = "input initial volume", default=VOL_MIN)
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
            DataFileName = DATA_FILE_PRE+DT+DATA_FILE_EXT 
            self.DataFile = open(DataFileName,"w")
        self.Cap = None
        #Determine source of the video stream
        Source = args["source"]
                         #Source is YouTube
        if ( Source == SOURCE_URL ): 
            self.URL = args["input"]
            video_url = pafy.new(self.URL)
            #Get the highest resolution available 
            video_stream = video_url.getbest(preftype="mp4")
            # video_stream = "/Users/omgoswami/Downloads/June 25, 2021.mp4"
            self.Cap = cv2.VideoCapture() 
            #self.Cap.open(video_stream.url)
            print ("Source: YouTube URL: ",self.URL)
        if ( Source == SOURCE_WCAM ):
            Cam = str(args["input"])
            self.Cap = cv2.VideoCapture() 
            self.Cap.open(Cam)
            print ("Source: IP/remote webcam[",Cam,"]")
        fps = self.Cap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
        #Set desired FPS rate that does not exceeds the stream's fps. Negative fps is not accepted: #self.SkipFrames = int(fps * RATE_OF_FRAME_ANALYSIS)
        print("Video stream: ",fps," FPS")
                         #Calibration data
        print("Calibration Data:")
        print("Treshold area:", AREA_THRESHOLD, " px^2")
        #Initial volume of the titrant
        self.IniVolume = float(args["volume"])
        print("Initial volume of the titrant: ",self.IniVolume," mL")
        #Volumetric calibration equation
        print("Volumetric calibration coefficients:")
        print("CALIB_V_A: ",CALIB_V_A)
        print("CALIB_V_A: ",CALIB_V_B," mL ")
        #Volume of titration solution
        
        
        self.Volume = 0.0
        #Area of the indicator spot 
        self.Area = 0.0
        #Is End-Point Reached?
        self.EndPointReached = False #Initialize parameters of the 
        self.EndPointArea = -1 
        self.EndPointVolume = -1
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

def Frame(self):
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

def Calibration(self, Frame):
    if (self.CalibrationMode): 
        cv2.imwrite(self.ImageFile, Frame)
        return True
    return False

def GetCalibratedVolume(self,VU, VL, VM):
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
        Warning = "Warning: Can't see marker(s)!"
        Volume = 0.0
        return (Warning, Volume)
    else:
        Warning = ""
        #Compute volume based on vectors AM and AB: 
        Volume = (self.VL-self.VU)*(Abs_AM/Abs_AB) #Apply the volumetric calibration equation 
        Volume = Volume*CALIB_V_A+CALIB_V_B
        return (Warning, Volume)

def IdentifyMarker(self, Frame, lower_color, upper_color):
    ImageHSV = cv2.cvtColor(Frame, cv2.COLOR_BGR2HSV)
    Mask_Marker = cv2.inRange (ImageHSV, lower_color, upper_color)
    Cnts_Marker = cv2.findContours(Mask_Marker.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(Cnts_Marker) > 0:
        Max_cnts_marker = max(Cnts_Marker, key=cv2.contourArea) 
        (xg,yg,wg,hg) = cv2.boundingRect(Max_cnts_marker)
    else:
        #Markers were not found, we don't want to crash just proceeed with zeroes 
        return MRectangle(0,0,0,0)
    return MRectangle(xg,yg,wg,hg)

def GetVolume(self, Frame):
    VU = self. IdentifyMarker(Frame, self.TmD, self.TmB) 
    cv2.rectangle(Frame,(VU.xg,VU.yg),(VU.xg+VU.wg, VU.yg+VU.hg),REC_VU,1)
    VL = self. IdentifyMarker(Frame, self.BmD, self.BmB) 
    cv2.rectangle(Frame,(VL.xg,VL.yg),(VL.xg+VL.wg, VL.yg+VL.hg),REC_VL,1)
    VM = self. IdentifyMarker(Frame, self.MmD, self.MmB) 
    cv2.rectangle(Frame,(VM.xg,VM.yg),(VM.xg+VM.wg, VM.yg+VM.hg),REC_M,1)
    (Warning, Volume) = self.GetCalibratedVolume(VU, VL, VM)
    return (Warning, Volume)

def GetArea(self, Frame):
    ImageInvRGB = ~Frame
    #Convert RGB to HSV
    ImageHSV = cv2.cvtColor(ImageInvRGB, cv2.COLOR_BGR2HSV)
    Mask = cv2.inRange (ImageHSV, self.CyanD, self.CyanB)
    #Find contours that have required color (cyanuchsia)
    Cnts = cv2.findContours(Mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2] 
    Flag = False
    #Find contours of spots with fuchsia color
    if len(Cnts)>0:
        Max_Cnt = max(Cnts, key=cv2.contourArea)
        Area = cv2.contourArea(Max_Cnt)
        (xg,yg,wg,hg) = cv2.boundingRect(Max_Cnt) 
        cv2.rectangle(Frame,(xg,yg),(xg+wg, yg+hg),REC_IND,1)
    else:
        Area = 0.0
    return Area

def IsEndPoint(self, Frame, Volume, Area):
    if ( Area >= AREA_THRESHOLD ):
        if (self.EndPointReached == False):
            self.EndPointArea = Area 
            self.EndPointVolume = Volume 
            self.EndPointReached = True
    if (self.EndPointReached):
        self.WriteEndPoint(Frame, self.EndPointVolume, self.EndPointArea)
    return self.EndPointReached

def ImageText(self, Frame, Message, POSITION):
    TypeFace = cv2.FONT_HERSHEY_SIMPLEX 
    TopLeftCornerOfText = POSITION 
    FontColor = MESSAGE_COLOR
    FontScale = 1
    LineType = 2
    cv2.putText(Frame,Message, TopLeftCornerOfText, TypeFace, FontScale, FontColor, LineType)

def WriteWelcome(self,Frame):
    Message = "Press ESC to exit"
    self.ImageText(Frame, Message, LINE_1)

def WriteWarning(self, Frame, Warning):
    self.ImageText(Frame, Warning, LINE_2)

def WriteVolume(self, Frame, Volume):
#Display volume of the dispensed titrant (if negative due to noise then set to zero):
    Vol_Dispensed = max(Volume+VOL_MIN-self.IniVolume, 0.0) 
    MessageVDisp = "Vol. Disp.: "+'{:5.1f}'.format(Vol_Dispensed)+" mL" 
    self.DataFile.write(str(Vol_Dispensed))
    self.DataFile.write("\t") 
    self.ImageText(Frame, "", LINE_3) #Empty line 
    self.ImageText(Frame, MessageVDisp, LINE_4)

def WriteArea(self, Frame, Area):
    Message = "Area: "+'{:5d}'.format(int(Area))+" px^2" 
    self.DataFile.write(str(Area)) 
    self.DataFile.write("\n")
    self.ImageText(Frame, Message, LINE_5)

def WriteEndPoint(self, Frame, Volume, Area):
    Vol_Dispensed = Volume+VOL_MIN-self.IniVolume
    Message_1 = "Vol. Disp.: "+'{:5.1f}'.format(Vol_Dispensed)+" mL"
    Message_2 = "Area:  "+'{:5.1f}'.format(int(Area))+" px^2"
    self.ImageText(Frame, "",LINE_6)
    self.ImageText(Frame, "Endpoint:",LINE_7)
    self.ImageText(Frame, Message_1,LINE_8)
    self.ImageText(Frame,Message_2,LINE_9)
    cv2.imwrite(self.ImageFile, Frame)

def DisplayImage(self, Frame):
    cv2.imshow('Frame',Frame)
    if (cv2.waitKey(1) == 27): 
        return True
    else:
        return False

def Finalize(self):
    if self.Cap:
        self.Cap.release
        cv2.destroyAllWindows()
        if ( not self.CalibrationMode):
            self.DataFile.close()
            print("Done.")
            return 0





def main():
    T = Titrator()
    
    while True:
        ret, Frame = T.Frame()
        
        if (not ret):
            cv2.waitKey(0)
            T.Finalize()
            break
            
        if T.Calibration(Frame):
            break
            
        T.WriteWelcome(Frame)
        
        (Warning, Volume) = T.GetVolume(Frame)
        
        Area = T.GetArea(Frame)
        
        T.WriteWarning(Frame, Warning)
        T.WriteVolume(Frame, Volume)
        T.WriteArea(Frame, Area)
        
        T.IsEndPoint(Frame, Volume, Area)
        
        if T.DisplayImage(Frame):
            break
                
    return 0

if __name__ == "__main__":
    main()
