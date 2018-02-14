import cv2
import numpy
from os.path import join
data_analysis = open('np.csv', 'a')
g = open("track.txt","w+")
f = open("NOSE.csv","w+")
def show_image_with_data(frame, blinks, landblinks, irises, window, err=None):
    """
    Helper function to draw points on eyes and display frame
    :param frame: image to draw on
    :param blinks: number of blinks
    :param window: for window dimension FW: added window for obtaining the window dimension
    :param irises: array of points with coordinates of irises
    :param err: for displaying current error in Lucas-Kanade tracker
    :return:
    """
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = window.get(cv2.CAP_PROP_FRAME_WIDTH) # float FW
    height = window.get(cv2.CAP_PROP_FRAME_HEIGHT) # float FW
    if err:
        cv2.putText(frame, str(err), (20, 450), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'blinks: ' + str(blinks), (int(0.9*width), int(0.97*height)), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    for w, h in irises:
        cv2.circle(frame, (w, h), 2, (0, 255, 0), 2)
    cv2.rectangle(frame,(int(0.01*width),int(0.0125*height)),(int(0.21*width),int(0.18*height)),(255,255,255),1) #takeoff rectangle FW
    cv2.rectangle(frame,(int(0.79*width),int(0.0125*height)),(int(0.99*width),int(0.18*height)),(255,255,255),1)   #landing rectangle FW
    cv2.imshow('Eyeris detector', frame)


class ImageSource:
    """
    Returns frames from camera
    """
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

    def get_current_frame(self, gray=False):
         # ds_factor = 0.5 # screen scale factor if needed
        ret, frame = self.capture.read()
        # frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA) #screen scale
        frame = cv2.flip(frame, 1)  # 60fps
        if not gray:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def release(self):
        self.capture.release()


class CascadeClassifier:
    """
    This classifier is trained by default in OpenCV
    """
    def __init__(self, glasses=True):
        if glasses:
            self.eye_cascade = cv2.CascadeClassifier(join('TrainedDataSet/haarcascade_eye_tree_eyeglasses.xml'))
        else:
            self.eye_cascade = cv2.CascadeClassifier(join('TrainedDataSet/haarcascade_eye.xml'))

    def get_irises_location(self, frame_gray):
        eyes = self.eye_cascade.detectMultiScale(frame_gray, 1.3, 5)  # if not empty - eyes detected
        irises = []

        for (ex, ey, ew, eh) in eyes:
            iris_w = int(ex + float(ew / 2))
            iris_h = int(ey + float(eh / 2))
            irises.append([numpy.float32(iris_w), numpy.float32(iris_h)])

        return numpy.array(irises)


class LucasKanadeTracker:
    """
    Lucaas-Kanade tracker used for minimizing cpu usage and blinks counter
    """
    def __init__(self, blink_threshold=9):
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.blink_threshold = blink_threshold

    def track(self, old_gray, gray, irises, blinks, blink_in_previous):
        lost_track = False
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, irises, None, **self.lk_params)
        if st[0][0] == 0 or st[1][0] == 0:  # lost track on eyes
            lost_track = True
            blink_in_previous = False
        elif err[0][0] > self.blink_threshold or err[1][0] > self.blink_threshold:  # high error rate in klt tracking
            lost_track = True
            if not blink_in_previous:
                blinks += 1
                blink_in_previous = True
        else:
            blink_in_previous = False
            irises = []
            for w, h in p1:
                irises.append([w, h])
            irises = numpy.array(irises)
        return irises, blinks, blink_in_previous, lost_track

class character: ##Fan Added
    def __init__(self):
        self.state = 0; # 0 for waiting, 1 for takeoff, -1 for landing
    def calibrate(self,input): # for modifying collected calibrated data
<<<<<<< HEAD
        output = []
        for i in range (0, len(input)):
            output.append(input[i][0])
=======
        output = input
>>>>>>> master
        output.remove(output[0])
        output.remove(output[-1])
        minvalue = min(output)
        maxvalue = max(output)
        output.remove(minvalue)
        output.remove(maxvalue)
        average = sum(output)/float(len(output))
<<<<<<< HEAD
        return average
=======
        return output, average
>>>>>>> master
    def statetracker(self, font, imagesource, frame, irises, counttf, countld):
        width = imagesource.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = imagesource.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        takeoffwidth = [int(0.01*width),int(0.21*width)]
        takeoffheight = [int(0.0125*height),int(0.18*height)]
        landingwidth = [int(0.79*width),int(0.99*width)]
        landingheight = takeoffheight
        intakeoff = (takeoffwidth[1]>=int(irises[0][0])>=takeoffwidth[0] and takeoffheight[1]>=int(irises[0][1])>=takeoffheight[0]) and (takeoffwidth[1]>=int(irises[1][0])>=takeoffwidth[0] and takeoffheight[1]>=int(irises[1][1])>=takeoffheight[0])
        inlanding = (landingwidth[1]>=int(irises[0][0])>=landingwidth[0] and landingheight[1]>=int(irises[0][1])>=landingheight[0]) and (landingwidth[1]>=int(irises[1][0])>=landingwidth[0] and landingheight[1]>=int(irises[1][1])>=landingheight[0])
        if not(intakeoff or inlanding):
            counttf[:] = []
            countld[:] = []
            g.write("waiting\r\n")
            self.state = 0

        elif intakeoff:
            countld[:] = []
<<<<<<< HEAD
=======
            cv2.rectangle(frame,(takeoffwidth[0],takeoffheight[0]),(takeoffwidth[1],takeoffheight[1]),(255,255,255),2)
>>>>>>> master
            counttf.append(irises)
            waittime = 3 - int(len(counttf)/6)
            if waittime > 0:
                cv2.putText(frame, 'takeoff in: ' + str(waittime) + 'sec', (takeoffwidth[0],int(takeoffheight[1]/2)), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                g.write("waiting\r\n")
                self.state = 0
            elif waittime <= 0:
                g.write("takingoff\r\n")
                cv2.putText(frame, 'taking off,please wait', (takeoffwidth[0],int(takeoffheight[1]/2)), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                self.state = 1

        elif inlanding:
            counttf[:] = []
<<<<<<< HEAD
=======
            cv2.rectangle(frame,(landingwidth[0],landingheight[0]),(landingwidth[1],landingheight[1]),(255,255,255),2)
>>>>>>> master
            countld.append(irises)
            waittime = 3 - int(len(countld)/6)
            if waittime > 0:
                g.write("waiting\r\n")
                cv2.putText(frame, 'landing in: ' + str(waittime) + 'sec', (landingwidth[0],int(landingheight[1]/2)), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                self.state = 0
            elif waittime <= 0:
                g.write("landing\r\n")
                cv2.putText(frame, 'landing,please wait', (landingwidth[0],int(landingheight[1]/2)), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                self.state = -1
        else:
            self.state = 4
            g.write("unexpected error\r\n")
                        

        return self.state


<<<<<<< HEAD
=======


>>>>>>> master


class EyerisDetector:
    """
    Main class which use image source, classifier and tracker to estimate iris postion
    Algorithm used in detector is designed for one person (with two eyes)
    It can detect more than two eyes, but it tracks only two
    """
    def __init__(self, image_source, classifier, tracker):
        self.tracker = tracker
        self.classifier = classifier
        self.image_source = image_source
        self.irises = []
        self.blink_in_previous = False
        self.blinks = 0
        #self.takeoffblinks = 0
        self.landblinks = 0
    
    def run(self):
        
        k = cv2.waitKey(30) & 0xff
        font = cv2.FONT_HERSHEY_SIMPLEX
        counttf = []
        countld = []
<<<<<<< HEAD
        #Calibration Array
        calibrate_array_middle_0 = []
        calibrate_array_middle_1 = []
        calibrate_array_left_0 = []
        calibrate_array_left_1 = []
        calibrate_array_right_0 = []
        calibrate_array_right_1 = []
        cali_start = 1
        # Store result for left and right eye. Index 0 for eye 0, index 1 for eye 1
        cali_centre = []
        cali_left = []
        cali_right = []
=======
>>>>>>> master
        while k != 32:  # space
            frame = self.image_source.get_current_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            width = self.image_source.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.image_source.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            #Calibration Image
            if cali_start == 1:
                if len(calibrate_array_middle_0) < 30:
                    cv2.circle(frame,(int(width/2),int(height/2)), 50, (153,255,255), -1) # Middle circle
                    cv2.putText(frame, 'Please look at yellow circle', (int(width/2)-120,int(height/2)-50), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                    middle_on = 1
                elif len(calibrate_array_middle_0) == 30 and len(calibrate_array_left_0)<30:
                    cv2.circle(frame,(50,int(height/2)), 50, (153,255,0), -1) # left circle
                    cv2.putText(frame, 'Please look at green circle', (0,int(height/2)-50), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                    left_on = 1
                elif len(calibrate_array_middle_0) == 30 and len(calibrate_array_left_0) == 30 and len(calibrate_array_right_0) < 30:
                    cv2.circle(frame,(int(width-50),int(height/2)), 50, (255,153,255), -1) # left circle
                    cv2.putText(frame, 'Please look at pink circle', (int(width-350),int(height/2)-50), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                    right_on = 1
                elif len(calibrate_array_middle_0) == 30 and len(calibrate_array_left_0) == 30 and len(calibrate_array_right_0) == 30:
                    cali_centre.append(character().calibrate(calibrate_array_middle_0))
                    cali_centre.append(character().calibrate(calibrate_array_middle_1))
                    cali_left.append(character().calibrate(calibrate_array_left_0))
                    cali_left.append(character().calibrate(calibrate_array_left_1))
                    cali_right.append(character().calibrate(calibrate_array_right_0))
                    cali_right.append(character().calibrate(calibrate_array_right_1))
                    cali_start = 0



            if len(self.irises) >= 2:  # irises detected, track eyes
                track_result = self.tracker.track(old_gray, gray, self.irises, self.blinks, self.blink_in_previous)
                self.irises, self.blinks, self.blink_in_previous, lost_track = track_result
                # Nose detector
                nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
                nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in nose_rects:
                    f.write("{}\t{}\t{}\t{}\t{}\t{}t\r\n".format(self.irises[0][0],self.irises[0][1],self.irises[1][0],self.irises[1][0],int(x+w/2),int(y+h/2))) ### TEST CODE ###
<<<<<<< HEAD
                
                if middle_on == 1:
                    calibrate_array_middle_0.append([self.irises[0][0],self.irises[0][1]]) # left eye
                    calibrate_array_middle_1.append([self.irises[1][0],self.irises[1][1]]) # right eye
                    middle_on = 0
                elif left_on == 1:
                    calibrate_array_left_0.append([self.irises[0][0],self.irises[0][1]])# left eye
                    calibrate_array_left_1.append([self.irises[1][0],self.irises[1][1]])# right eye
                    left_on = 0
                elif right_on == 1:
                    calibrate_array_right_0.append([self.irises[0][0],self.irises[0][1]])# left eye
                    calibrate_array_right_1.append([self.irises[1][0],self.irises[1][1]])# right eye
                    right_on = 0

                # Take off and Landing
                status = character().statetracker(font, self.image_source, frame, self.irises, counttf, countld) # for status,0 is waiting, -1 is landing, 1 is takingoff, 4 is unexpected error
                
=======
                #data_analysis.write("{}\t{}\t{}\t{}\r\n".format(irises[0][0],irises[0][1],irises[1][0],irises[1][0])) ### TEST CODE ###
                # Take off and Landing
                status = character().statetracker(font, self.image_source, frame, self.irises, counttf, countld) # for status,0 is waiting, -1 is landing, 1 is takingoff, 4 is unexpected error
                
                tp = self.irises
                
                #tp = tp.reshape((1,4))
                #numpy.savetxt(data_analysis, tp, fmt='%1.2f',delimiter=',')
                print (tp)
                
>>>>>>> master
                      
                if lost_track:
                    self.irises = self.classifier.get_irises_location(gray)
            else:  # cannot track for some reason -> find irises
                self.irises = self.classifier.get_irises_location(gray)
            show_image_with_data(frame, self.blinks, self.landblinks, self.irises, self.image_source.capture)
            k = cv2.waitKey(30) & 0xff
            old_gray = gray.copy()

        self.image_source.release()
        cv2.destroyAllWindows()


eyeris_detector = EyerisDetector(image_source=ImageSource(), classifier=CascadeClassifier(),
                                 tracker=LucasKanadeTracker())
eyeris_detector.run()

g.close()
f.close()### TEST CODE ###
