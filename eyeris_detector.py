import cv2
import numpy
from os.path import join
data_analysis = open('np.csv', 'a')
g = open("track.txt","w+")
# f = open("NOSE.csv","w+")
f = open("Nose.txt","w+")
state = 0
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
    cv2.rectangle(frame,(int(0.01*width),int(0.0125*height)),(int(0.31*width),int(0.21*height)),(255,255,255),2) #takeoff rectangle FW
    cv2.rectangle(frame,(int(0.69*width),int(0.0125*height)),(int(0.99*width),int(0.21*height)),(255,255,255),2)   #landing rectangle FW
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
        output = []
        for i in range (0, len(input)):
            output.append(input[i][0])
        output.remove(output[0])
        output.remove(output[-1])
        minvalue = min(output)
        maxvalue = max(output)
        output.remove(minvalue)
        output.remove(maxvalue)
        average = sum(output)/float(len(output))
        return average
    def drone_state(self,pos,command):
        if pos == -1:
            if command == -1 or command == 0:
                flight = -1
            elif command == 1:
                flight = 0
        elif pos == 0:
            flight = pos + command
        elif pos == 1:
            if command == 1 or command == 0:
                flight = 1
            elif command == -1:
                flight = 0 
        return flight

    def statetracker(self, font, imagesource, frame, irises, counttf, countld):
        width = imagesource.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = imagesource.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        takeoffwidth = [int(0.01*width),int(0.31*width)]
        takeoffheight = [int(0.0125*height),int(0.21*height)]
        landingwidth = [int(0.69*width),int(0.99*width)]
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
        nose_px = []
        nose_py = []
        #Calibration Array
        calibrate_array_middle_0 = []
        calibrate_array_middle_1 = []
        calibrate_array_left_0 = []
        calibrate_array_left_1 = []
        calibrate_array_right_0 = []
        calibrate_array_right_1 = []
        calibrate_array_middle_nose = []
        calibrate_array_left_nose = []
        calibrate_array_right_nose = []
        cali_start = -1 #prepare -1; start 1; done 0 
        signal = 0
        recali = 0 # recali enable 1
        # Store result for left and right eye. Index 0 for eye 0, index 1 for eye 1
        cali_centre = []
        cali_left = []
        cali_right = []
        nose_centre = 0
        nose_left = 0
        nose_right = 0
        # Left/right detection
        detector_0 = []
        detector_1 = []
        detector_nose = []
        drone_start = 1
        flight_pos = 0
        instruction_on = 1
        timer = []
        while k != 32:  # space
            frame = self.image_source.get_current_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            width = self.image_source.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.image_source.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            #Calibration Image
            if cali_start == -1:
                if len(timer) < 30:
                    cv2.putText(frame, 'Get ready for Calibration', (int(width/2)-200,int(height/2)-100), font, 0.8, (0, 177, 255), 2, cv2.LINE_AA)
                else:
                    cali_start = 1
                    timer = [] 

            if cali_start == 1:
                if len(calibrate_array_middle_0) < 50:
                    cv2.circle(frame,(int(width/2),int(height/2)), 50, (153,255,255), -1) # Middle circle
                    cv2.putText(frame, 'Please look at yellow circle', (int(width/2)-150,int(height/2)-50), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                    middle_on = 1
                elif len(calibrate_array_middle_0) == 50 and len(calibrate_array_left_0)<50:
                    cv2.circle(frame,(50,int(height/2)), 50, (153,255,0), -1) # left circle
                    cv2.putText(frame, 'Please look at green circle', (0,int(height/2)-50), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                    left_on = 1
                elif len(calibrate_array_middle_0) == 50 and len(calibrate_array_left_0) == 50 and len(calibrate_array_right_0) < 50:
                    cv2.circle(frame,(int(width-50),int(height/2)), 50, (255,153,255), -1) # left circle
                    cv2.putText(frame, 'Please look at pink circle', (int(width-350),int(height/2)-50), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                    right_on = 1
                elif len(calibrate_array_middle_0) == 50 and len(calibrate_array_left_0) == 50 and len(calibrate_array_right_0) == 50:
                    cali_centre.append(character().calibrate(calibrate_array_middle_0))
                    cali_centre.append(character().calibrate(calibrate_array_middle_1))
                    cali_left.append(character().calibrate(calibrate_array_left_0))
                    cali_left.append(character().calibrate(calibrate_array_left_1))
                    cali_right.append(character().calibrate(calibrate_array_right_0))
                    cali_right.append(character().calibrate(calibrate_array_right_1))
                    nose_centre = character().calibrate(calibrate_array_middle_nose)
                    nose_left = character().calibrate(calibrate_array_left_nose)
                    nose_right = character().calibrate(calibrate_array_right_nose)
                    # Unsuccess Calibration
                    if (cali_left[0] >  cali_centre[0] or cali_left[1] > cali_centre[1]) or (cali_centre[0] > cali_right[0] or cali_centre[1] > cali_right[1]) or (cali_left[0] > cali_right[0] or cali_left[1] > cali_right[1]) or (nose_left > nose_right):
                        recali = 1
                        if len(timer) < 10:
                            cv2.putText(frame, 'Unsuccessful Calibration, Please follow the instruction and re-calibrate', (int(width/2)-450,int(height/2)-100), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        else:
                            cali_start = -1
                            recali = 0
                            timer = []
                            calibrate_array_middle_0 = []
                            calibrate_array_middle_1 = []
                            calibrate_array_left_0 = []
                            calibrate_array_left_1 = []
                            calibrate_array_right_0 = []
                            calibrate_array_right_1 = [] 
                            cali_centre = []
                            cali_left = []
                            cali_right = []
                    else: 
                        cali_start = 0
   
            elif cali_start == 0:
                if instruction_on == 1:
                    cv2.putText(frame, 'Taking off the Drone Now', (int(width/2)-200,int(height/2)-100), font, 0.8, (0, 163, 24), 2, cv2.LINE_AA)
                    # instruction_on = 1
                if instruction_on == 2:
                    if len(timer) < 30:
                        cv2.putText(frame, 'Get ready to control the Drone', (int(width/2)-200,int(height/2)-100), font, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
                    else:
                        instruction_on = 3 
                        timer = []
                elif instruction_on == 3:
                    cv2.putText(frame, 'Landing the drone by putting eyes into the upper right box anytime you want', (int(width/2)-300,int(height)-50), font, 0.6, (204, 0, 0), 1, cv2.LINE_AA)
                    if signal == -1:
                        cv2.putText(frame, 'Left', (20,int(height/2)), font, 1, (255, 102, 178), 2, cv2.LINE_AA)
                        signal = 0
                    elif signal == 1:
                        cv2.putText(frame, 'Right', (int(width)-100,int(height/2)), font, 1, (255, 102, 178), 2, cv2.LINE_AA)
                        signal = 0
                    else:
                        cv2.putText(frame, '', (int(width/2)-20,int(height/2)), font, 1, (255, 102, 178), 2, cv2.LINE_AA)
                        signal = 0

                elif instruction_on == 4:
                    cv2.putText(frame, 'Drone is landed. Take off whenever you want', (int(width/2)-300,int(height/2)), font, 0.8, (0, 140, 255), 2, cv2.LINE_AA)


            if len(self.irises) >= 2:  # irises detected, track eyes
                track_result = self.tracker.track(old_gray, gray, self.irises, self.blinks, self.blink_in_previous)
                self.irises, self.blinks, self.blink_in_previous, lost_track = track_result
                # Nose detector
                nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
                nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
                f.write("{}\t\r\n".format(nose_rects)) ### TEST CODE ###
                for (x,y,w,h) in nose_rects:
                    nose_x = int(x+w/2)
                    nose_y = int(y+h/2)
                    nose_px.append(nose_x)
                    nose_py.append(nose_y)
                    # f.write("{}\t\r\n".format(nose_rects)) ### TEST CODE ###
                
                if cali_start == -1 or recali == 1:
                    timer.append([self.irises[0][0],self.irises[0][1]]) #for timing purpose only
                elif cali_start == 1:

                    if middle_on == 1:
                        calibrate_array_middle_0.append([self.irises[0][0],self.irises[0][1]]) # left eye
                        calibrate_array_middle_1.append([self.irises[1][0],self.irises[1][1]]) # right eye
                        calibrate_array_middle_nose.append([nose_px[-1],nose_py[-1]]) #nose
                        middle_on = 0
                    elif left_on == 1:
                        calibrate_array_left_0.append([self.irises[0][0],self.irises[0][1]])# left eye
                        calibrate_array_left_1.append([self.irises[1][0],self.irises[1][1]])# right eye
                        calibrate_array_left_nose.append([nose_px[-1],nose_py[-1]]) #nose
                        left_on = 0
                    elif right_on == 1:
                        calibrate_array_right_0.append([self.irises[0][0],self.irises[0][1]])# left eye
                        calibrate_array_right_1.append([self.irises[1][0],self.irises[1][1]])# right eye
                        calibrate_array_right_nose.append([nose_px[-1],nose_py[-1]]) #nose
                        right_on = 0
 
                #Generate left/right detection
                elif cali_start == 0:
                    # Procedure: Calibration -> taking off -> left/right
                    # Take off and Landing
                    status = character().statetracker(font, self.image_source, frame, self.irises, counttf, countld) # for status,0 is waiting, -1 is landing, 1 is takingoff, 4 is unexpected error
                    if instruction_on == 1 or instruction_on == 3 or instruction_on == 4: # Take off instruction   
                        # Global variable for drone communication
                        global switch_state  # 0b0011 for waiting, 0b0111 for takeoff, 0b1111 for landing
                        if status == 0:
                            switch_state = 0b0011
                        elif status == 1:
                            switch_state = 0b0111
                            drone_start = 0
                            instruction_on = 2
                        elif status == -1:
                            switch_state = 0b1111
                            drone_start = 1
                            instruction_on = 4

                        if instruction_on == 3: # Left/Right command
                            global drone_pos
                            if len(detector_0) < 10:
                                detector_0.append([self.irises[0][0],self.irises[0][1]]) # left eye
                                detector_1.append([self.irises[1][0],self.irises[1][1]]) # right eye
                                detector_nose.append([nose_px[-1],nose_py[-1]]) # nose
                            if len(detector_0) == 10:
                                current_0 = character().calibrate(detector_0) # left eye current location
                                current_1 = character().calibrate(detector_1) # right eye current location
                                current_nose = character().calibrate(detector_nose) # nose current location
                                if (current_0 <= cali_left[0] and current_1 <= cali_left[1]) and current_nose < nose_left: #see left
                                    det = -1
                                    flight_pos = character().drone_state(flight_pos,det)
                                    signal = -1
                                elif (current_0 >= cali_right[0] and current_1 >= cali_right[1]) and current_nose > nose_right: #see right
                                    det = 1
                                    flight_pos = character().drone_state(flight_pos,det)
                                    signal = 1
                                elif (current_0 > cali_left[0] and current_0 < cali_right[0]) and (current_1 > cali_left[1] and current_1 < cali_right[1]): # in middle range
                                    det = 0
                                    flight_pos = character().drone_state(flight_pos,det)
                                    signal = 0
                                detector_0.remove(detector_0[0])
                                detector_1.remove(detector_1[0])
                                drone_pos = flight_pos
                                f.write("{}\r\n".format(flight_pos))
                    
                    elif instruction_on == 2: #Back to position intruction
                        timer.append([self.irises[0][0],self.irises[0][1]]) #for timing purpose only
                            
                
                      
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
