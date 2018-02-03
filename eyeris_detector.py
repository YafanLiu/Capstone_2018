import cv2
import numpy
from os.path import join
f = open("state.txt","w+")
g = open("track.txt","w+")
# I added window for obtaining the window dimension
def show_image_with_data(frame, blinks, landblinks, irises, window, err=None):
    """
    Helper function to draw points on eyes and display frame
    :param frame: image to draw on
    :param blinks: number of blinks
    :param irises: array of points with coordinates of irises
    :param err: for displaying current error in Lucas-Kanade tracker
    :param takeoffblinks: number of blinks in take off circle frame         -----deleted
    :param landblinks: number of blinks in land circle frame    -----deleted
    :return:
    """
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = window.get(cv2.CAP_PROP_FRAME_WIDTH) # float
    height = window.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    #width = 2800;
    #height = 1800;
    #window.set(cv2.CAP_PROP_FRAME_WIDTH, width) # float
    #window.set(cv2.CAP_PROP_FRAME_HEIGHT,height) # float
    if err:
        cv2.putText(frame, str(err), (20, 450), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'blinks: ' + str(blinks), (int(90*width/100), int(79*height/80)), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    #cv2.putText(frame, 'Irises Location' + str(irises), (20, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    for w, h in irises:
        cv2.circle(frame, (w, h), 2, (0, 255, 0), 2)
    #cv2.line(frame,(int(width/2),0),(int(width/2),int(height)),(255,0,0),1)
    #cv2.rectangle(frame,(int(width/2-150),int(height/2-175)),(int(width/2+150),int(height/2+175)),(0,255,0),2)
    #cv2.rectangle(frame,(int(width/2-180),int(height/2-205)),(int(width/2+180),int(height/2+205)),(0,255,255),2)
    # we want to rectangular frame work where to put the eyes and blink 3 times for the drone to take off and land
    #takeoffleft = int(width/100)
    #takeoffright = int(24*width/100)
    #takeoffup = int(height/80)
    #takeofflow = int(15*height/80)
    cv2.rectangle(frame,(int(width/100),int(height/80)),(int(24*width/100),int(15*height/80)),(255,255,255),1) #takeoff rectangle
    #cv2.putText(frame, '['+ str(width/100)+ ' ,'+ str(height/80)+']'+'['+ str(20*width/100)+ ' ,'+ str(4*height/25)+']'+'takeoff blinks: ' + str(takeoffblinks), (int(width/100),int(18*height/80)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #landingleft = int(80*width/100)
    #landingright = int(99*width/100)
    #landingup = int(height/80)
    #landinglow = int(15*height/80)
    cv2.rectangle(frame,(int(80*width/100),int(height/80)),(int(99*width/100),int(15*height/80)),(255,255,255),1)   #landing rectangle
    #cv2.putText(frame, 'landblinks: ' + str(landblinks), (int(width/2+320),int(height/2-125)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Eyeris detector', frame)
    #cv2.namedWindow('Eyeris detector', cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('Eyeris detector', 1920, 1080)

class ImageSource:
    """
    Returns frames from camera
    """
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

    def get_current_frame(self, gray=False):
        ret, frame = self.capture.read()
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
            self.eye_cascade = cv2.CascadeClassifier(join('haar', 'haarcascade_eye_tree_eyeglasses.xml'))
        else:
            self.eye_cascade = cv2.CascadeClassifier(join('haar', 'haarcascade_eye.xml'))

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
        count = []
        k = cv2.waitKey(30) & 0xff
        font = cv2.FONT_HERSHEY_SIMPLEX
        while k != 32:  # space
            frame = self.image_source.get_current_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if len(self.irises) >= 2:  # irises detected, track eyes
                track_result = self.tracker.track(old_gray, gray, self.irises, self.blinks, self.blink_in_previous)
                self.irises, self.blinks, self.blink_in_previous, lost_track = track_result
                
                #### trying to check if irises is within takeoff or landing range#####
                #windowwidth = self.image_source.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                #windowheight = self.image_source.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                windowwidth = self.image_source.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                windowheight = self.image_source.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                takeoffwidth = [int(windowwidth/100),int(24*windowwidth/100)]
                takeoffheight = [int(windowheight/80),int(15*windowheight/80)]
                landingwidth = [int(80*windowwidth/100),int(99*windowwidth/100)]
                landingheight = takeoffheight
                #f.write("irises: {} \r\n".format(self.irises[0][0]))
                intakeoff = (takeoffwidth[1]>=int(self.irises[0][0])>=takeoffwidth[0] and takeoffheight[1]>=int(self.irises[0][1])>=takeoffheight[0]) or (takeoffwidth[1]>=int(self.irises[1][0])>=takeoffwidth[0] and takeoffheight[1]>=int(self.irises[1][1])>=takeoffheight[0])
                inlanding = (landingwidth[1]>=int(self.irises[0][0])>=landingwidth[0] and landingheight[1]>=int(self.irises[0][1])>=landingheight[0]) or (landingwidth[1]>=int(self.irises[1][0])>=landingwidth[0] and landingheight[1]>=int(self.irises[1][1])>=landingheight[0])
                if not(intakeoff or inlanding):
                    count[:] = []
                    f.write(str(len(count)))
                    g.write("waiting\r\n")
                
                if intakeoff:
                    cv2.rectangle(frame,(int(windowwidth/100),int(windowheight/80)),(int(24*windowwidth/100),int(15*windowheight/80)),(255,255,255),2)
                    count.append(self.irises)
                    time = 3 #3secs
                    f.write(str(len(count)))
                    waittime = 3 - int(len(count)/15)
                    if waittime > 0:
                        cv2.putText(frame, 'takeoff wait: ' + str(waittime) + 'sec', (int(10*windowwidth/100),int(2*windowheight/80)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        g.write("waiting\r\n")
                    if waittime <= 0:
                        #if waittime > -1:
                        g.write("takingoff\r\n")
                        cv2.putText(frame, 'taking off, please wait', (int(10*windowwidth/100),int(2*windowheight/80)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        '''else:
                            cv2.putText(frame, 'reset counter', (int(10*windowwidth/100),int(2*windowheight/80)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                                #count[:] = []'''
            
                if inlanding:
                    cv2.rectangle(frame,(int(80*windowwidth/100),int(windowheight/80)),(int(99*windowwidth/100),int(15*windowheight/80)),(255,255,255),2)
                    count.append(self.irises)
                    time = 3 #3secs
                    f.write(str(len(count)))
                    waittime = 3 - int(len(count)/15)
                    if waittime > 0:
                        g.write("waiting\r\n")
                        cv2.putText(frame, 'landing wait: ' + str(waittime) + 'sec', (int(80*windowwidth/100),int(2*windowheight/80)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    if waittime <= 0:
                        g.write("landing\r\n")
                        cv2.putText(frame, 'taking off, please wait', (int(80*windowwidth/100),int(2*windowheight/80)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                       
                       #cv2.putText(frame, 'reset counter', (int(80*windowwidth/100),int(2*windowheight/80)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                       #count[:] = []
                                    
                #if (len(count)!=0 and (int(self.irises[0][0])>takeoffwidth[1] and int(self.irises[0][0])<landingwidth[1] )
                    '''if len(count) == time*15:
                        g.write("takeoff")
                        count[:] = []'''
                
                '''if self.blinks not in count:
                            count.append(self.blinks)
                            g.write("{} \r\n".format(count))
                            self.takeoffblinks = len(count)
                            if self.takeoffblinks == 3:
                                f.write("takeoff")
                                count[:] = []
                #landingwidth = [int(windowwidth/2+320),int(windowwidth/2+200)]
                #landingheight = [int(windowheight/2-95),int(windowheight/2-175)]
                ################### end checking ####################################
                '''
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

f.close()
g.close()
