import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def nothing(x):
    pass

# Function to find angle between two vectors
def angle(v1, v2):
    dot = np.dot(v1, v2)
    x_modulus = np.sqrt((v1 * v1).sum())
    y_modulus = np.sqrt((v2 * v2).sum())
    cos_angle = dot / x_modulus / y_modulus
    angle = np.degrees(np.arccos(cos_angle))
    return angle

# Function to find distance between two points in a list of lists
def findDistance(A, B):
    return np.sqrt(np.power((A[0][0] - B[0][0]), 2) + np.power((A[0][1] - B[0][1]), 2))

def detect_hand(imgName, img, plot=False, debug=False):
    """Gets an image and return (x,y) of finger"""
    # Parameters
    blur_ksize = 3
    hsv_skin_low = np.array([2, 50, 50])
    hsv_skin_high = np.array([17, 255, 255])
    min_contour_area = 5000  # Minimal contour area to be considered as hand, in pixels.
    img_debug = img

    # Blur the image
    blur = cv2.blur(img, (blur_ksize, blur_ksize))

    # Convert to HSV color space
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    if debug:
        plt.figure()
        plt.imshow(hsv, cmap='hsv')
        plt.title('HSV')
        plt.show()

    # Create a binary image where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, hsv_skin_low, hsv_skin_high)

    if debug:
        plt.figure()
        plt.imshow(mask2, cmap='gray')
        plt.title('BW mask according to HSV')
        plt.show()

    # Kernel matrices for morphological transformation
    kernel_square = np.ones((11, 11), np.uint8)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Perform morphological transformations to filter out the background noise
    # Dilation increase skin color area
    # Erosion increase skin color area
    dilation = cv2.dilate(mask2, kernel_ellipse, iterations=1)
    if debug:
        plt.figure()
        plt.imshow(dilation, cmap='gray')
        plt.title('dilation')
        plt.show()

    erosion = cv2.erode(dilation, kernel_square, iterations=1)

    if debug:
        plt.figure()
        plt.imshow(erosion, cmap='gray')
        plt.title('erosion')
        plt.show()
    dilation2 = cv2.dilate(erosion, kernel_ellipse, iterations=1)
    filtered = cv2.medianBlur(dilation2, 5)
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    dilation2 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    #kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #dilation3 = cv2.dilate(filtered, kernel_ellipse, iterations=1)
    median = cv2.medianBlur(dilation2, 5)
    ret, thresh = cv2.threshold(median, 127, 255, 0)

    if debug:
        plt.figure()
        plt.imshow(thresh, cmap='gray')
        plt.title('BW image after filtering')
        plt.show()

    # Find contours of the filtered frame
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw Contours
    #cv2.drawContours(frame, cnt, -1, (122,122,0), 3)
    # cv2.imshow('Dilation',median)

    # If no contours in image, return and assume no hand in image
    if len(contours) <= 0:
        return None

    # Find Max contour area (Assume that hand is in the frame)
    max_area = min_contour_area
    ci = -1
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area > max_area):
            max_area = area
            ci = i
    # Make sure we found contour large enough
    if ci == -1:
        return None

    # Largest area contour
    cnts = contours[ci]

    # Find convex hull
    hull = cv2.convexHull(cnts)

    # Find convex defects
    hull2 = cv2.convexHull(cnts, returnPoints=False)
    defects = cv2.convexityDefects(cnts, hull2)

    # Get defect points and draw them in the original image
    FarDefect = []
    if defects.shape != 0:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnts[s][0])
            end = tuple(cnts[e][0])
            far = tuple(cnts[f][0])
            FarDefect.append(far)
            cv2.line(img_debug, start, end, [0, 255, 0], 1)
            cv2.circle(img_debug, far, 10, [100, 255, 255], 3)

            # Find moments of the largest contour
            moments = cv2.moments(cnts)

        # Central mass of first order moments
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
            cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
        centerMass = (cx, cy)

    # Draw center mass
    cv2.circle(img_debug, centerMass, 7, [100, 0, 255], 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_debug, 'Center', tuple(centerMass), font, 2, (255, 255, 255), 2)
    cv2.putText(img_debug, imgName, (30, 30), font, 1, (255, 255, 255), 2)

    # Distance from each finger defect(finger webbing) to the center mass
    distanceBetweenDefectsToCenter = []
    for i in range(0, len(FarDefect)):
        x = np.array(FarDefect[i])
        centerMass = np.array(centerMass)
        distance = np.sqrt(np.power(x[0] - centerMass[0], 2) + np.power(x[1] - centerMass[1], 2))
        distanceBetweenDefectsToCenter.append(distance)

    # Get an average of three shortest distances from finger webbing to center mass
    sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
    AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])

    # Get fingertip points from contour hull
    # If points are in proximity of 40 pixels, consider as a single point in the group
    finger = []
    # find points with close proximity and choose the maximal y between them
    close_points = []
    stray_points = True
    stray_points_max_ind = -1
    centerMass_gap = 70
    fingers_gap = 30
    if False:
        for i in range(0, len(hull)-1):
            if hull[i][0][0] <= hull[i+1][0][0]:
                if stray_points:
                    stray_points_max_ind = i
                    stray_points = False
                dist = np.sqrt((hull[i][0][0] - hull[i + 1][0][0])**2 + (hull[i][0][1] - hull[i + 1][0][1])**2)
                #if (np.absolute(hull[i][0][0] - hull[i + 1][0][0]) > 40) or (np.absolute(hull[i][0][1] - hull[i + 1][0][1]) > 40):
                if hull[i][0][1] < (centerMass[1] + centerMass_gap):
                    if dist < fingers_gap:
                        close_points.append(hull[i][0])
                    else:
                        close_points.append(hull[i][0])
                        close_points = sorted(close_points, key=lambda v: v[1])
                        finger.append(close_points[0])
                        close_points = []

        if stray_points_max_ind != -1:
            # add stray points that were skipped
            for i in range(0, stray_points_max_ind):
                if hull[i][0][1] < (centerMass[1] + centerMass_gap):
                    close_points.append(hull[i][0])

            # add final point
            if hull[i][0][1] < (centerMass[1] + centerMass_gap):
                close_points.append(hull[len(hull) - 1][0])
            close_points = sorted(close_points, key=lambda v: v[1])
            # take the point with the smallest y
            if len(close_points):
                finger.append(close_points[0])
    if True:
        xGap = 7
        sortedHull = []
        for i in range(0, len(hull)):
            if hull[i][0][1] < (centerMass[1] + centerMass_gap):
                # Calculate distance of each finger tip to the center mass
                distance = np.sqrt(np.power(hull[i][0][0] - centerMass[0], 2) + np.power(hull[i][0][1] - centerMass[1], 2))
                if distance > round(AverageDefectDistance * 1.75):
                    sortedHull.append(hull[i])
        sortedHull = sorted(sortedHull, key=lambda v: v[0][0])
        for i in range(0, len(sortedHull) - 1):
            distX = np.sqrt((sortedHull[i][0][0] - sortedHull[i + 1][0][0])** 2)
            dist = np.sqrt((sortedHull[i][0][0] - sortedHull[i + 1][0][0]) ** 2  + (sortedHull[i][0][1] - sortedHull[i + 1][0][1]) ** 2)
            if sortedHull[i][0][1] < (centerMass[1] + centerMass_gap):
                if dist < fingers_gap:
                    close_points.append(sortedHull[i][0])
                elif distX < xGap:
                    close_points.append(sortedHull[i][0])
                else:
                    close_points.append(sortedHull[i][0])
                    close_points = sorted(close_points, key=lambda v: v[1])
                    finger.append(close_points[0])
                    close_points = []

        # add final point
        if len(sortedHull):
            if sortedHull[len(sortedHull) - 1][0][1] < (centerMass[1] + centerMass_gap):
                close_points.append(sortedHull[len(sortedHull) - 1][0])
            close_points = sorted(close_points, key=lambda v: v[1])
        # take the point with the smallest y
        if len(close_points):
            finger.append(close_points[0])

        # The fingertip points are 5 hull points with largest y coordinates
        fingers_result = sorted(finger, key=lambda v: v[1])
        fingers = fingers_result[0:5]


    if False:
        # Calculate distance of each finger tip to the center mass
        fingerDistance = []
        for i in range(0, len(finger)):
            distance = np.sqrt(np.power(finger[i][0] - centerMass[0], 2) + np.power(finger[i][1] - centerMass[1], 2))
            fingerDistance.append(distance)


        # Finger is pointed/raised if the distance of between fingertip to the center mass is larger
        # than the distance of average finger webbing to center mass by 130 pixels
        fingers_result = []
        result_ind = 0
        for i in range(0, len(finger)):
            if fingerDistance[i] > round(AverageDefectDistance*1.75):
                fingers_result.append(finger[i])
                result_ind = result_ind + 1

        # The fingertip points are 5 hull points with largest y coordinates
        fingers_result = sorted(fingers_result, key=lambda v: v[1])
        fingers = fingers_result[0:5]

    if debug:
        plt.figure()
        plt.imshow(img_debug)
        for xy in fingers:
            plt.scatter(xy[0], xy[1], s=50, marker='+', color='red')


    if plot:
        # Print bounding rectangle
        x, y, w, h = cv2.boundingRect(cnts)
        img_debug = cv2.rectangle(img_debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(img_debug, [hull], -1, (255, 255, 255), 2)
        for i in range(0, len(finger)):
            cv2.circle(img_debug, tuple(finger[i]), 10, [255, 0, 0], 3)
        for i in range(0, len(fingers)):
            cv2.circle(img_debug, tuple(fingers[i]), 10, [0, 0, 255], 2)

        #Show final image
        cv2.imshow('Plot', img_debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return fingers

def find_video(save_video = False):
    # Open Camera object
    cap = cv2.VideoCapture(1)
    # Decrease frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if save_video:
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('hand_capture_13.avi', fourcc, 30.0, (640, 480))



    while True:
        # Capture frames from the camera
        ret, frame = cap.read()
        fingers_xy = detect_hand('', img=frame, plot=False)
        if fingers_xy is not None:
            for finger in fingers_xy:
                cv2.circle(frame, tuple(finger), 10, [255, 0, 0], 3)

        if fingers_xy == None:
            cv2.putText(frame, 'no hand detected', (30, 30), font, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, str(len(fingers_xy)), (30, 30), font, 1, (255, 255, 255), 2)



        cv2.imshow('FingerDetection', frame)

        if save_video:
            # write the frame
            out.write(frame)

        # close the output video by pressing 'ESC'
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        if k == ord('s'):
            cv2.imwrite('/tmp/hand.png', frame)
    cap.release()

    if save_video:
        out.release()
    cv2.destroyAllWindows()

# TEST SCRIPT
if __name__ == "__main__":
    # Test basic finger detection on all images
    if False:
        df = pd.read_csv("handsDB_adapted.csv")
        for row_i, row in df.iterrows():
            im = cv2.imread(row['image_name'])
            imgName = row['image_name']
            fingers_xy = detect_hand(imgName, img=im , plot=True, debug=True)


    # Test basic finger detection on specific image
    if True:
        imgName = 'img_0036.png'
        im = cv2.imread(imgName)
        fingers_xy = detect_hand(imgName, img=im, plot=True, debug=True)
        print(fingers_xy)
    # Test Movie
    if False:
        find_video(save_video = True)

