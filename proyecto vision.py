# Dependencies: Python 3.6.x, Open CV, pySerial, matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cv2, time, serial
np.seterr(divide='ignore')

FRAME_INDICATOR = 200
KXC = 0.005; KXI = 0.00001; KTHETAC = 10; KTHETAI = 0.001;
x_error_prev = 0
theta_error_prev = 0
motor_dif_prev = 0
time_prev = initial_time = time.time()

arduino = serial.Serial('COM3', 19200)
cap = cv2.VideoCapture(0)
cv2.namedWindow('original', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('aerial', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar('outlook', 'aerial', 50, 100, lambda _: None)
cv2.createTrackbar('spread', 'aerial', 100, 100, lambda _: None)
cv2.namedWindow('filtered', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar('th', 'filtered', 95, 255, lambda _: None)
cv2.createTrackbar('kernel', 'filtered', 6, 50, lambda _: None)
cv2.namedWindow('final', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
plt.figure()
plt.ioff()

while(cv2.waitKey(1) & 0xFF != 27):
    # Gets Camara Frame
    _, original = cap.read()
    h, w, c = original.shape
    
    # Adjusts Image Wrap, Filter Limits, and Kernel Size based on User Input
    ol = cv2.getTrackbarPos('outlook', 'aerial') / 100
    s = cv2.getTrackbarPos('spread', 'aerial') / 100
    trans_src = np.float32([[w * (1 - s) / 2, h - h * ol], [0, h], [w, h],
                            [w - w * (1 - s) / 2, h - h * ol]])
    trans_dst = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
    
    th = cv2.getTrackbarPos('th', 'filtered')
    lower_white = np.array([0, 0, 255 - th], np.uint8)
    upper_white = np.array([179, th, 255], np.uint8)
    
    k = cv2.getTrackbarPos('kernel', 'filtered')
    kernel = np.ones((k + 1, k + 1), np.uint8)
    
    # Wraps the Image to get Aerial
    pt = cv2.getPerspectiveTransform(trans_src, trans_dst)
    aerial = cv2.warpPerspective(original, pt, (w, h), flags=cv2.INTER_NEAREST)

    # Converts to hsv and applies Filter
    hsv = cv2.cvtColor(aerial, cv2.COLOR_BGR2HSV)
    filtered = cv2.inRange(hsv, lower_white, upper_white)
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

    # Applies Probabilistic Hough Transform and gets the main Lane
    lines = cv2.HoughLinesP(
        filtered, 1, np.pi/180, 250, minLineLength=0.6*h, maxLineGap=0.3*h)
    if lines is not None:
        terms = len(lines[:15])
        sum_x1 = sum_y1 = sum_x2 = sum_y2 = 0
        for line in lines[:15]:
            x1, y1, x2, y2 = line[0]
            cv2.line(aerial, (x1, y1), (x2, y2), (0, 0, 255), 2)
            sum_x1 += x1; sum_y1 += y1; sum_x2 += x2; sum_y2 += y2;
        x1 = sum_x1/terms; y1 = sum_y1/terms; x2 = sum_x2/terms; y2 = sum_y2/terms
        m = (y2 - y1) / (x2 - x1)
        x_int = x1 + (h - y1) / m if m != np.inf else w/2
    else:
        m, x_int = (np.inf, w/2)  # Default main Lane
    cv2.line(aerial, (int(x_int), h), (int(x_int - h / m), 0), (255, 0, 0), 3)

    # Draw final Image and Measurements
    final = cv2.warpPerspective(original, pt, (w, h), flags=cv2.INTER_NEAREST)
    cv2.line(final, (int(x_int), h), (int(x_int - h / m), 0), (255, 255, 0), 3)
    cv2.line(final, (w//2, h), (w//2, 0), (0, 165, 255), 3)
    # Display Errors
    x_error = x_int - w/2
    theta_error = np.arctan(m) + np.pi * (m < 0) - np.pi/2
    cv2.putText(final, 'X{0:03.4f}'.format(x_error), (w//2 + 5, h),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(final, 'T{0:03.4f}'.format(theta_error), (w//2 + 5, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
    final = cv2.warpPerspective(final, pt, (w, h), flags=cv2.WARP_INVERSE_MAP)

    # Control
    time_dif = time.time() - time_prev
    motor_dif = (  # Should be in range [-1, 1]; (-) left turn, (+) right turn
        motor_dif_prev +
        KXC * (x_error - x_error_prev + time_dif*KXI * x_error) +
        KTHETAC * (theta_error - theta_error_prev + time_dif*KTHETAI * theta_error))
    # To add D Component: K_D/time_dif * (_error - 2*_error_prev + _error_prev2)
    leftPower = 100 * max(0, 1 if motor_dif > 0 else 1 + motor_dif) 
    rightPower = 100 * max(0, 1 if motor_dif < 0 else 1 - motor_dif)
    # Saving previous state
    x_error_prev = x_error
    theta_error_prev = theta_error
    motor_dif_prev = motor_dif
    time_prev = time.time()

    # Plotting
    run_time = time.time() - initial_time
    plt.subplot(311)
    plt.axis('auto')
    plt.title('X Error')
    plt.scatter(run_time, x_error, 10, 2)
    plt.subplot(312)
    plt.axis('auto')
    plt.title('Theta Error')
    plt.scatter(run_time, theta_error, 10, 2)
    plt.subplot(313)
    plt.axis('auto')
    plt.title('Motor Dif')
    plt.scatter(run_time, motor_dif, 10, 2)
    plt.tight_layout()

    # Shows Results and Execute
    arduino.write(bytes([FRAME_INDICATOR,
                         min(100, max(0, int(leftPower))),
                         min(100, max(0, int(rightPower)))]))
    cv2.imshow('original', original)
    cv2.imshow('aerial', aerial)
    cv2.imshow('filtered', filtered)
    cv2.imshow('final', final)

arduino.write(bytes([FRAME_INDICATOR, 0, 0]))      
arduino.close()
cap.release()
cv2.destroyAllWindows()
plt.savefig('plot.png')
