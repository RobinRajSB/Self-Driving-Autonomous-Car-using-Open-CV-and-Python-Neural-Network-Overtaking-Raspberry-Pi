import cv2



faceCascade = cv2.CascadeClassifier('C:\Users\Robin Raj SB\Downloads\Webcam-Face-Detect-master\Webcam-Face-Detect-master\haarcascade_frontalface_default.xml')
CustomCascade = cv2.CascadeClassifier('C:\Users\Robin Raj SB\Desktop\Custom Detection\classifier\cascade.xml')
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Human Detected', (x, y), font, 1, (200, 255, 155))



    code = CustomCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in code:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Custom Code Detected', (x, y), font, 1, (200, 255, 155))




    # Display the resulting frame
    cv2.imshow('Video', frame)












    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()