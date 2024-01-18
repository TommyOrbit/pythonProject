import cv2
import mediapipe as mp
#----------------------------------------------------------------------------------------------------------------------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
#----------------------------------------------------------------------------------------------------------------------------------------------
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    if not cap.isOpened():
        print("Camera isn't open.")
        exit()
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame.")
            break

        #img = cv2.resize(img,(520,330))
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = holistic.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#----------------------------------------------------------------------------------------------------------------------------------------------
        #face

        mp_drawing.draw_landmarks(
            img,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec = mp_drawing_style
            .get_default_face_mesh_contours_style()
        )

        #body

        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=None,
            #connection_drawing_spec = mp_drawing_style
            #.get_default_pose_landmarks_style()
        )

        # hands

        mp_drawing.draw_landmarks(
            img,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=None,
            #connection_drawing_spec = mp_drawing_style
            #.get_default_hand_connections_style()
        )

        mp_drawing.draw_landmarks(
            img,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=None,
            # connection_drawing_spec = mp_drawing_style
            # .get_default_hand_connections_style()
        )

        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()



