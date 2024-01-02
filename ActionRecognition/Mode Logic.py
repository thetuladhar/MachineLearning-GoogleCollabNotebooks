def capture_video(var1,var2):
    count2 = 0
    cap = cv2.VideoCapture(0)
    f_count = 0
    w_count = 0
    mode_counter = 0
    side = " "
    MODE_COUNTER_THRESHOLD = 30
    while True:
        if cap:
            success, image = cap.read()
            # Converting the BGR image to RGB before processing
            #print(image)
            #image = cv2.flip(image,1)
            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = hands.process(imgRGB)
            h,w,c = imgRGB.shape
            #print(h,w,c)
            #There are 21 landmarks on hand from 0 - 20
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    #print(handLms.landmark[8].z - handLms.landmark[20].z)
                    Thumb_MCP_X = handLms.landmark[2].x
                    Pinky_MCP_X = handLms.landmark[17].x
                    Middle_MCP_Y = handLms.landmark[9].y
                    Wrist_Y = handLms.landmark[0].y
                    b2 = 0.05
                    #side = ""
                    mode_counter = ""
                    if side != "Fingers" and (Middle_MCP_Y < Wrist_Y):
                        if Thumb_MCP_X < Pinky_MCP_X:
                            #side = "Back - up"
                            w_count = 0
                            if f_count < MODE_COUNTER_THRESHOLD:
                                f_count = f_count + 1
                                mode_counter = MODE_COUNTER_THRESHOLD - f_count
                            elif f_count == MODE_COUNTER_THRESHOLD:
                                f_count = 0
                                side = "Fingers"
                        else:
                            f_count = 0
                    elif side != "Words":
                        if Thumb_MCP_X > Pinky_MCP_X:
                            #side = "Back - down"
                            f_count = 0
                            if w_count < MODE_COUNTER_THRESHOLD:
                                w_count = w_count + 1
                                mode_counter = MODE_COUNTER_THRESHOLD - w_count
                            elif w_count == MODE_COUNTER_THRESHOLD:
                                w_count = 0
                                side = "Words"
                        else:
                            w_count = 0


                    var1.set(side)
                    var2.set(mode_counter)
                    mpDraw.draw_landmarks(image,handLms,mpHands.HAND_CONNECTIONS)

            cv2.imshow("Hand Image",image)#param 1 - window title, param 2 - array of image pixels
            cv2.waitKey(1)