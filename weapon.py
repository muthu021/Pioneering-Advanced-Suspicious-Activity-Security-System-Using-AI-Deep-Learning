import cv2
import numpy as np
import pygame

# Initialize pygame mixer for audio
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav.mp3")
  # Replace with your alarm sound file

# Load Resnet
net = cv2.dnn.readNet("resnet.weights", "Resnet.cfg")
classes = ["Weapon"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Initialize webcam capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera, change it if needed

while True:
    ret, img = cap.read()

    if not ret:
        continue

    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    weapon_detected = False  # Initialize a flag for weapon detection

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                weapon_detected = True  # Set the flag to True if a weapon is detected

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):  # Corrected line
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    if weapon_detected:
        print("Weapon detected!")
        pygame.mixer.Sound.play(alarm_sound)  # Play the alarm sound
    else:
        print("Weapon not detected!")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
