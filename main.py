from ultralytics import YOLO
import cv2
import math
import time

people = {}
objects = {}

IDLE_THRESHOLD = 20     
OBJECT_THRESHOLD = 20     

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("videos/test.mp4")

def get_direction(prev, curr):
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]

    dist = math.sqrt(dx**2 + dy**2)

    if dist < 5:
        return "STILL"
    else:
        return "MOVE"
    
def is_near(person_pos, obj_pos, threshold=100):
    dx = person_pos[0] - obj_pos[0]
    dy = person_pos[1] - obj_pos[1]
    return (dx**2 + dy**2) ** 0.5 < threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)[0]

    if results.boxes is not None:
        for box in results.boxes:
            # Get class
            cls = int(box.cls[0])

            # Only keep PERSON + OBJECTS
            if cls not in [0, 24, 26, 28]:  # person, backpack, handbag, suitcase
                continue

            # Get ID
            track_id = int(box.id[0]) if box.id is not None else -1

            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Set label + color
            if cls == 0:
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                if track_id not in people:
                    people[track_id] = {
                        "pos": (cx, cy),
                        "last_move_time": time.time()
                    }
                    direction = "START"
                    idle_time = 0
                else:
                    prev = people[track_id]["pos"]
                    dx = cx - prev[0]
                    dy = cy - prev[1]
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist < 5:
                        direction = "STILL"
                    else:
                        direction = "MOVE"
                        people[track_id]["pos"] = (cx, cy)
                        people[track_id]["last_move_time"] = time.time()
                    idle_time = int(time.time() - people[track_id]["last_move_time"])
                status = ""
                if idle_time > IDLE_THRESHOLD:
                    status = "WATCHING"
                label = f"Person {track_id} {direction} Idle:{idle_time}s {status}"
                color = (0,255,0)
            elif cls in [24, 26, 28]:
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                if track_id not in objects:
                    objects[track_id] = {
                        "pos": (cx, cy),
                        "last_seen_with_person": time.time()
                    }
                else:
                    objects[track_id]["pos"] = (cx, cy)
                near_person = False
                for pid in people:
                    person_pos = people[pid]["pos"]
                    dx = person_pos[0] - cx
                    dy = person_pos[1] - cy
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist < 100:
                        near_person = True
                        objects[track_id]["last_seen_with_person"] = time.time()
                        break
                unattended_time = int(time.time() - objects[track_id]["last_seen_with_person"])
                status = ""
                if unattended_time > OBJECT_THRESHOLD:
                    status = "LEFT"
                label = f"Object {track_id} Alone:{unattended_time}s {status}"
                color = (255,0,0)
            else:
                continue
            # Draw box
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            # Show label
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Orvex AI - Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()