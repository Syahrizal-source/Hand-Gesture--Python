import cv2
import mediapipe as mp
import time

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Landmark ujung jari
tip_ids = [4, 8, 12, 16, 20]

prev_time = 0

while True:
    success, img = cap.read()

    if not success:
        break

    # Mirror
    img = cv2.flip(img, 1)

    # Convert ke RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process
    results = hands.process(img_rgb)

    lm_list = []

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Gambar landmark
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Ambil koordinat landmark
            for id, lm in enumerate(hand_landmarks.landmark):

                h, w, c = img.shape

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                lm_list.append((id, cx, cy))

        # ==========================
        # HITUNG JARI
        # ==========================

        fingers = []

        # Thumb
        if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 jari lainnya
        for id in range(1, 5):

            if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)

        # Tampilkan jumlah jari
        cv2.rectangle(img, (20, 20), (220, 120), (0, 255, 0), -1)

        cv2.putText(
            img,
            f'Fingers: {total_fingers}',
            (40, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        img,
        f'FPS: {int(fps)}',
        (10, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Show
    cv2.imshow("Finger Counter", img)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()