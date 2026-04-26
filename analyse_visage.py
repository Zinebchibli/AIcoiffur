import os
import time

cv2 = None
mp = None

try:
    import cv2
except ImportError:
    print("Le module 'cv2' (OpenCV) n'est pas installe. Veuillez l'installer avec 'pip install opencv-python' et reessayez.")

try:
    import mediapipe as mp
except ImportError:
    print("Le module 'mediapipe' n'est pas installe. Veuillez l'installer avec 'pip install mediapipe' et reessayez.")


def analyser_visage_camera():
    if cv2 is None or mp is None:
        print("Execution annulee: dependances manquantes.")
        return

    if not hasattr(mp, "tasks"):
        print("Execution annulee: cette version de mediapipe ne fournit pas l'API Tasks.")
        return

    model_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
    if not os.path.exists(model_path):
        print("Modele introuvable: 'face_landmarker.task'.")
        return

    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d'ouvrir la camera.")
        return

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        print("Camera activee. Appuyez sur 'q' pour quitter.")

        while cap.isOpened():
            success, image_bgr = cap.read()
            if not success:
                print("Probleme de lecture de la camera.")
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            timestamp_ms = int(time.time() * 1000)

            resultats = landmarker.detect_for_video(mp_image, timestamp_ms)

            if resultats.face_landmarks:
                for visage_landmarks in resultats.face_landmarks:
                    mp.tasks.vision.drawing_utils.draw_landmarks(
                        image=image_bgr,
                        landmark_list=visage_landmarks,
                        connections=mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.tasks.vision.drawing_styles.get_default_face_mesh_tesselation_style(),
                    )

            cv2.imshow("Analyse du Visage IA (Temps Reel)", cv2.flip(image_bgr, 1))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    analyser_visage_camera()