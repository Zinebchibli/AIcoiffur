import math
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Indices de points du visage (Face Mesh 468) utilises pour des ratios simples.
IDX_HAUT_FRONT = 10
IDX_BAS_MENTON = 152
IDX_POMMETTE_GAUCHE = 234
IDX_POMMETTE_DROITE = 454
# Utilisation des points 132 et 361 qui représentent l'angle extérieur de la mâchoire
IDX_MACHOIRE_GAUCHE = 132 
IDX_MACHOIRE_DROITE = 361
# Nouveaux points pour mesurer la largeur du front
IDX_FRONT_GAUCHE = 103
IDX_FRONT_DROIT = 332
MAX_INDEX_UTILISE = max(
    IDX_HAUT_FRONT,
    IDX_BAS_MENTON,
    IDX_POMMETTE_GAUCHE,
    IDX_POMMETTE_DROITE,
    IDX_MACHOIRE_GAUCHE,
    IDX_MACHOIRE_DROITE,
    IDX_FRONT_GAUCHE,
    IDX_FRONT_DROIT,
)


def calculer_distance(p1, p2, largeur_img, hauteur_img):
    """Calcule la distance en pixels entre deux points normalises."""
    x1, y1 = int(p1.x * largeur_img), int(p1.y * hauteur_img)
    x2, y2 = int(p2.x * largeur_img), int(p2.y * hauteur_img)
    return math.hypot(x2 - x1, y2 - y1)


def determiner_forme_visage(landmarks, largeur_img, hauteur_img):
    """Determine la forme du visage (7 types) basee sur des proportions geometriques."""
    if len(landmarks) <= MAX_INDEX_UTILISE:
        return "Indetermine"
    
    # 1. Récupération des points
    pt_haut_front = landmarks[IDX_HAUT_FRONT]
    pt_bas_menton = landmarks[IDX_BAS_MENTON]
    pt_pommette_gauche = landmarks[IDX_POMMETTE_GAUCHE]
    pt_pommette_droite = landmarks[IDX_POMMETTE_DROITE]
    pt_machoire_gauche = landmarks[IDX_MACHOIRE_GAUCHE]
    pt_machoire_droite = landmarks[IDX_MACHOIRE_DROITE]
    pt_front_gauche = landmarks[IDX_FRONT_GAUCHE]
    pt_front_droit = landmarks[IDX_FRONT_DROIT]

    # 2. Calcul des 4 distances principales
    longueur_visage = calculer_distance(pt_haut_front, pt_bas_menton, largeur_img, hauteur_img)
    largeur_pommettes = calculer_distance(pt_pommette_gauche, pt_pommette_droite, largeur_img, hauteur_img)
    largeur_machoire = calculer_distance(pt_machoire_gauche, pt_machoire_droite, largeur_img, hauteur_img)
    largeur_front = calculer_distance(pt_front_gauche, pt_front_droit, largeur_img, hauteur_img)

    if largeur_pommettes <= 1e-6:
        return "Indetermine"

    # 3. Calcul des ratios
    ratio_longueur_largeur = longueur_visage / largeur_pommettes
    
    # Fonction pour vérifier si deux mesures sont presque égales (à 10% près)
    def presque_egal(val1, val2, tolerance=0.1):
        return abs(val1 - val2) < (max(val1, val2) * tolerance)

    # 4. Arbre de décision (La logique des 7 formes)
    
    # Règle 1 : Triangle (Mâchoire > Pommettes > Front)
    if largeur_machoire > largeur_pommettes and largeur_machoire > largeur_front:
        return "Triangle"
        
    # Règle 2 : Diamant (Pommettes plus larges que le front et la mâchoire)
    elif largeur_pommettes > largeur_front * 1.1 and largeur_pommettes > largeur_machoire * 1.1:
        return "Diamond"
        
    # Règle 3 : Coeur (Front très large, mâchoire étroite)
    elif largeur_front > largeur_pommettes and largeur_machoire < largeur_front * 0.8:
        return "Coeur"
        
    # Règle 4 : Visages longs (Rectangle ou Ovale)
    elif ratio_longueur_largeur > 1.35:
        if presque_egal(largeur_front, largeur_pommettes) and presque_egal(largeur_pommettes, largeur_machoire):
            return "Rectangle"
        else:
            return "Ovale"
            
    # Règle 5 : Visages courts/équilibrés (Carré ou Rond)
    else:
        if presque_egal(largeur_front, largeur_pommettes) and presque_egal(largeur_pommettes, largeur_machoire):
            return "Carre"
        else:
            return "Round"


def creer_face_landmarker():
    """Cree un FaceLandmarker a partir du modele local face_landmarker.task."""
    model_path = Path(__file__).with_name("face_landmarker.task")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modele introuvable: {model_path}. Placez face_landmarker.task a cote du script."
        )

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def analyser_visage_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la camera.")

    face_landmarker = creer_face_landmarker()

    print("Camera activee. Appuyez sur 'q' pour quitter.")
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image = cv2.flip(image, 1)
            hauteur, largeur, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            resultats = face_landmarker.detect(mp_image)

            forme = "Aucun visage detecte"
            if resultats.face_landmarks:
                landmarks = resultats.face_landmarks[0]
                forme = determiner_forme_visage(landmarks, largeur, hauteur)

                # Dessin leger des points pour visualiser le suivi.
                for p in landmarks:
                    x, y = int(p.x * largeur), int(p.y * hauteur)
                    cv2.circle(image, (x, y), 1, (0, 255, 255), -1)

            cv2.putText(
                image,
                f"Forme : {forme}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Analyse du Visage IA (Temps Reel)", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        face_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    analyser_visage_camera()