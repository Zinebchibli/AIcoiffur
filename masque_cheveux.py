import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
from pathlib import Path

def preparer_modeles():
    """S'assure que les deux modèles IA sont présents dans le dossier."""
    nom_seg = "selfie_segmenter.tflite"
    url_seg = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    if not os.path.exists(nom_seg):
        print("Téléchargement de l'IA de segmentation...")
        urllib.request.urlretrieve(url_seg, nom_seg)
        
    chemin_face = Path(__file__).with_name("face_landmarker.task")
    if not chemin_face.exists():
        raise FileNotFoundError(f"Modèle introuvable : {chemin_face}. Assurez-vous qu'il est dans le dossier.")
        
    return nom_seg, str(chemin_face)

def masque_cheveux_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra.")

    chemin_seg, chemin_face = preparer_modeles()

    options_seg = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=chemin_seg),
        output_category_mask=True
    )
    segmentateur = vision.ImageSegmenter.create_from_options(options_seg)

    options_face = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=chemin_face),
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options_face)

    print("Fusion des IA activée ! Appuyez sur 'q' pour quitter.")

    IDX_MENTON = 152

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image = cv2.flip(image, 1)
            hauteur, largeur, _ = image.shape
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # ---------------------------------------------------------
            # ÉTAPE 1 : GÉNÉRER LA SILHOUETTE (Blanc = Personne, Noir = Fond)
            # ---------------------------------------------------------
            res_seg = segmentateur.segment(mp_image)
            if res_seg.category_mask is not None:
                masque = res_seg.category_mask.numpy_view()
                # INVERSION PARFAITE : Fond Noir (0), Silhouette Blanche (255)
                masque_visuel = np.where(masque > 0, 0, 255).astype(np.uint8)
            else:
                masque_visuel = np.zeros((hauteur, largeur), dtype=np.uint8)

            # ---------------------------------------------------------
            # ÉTAPE 2 : SOUSTRAIRE LE VISAGE ET LE CORPS (Noir)
            # ---------------------------------------------------------
            res_face = landmarker.detect(mp_image)
            
            if res_face.face_landmarks and len(res_face.face_landmarks) > 0:
                landmarks_du_visage = res_face.face_landmarks[0]
                
                # A. Effacer le corps (couper tout ce qui est sous la tête)
                try:
                    pt_menton = landmarks_du_visage[IDX_MENTON]
                    val_y_menton = pt_menton.y if hasattr(pt_menton, 'y') else pt_menton[1]
                    y_menton = int(float(val_y_menton) * hauteur)
                    
                    masque_visuel[y_menton:, :] = 0 # Tranchage NumPy
                except Exception:
                    pass # Sécurité en cas d'erreur de détection du menton

                # B. Effacer le visage (dessiner l'ovale en Noir)
                points_visage = []
                for pt in landmarks_du_visage:
                    # Extraction hyper-robuste des coordonnées
                    if hasattr(pt, 'x') and hasattr(pt, 'y'):
                        val_x = pt.x
                        val_y = pt.y
                    else:
                        val_x = pt[0]
                        val_y = pt[1]
                    
                    # Conversion forcée en nombre décimal puis entier
                    x = int(float(val_x) * largeur)
                    y = int(float(val_y) * hauteur)
                    points_visage.append([x, y])
                
                points_visage = np.array(points_visage, dtype=np.int32)
                contour_visage = cv2.convexHull(points_visage)
                
                # Remplissage du visage en Noir (0)
                cv2.fillConvexPoly(masque_visuel, contour_visage, 0)

            # Affichage
            cv2.imshow('Camera Originale', image)
            cv2.imshow('Masque des Cheveux Pur', masque_visuel)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        segmentateur.close()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    masque_cheveux_camera()