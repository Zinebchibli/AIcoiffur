import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

def telecharger_modele():
    """Télécharge le modèle IA de segmentation s'il n'est pas déjà là."""
    nom_fichier = "selfie_segmenter.tflite"
    url = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    
    if not os.path.exists(nom_fichier):
        print("Téléchargement du modèle IA (cela peut prendre quelques secondes)...")
        urllib.request.urlretrieve(url, nom_fichier)
        print("Téléchargement terminé !")
        
    return nom_fichier

def tester_masque_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la camera.")

    # 1. Configuration de la nouvelle API MediaPipe
    chemin_modele = telecharger_modele()
    base_options = python.BaseOptions(model_asset_path=chemin_modele)
    
    # output_category_mask=True indique qu'on veut une carte binaire (fond vs personne)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        output_category_mask=True
    )
    segmentateur = vision.ImageSegmenter.create_from_options(options)

    print("Caméra activée. Appuyez sur 'q' pour quitter.")

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Effet miroir et conversion des couleurs
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 2. Formatage pour l'API MediaPipe Tasks
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # 3. Prédiction de la silhouette
            resultats = segmentateur.segment(mp_image)

            # 4. Création du masque visuel noir et blanc
            if resultats.category_mask is not None:
                # Le numpy_view() renvoie un tableau où 0 = personne, >0 = fond
                masque = resultats.category_mask.numpy_view()
                
                # On détecte le fond
                masque_binaire = masque > 0
                
                # INVERSION ICI : on met 0 (Noir) pour le fond, et 255 (Blanc) pour la personne
                masque_visuel = np.where(masque_binaire, 0, 255).astype(np.uint8)
            else:
                masque_visuel = np.zeros(image.shape[:2], dtype=np.uint8)

            # Affichage
            cv2.imshow('Camera Originale', image)
            cv2.imshow('Masque Genere (Noir et Blanc)', masque_visuel)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        segmentateur.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tester_masque_camera()