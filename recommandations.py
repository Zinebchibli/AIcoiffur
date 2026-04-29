# recommandations.py

# Dictionnaire associant les 7 formes de visage à des coupes flatteuses
CATALOGUE_COUPES = {
    "Ovale": [
        {
            "nom_fr": "Carré plongeant lisse",
            "prompt_ia": "a photorealistic woman with a sleek straight asymmetrical bob haircut, professional salon lighting, high quality, 8k"
        },
        {
            "nom_fr": "Longues ondulations",
            "prompt_ia": "a photorealistic woman with long voluminous wavy hair, beach waves, cinematic lighting, high resolution"
        }
    ],
    "Rond": [
        {
            "nom_fr": "Dégradé long avec volume",
            "prompt_ia": "a photorealistic woman with long layered hair, voluminous top, face framing layers, highly detailed"
        },
        {
            "nom_fr": "Coupe Pixie texturée",
            "prompt_ia": "a photorealistic woman with an asymmetrical textured pixie cut, modern short hair, ultra detailed"
        }
    ],
    "Carre": [
        {
            "nom_fr": "Coupe mi-longue effilée",
            "prompt_ia": "a photorealistic woman with medium length wispy layered hair, soft fringes, softening jawline, realistic"
        }
    ],
    "Coeur": [
        {
            "nom_fr": "Carré avec frange rideau",
            "prompt_ia": "a photorealistic woman with a shoulder length bob and soft curtain bangs, elegant, 4k"
        }
    ],
    "Triangle": [
        {
            "nom_fr": "Dégradé avec volume sur le dessus",
            "prompt_ia": "a photorealistic woman with a layered haircut with volume on top, light bangs, balancing jawline, beautiful"
        }
    ],
    "Rectangle": [
        {
            "nom_fr": "Ondulations souples avec frange",
            "prompt_ia": "a photorealistic woman with soft waves and straight blunt bangs, medium length, fashion photography"
        }
    ],
    "Diamant": [
        {
            "nom_fr": "Carré court derrière les oreilles",
            "prompt_ia": "a photorealistic woman with a short blunt bob tucked behind ears, sharp and modern, 8k resolution"
        }
    ],
    "Indetermine": [
         {
            "nom_fr": "Coupe classique mi-longue",
            "prompt_ia": "a photorealistic woman with classical medium length straight hair, simple and elegant"
        }
    ]
}

def obtenir_recommandations(forme_visage):
    """
    Prend en entrée la forme du visage et retourne la liste des recommandations.
    """
    # On cherche la forme dans le dictionnaire. 
    # Si elle n'existe pas (erreur), on renvoie "Indetermine" par défaut avec .get()
    recommandations = CATALOGUE_COUPES.get(forme_visage, CATALOGUE_COUPES["Indetermine"])
    
    return recommandations

# Petit test rapide pour vérifier que le fichier fonctionne tout seul
if __name__ == "__main__":
    print("--- Test du module de recommandations ---")
    forme_test = "Rond"
    resultats = obtenir_recommandations(forme_test)
    
    print(f"Pour un visage {forme_test}, les coupes suggérées sont :")
    for coupe in resultats:
        print(f"- {coupe['nom_fr']}")
        print(f"  Prompt IA caché : {coupe['prompt_ia']}\n")