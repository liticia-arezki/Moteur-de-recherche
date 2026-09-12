import bib
import os
import torch
import faiss
import numpy as np


print ("\n-------------- text - Image ( avec FAISS ) ------------------\n")
# on va chercher l'image la plus proche a ce text
txt = input ("\nDonnez un text :")
k= int (input ("\nCombien de photos vous voulez :"))
fichier ="vecteurs_image.pth"
# on a déja tout nous vecteur des images dans un seul fichier q'uon va récupérer
if not os.path.exists(fichier) :
   bib.extraction_vecteur_img ()
image_vecteurs=torch.load(fichier,weights_only=False)
# on enregistre notre vecteur pour le text
v1 = bib.get_embd_txt (txt)
#forcer le vecteur a etre en 2D pour faire uen ligen ( 1 , 512 )
v1=v1.view(1,-1)
#sauvegarder la taille de vecteur pour la suite 
dim_embedding = v1.shape[1]
#----------------------------- On utilise l'index Faiss  ------------------#
#on recupere les noms des images et leurs vecteurs
nom_images = list(image_vecteurs.keys())
if not nom_images : 
    print("fichier vide erreur\n")
    exit()
#on empile tous kes vecteurs pour créer une grande matrice 
matrice_image = torch.stack([image_vecteurs[nom] for nom in nom_images])
#par mesure de sécurité on force la matrice a etre en deux dimensions (N,512) pour faire la multiplication aprés 
matrice_image = matrice_image.view (-1,dim_embedding)

#pour utiliser FAISS on travaille avec numpy float32 et pas avec les tenseurs de PyTorch
matrice_image_np = matrice_image.numpy().astype('float32')
vecteur_text_np = v1.numpy().astype('float32')

#normalisation L2 : faiss fournit une fonction qui modifie les bleaux en place
faiss.normalize_L2(matrice_image_np)
faiss.normalize_L2(vecteur_text_np)

#IndexFlatIP recherche exacte par produit scalaire
# apres la normalisation des vecteurs on fait le produit scalaire
index = faiss.IndexFlatIP(dim_embedding)
index.add(matrice_image_np)

#on cherche les k meilleurs images 
score, index_max = index.search(vecteur_text_np,k)

meilleur_score = score[0][0]
index_m = index_max[0][0]


if meilleur_score > -1 :
    image_proche = nom_images[index_max[0][0]]
    noms_top_k = [nom_images[idx] for idx in index_max[0]]
    score_top_k = [ s for s in score[0]]
    bib.afficher_resultat(noms_top_k,score_top_k)
else : 
    print("\non a pas l'image ")