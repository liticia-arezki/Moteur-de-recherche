import os
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch
from PIL import Image , ImageDraw , ImageFont
from transformers import CLIPProcessor, CLIPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"on utilise : {device}")



model = CLIPModel.from_pretrained ("openai/clip-vit-base-patch32", local_files_only=True).to(device)
processor = CLIPProcessor.from_pretrained ("openai/clip-vit-base-patch32", local_files_only=True)


def get_embd_txt ( txt : str ) :


     inputs = processor(   text = [txt] ,
                           return_tensors="pt",
                           padding = True, ).to(device)
     with torch.no_grad() :
     
         outputs= model.get_text_features(**inputs)
    
     if hasattr( outputs, "pooler_output" ) :
        vecteur = outputs.pooler_output
     elif hasattr ( outputs,"text_embeds") :
         vecteur = outputs.text_embeds
     else :
        
         vecteur = outputs.cpu()
     vecteur = vecteur / vecteur.norm(dim=-1,keepdim=True)
     return vecteur.cpu().squeeze(0)



def get_embd_img_direct (image) :
   
    inputs = processor(   images=image,
                          return_tensors="pt",
                          padding = True,   ).to(device)
    
    with torch.no_grad() :
       
        outputs= model.get_image_features(inputs["pixel_values"])
        
    if hasattr( outputs , "pooler_output" ) :
          vecteur = outputs.pooler_output.cpu()
    else :
        
         vecteur = outputs.cpu()
    vecteur = vecteur / vecteur.norm(dim=-1,keepdim=True)
    return vecteur.cpu().squeeze(0)
   



def extraction_vecteur_img () :

    dossier_images = "Images"
    
    if not os.path.exists(dossier_images):
      print ("dossier n'existe pas \n")
      exit()


  
    Liste_images = [f for f in os.listdir(dossier_images) if f.lower().endswith(('.jpg','.jpeg','.jfif','.webp','.png'))]


   
    total_images = len(Liste_images)
    if total_images == 0:
        print("on a pas d'images")
        exit()


    image_vecteurs = {}
    compteur = 0


    print ("extraction des vecteurs pour les images \n")

    with torch.no_grad() :
       
            for index , nom in enumerate(Liste_images,1) :
               
                image_path = os.path.join(dossier_images,nom)
                try :
                    
                    image = Image.open(image_path).convert("RGB")
                    vecteur = get_embd_img_direct(image)
                    image_vecteurs[nom]=vecteur
                    compteur +=1
                    if index % 10 == 0 or index == total_images:
                         print(f"progression : {index}/{total_images} image analysée : ({(index/total_images)*100:.1f}%)")
                except Exception as e :
                        print(f" erreur sur {e} \n")
                        continue
    chemin_sauvegarde = "vecteurs_image.pth"
   
    torch.save ( image_vecteurs, chemin_sauvegarde )
    print (" les vecteurs sont stocké dans : vecteurs_image.pth")
    return image_vecteurs
   

def afficher_img (nom) :
    image_path = "images"+"/"+nom
    image = Image.open(image_path)
    image.show()

def afficher_resultat (noms,scores,dossier="Images",taille = 200 ) :
     k = len(noms)
     marge=10
     zone_text=30
     largeur_totale = k*(taille+marge)+marge
     hauteur_total = taille + zone_text + 2*marge

     planche = Image.new("RGB",(largeur_totale,hauteur_total),"white")
     draw = ImageDraw.Draw (planche)

     for i, (nom ,score) in enumerate (zip(noms,scores)) :
          chemin = os.path.join(dossier,nom)
          img = Image.open(chemin).convert("RGB")
          img.thumbnail((taille,taille))

          x = marge + i * (taille+marge)
          y = marge
          planche.paste(img,(x,y))

          texte = f"{score : .3f}"
          draw.text((x,y+taille+5),texte,fill="black")
     planche.show()