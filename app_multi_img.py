import bib
import os
import torch
import threading
import faiss
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk




# ---- Configuration de la charte graphique globale de l'interface ----




COLOR_BG = "#0A0512"
COLOR_CARD = "#140A24"
COLOR_PRIMARY = "#F472B6"
COLOR_PRIMARY_HOVER = "#D91B75"
COLOR_ACCENT = "#A855F7"
COLOR_ACCENT_BG = "#25113B"
COLOR_TEXT_MAIN = "#FDF2F8"
COLOR_TEXT_MUTED = "#C084FC"
COLOR_BORDER = "#FF2E93"
COLOR_BORDER_ACTIVE = "#10B981"
COLOR_ERROR = "#FF4B4B"




# Définition de chemin d'accès au fichier des descripteurs
FICHIER_PTH = "vecteurs_image.pth"








class ImageSearchAPP(tk.Tk) :
    def __init__(self) :
        # initialisation et configuration géométrique de la fenêtre principale
        super().__init__()
        self.title("Recherche d'Image Multimodale avec CLIP")
        self.geometry("1000x820")
        self.minsize(800,600)
        self.configure(bg=COLOR_BG)




        # Attributs d'état de l'application
        self.embeddings = None
        self.image_names =[]
        self.cashed_photos = []
        self.last_results = []
        self.resize_debounce_id = None
        self.index = None
        self.dim_embedding = None
        self.setup()
        self.load_embeddings()








    def setup (self) :
        #Conteneur supérieur dédié aux paramètre et à la barre de recherche
        top_panel = tk.Frame ( self,bg=COLOR_BG,padx=40,pady=25)
        top_panel.pack(fill="x",side="top")
        #Label de titre de l'application
        title = tk.Label( top_panel, text="Find your images, just describe it",fg = COLOR_TEXT_MAIN , bg = COLOR_BG , font = ("Segoe UI",25,"italic"))
        title.pack(pady=(0,15))
        search = tk.Frame ( top_panel, bg = COLOR_CARD , highlightbackground=COLOR_BORDER, highlightthickness=1 )
        search.pack(fill="x",ipady=3)
       
        # champs de saisie du texte de l'utilisateur (Touche Entrée)
        self.search_entry = tk.Entry ( search,bg=COLOR_CARD,fg = COLOR_TEXT_MUTED , bd = 0 , highlightthickness=0 , highlightbackground=COLOR_PRIMARY,font=("Segoe UI",15,"bold"))
        self.search_entry.pack(side="left",fill="x",expand=True,padx=18,pady=10)
        self.search_entry.insert(0,"")
        self.search_entry.bind("<Return>", lambda e : self.handle_search())
        self.search_entry.bind("<FocusIn>", self.clear_placeholder )
        self.search_entry.bind("<FocusOut>",  self.restore_placeholder)




        # Module de configuration du paramètre de voisinage K
        k_param = tk.Frame(search,bg=COLOR_CARD,padx=15)
        k_param.pack(side="left")
        k_lbl = tk.Label ( k_param,text="How many ? ",bg=COLOR_CARD,fg=COLOR_TEXT_MAIN,font=("Segoe UI",12,"italic"))
        k_lbl.pack(side="left",padx=15)
        self.k_entry = tk.Entry( k_param,bg=COLOR_PRIMARY,fg=COLOR_TEXT_MAIN,font=("Segoe UI",15,"bold"),insertbackground=COLOR_PRIMARY,width=4,bd=0,highlightbackground=COLOR_BORDER,highlightthickness=1,justify="center")
        # par défaut initialisation de la valeur a K=1
        self.k_entry.insert(0,"1")
        self.k_entry.pack(side="left",padx=5,pady=3)




        #Bouton de déclenchement de la requête de recherche
        self.submit_bttn = tk.Button ( search, text= "SEARCH" , bg=COLOR_PRIMARY,fg="white",font=("Segoe UI",11,"bold"),bd=0,padx=25,pady=8,cursor="hand2",activebackground=COLOR_PRIMARY_HOVER,activeforeground="white",command=self.handle_search)
        self.submit_bttn.pack(side="right",padx=6,pady=6)
        self.submit_bttn.bind("<Enter>",lambda e : self.submit_bttn.config(bg=COLOR_PRIMARY_HOVER))
        self.submit_bttn.bind("<Leave>",lambda e : self.submit_bttn.config(bg=COLOR_PRIMARY))
       




        # indicateur textuelle de l'état d'avancement des processus en arrière-plan
        self.statut_text = tk.StringVar (value="... Connexion With CLIP ...")
        self.statut_lbl = tk.Label(top_panel,textvariable=self.statut_text,bg=COLOR_BG,fg=COLOR_TEXT_MUTED,font=("Segoe UI",9,"italic"))
        self.statut_lbl.pack(anchor="w",pady=(8,0))




        # Zone d'affichage dynamique de la grille des photos intégrant une barre de défilement
        main_content= tk.Frame(self,bg=COLOR_BG,padx=30)
        main_content.pack(fill="both",expand=True)
        self.canvas = tk.Canvas(main_content,bg=COLOR_BG,highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(main_content,orient="vertical",command=self.canvas.yview)
        self.results_grid = tk.Frame(self.canvas,bg=COLOR_BG)
        self.results_grid.bind("<Configure>", lambda e : self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((500,0),window=self.results_grid,anchor="n")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left",fill="both",expand=True)
        self.scrollbar.pack(side="right",fill="y")




        #Liaison de l'événement de défilement à la molette de la souris
        self.canvas.bind_all("<MouseWheel>",self.on_mousewheel)




        # Affichage d'un écran d'accueil par défaut avant l'exécution de la première requête
        self.placeholder= tk.Label(self.results_grid,text="Everything is ready\n Write a description, choose the number of photos and press Search",bg=COLOR_BG,fg=COLOR_TEXT_MUTED,font=("Segoe UI",13),justify="center")
        self.placeholder.pack(pady=120,expand=True,anchor="center")
        self.bind("<Configure>",self.on_window_resize)




    def on_mousewheel(self,event) :
        self.canvas.yview_scroll(int(-1*(event.delta/120)),"units")




    # Gestion d'affichage du texte d'invite (placeholder) dans le champs de la recherche
    def clear_placeholder (self,event) :
        if self.search_entry.get() == "" :
            self.search_entry.delete(0,"end")
            self.search_entry.config(fg=COLOR_TEXT_MAIN)
   
    # Restauration de la valeur par défaut en cas d'absence d'entrée utilisateur
    def restore_placeholder ( self , event ) :
        if not self.search_entry.get().strip() :
            self.search_entry.insert(0,"a dog")
            self.search_entry.config(fg=COLOR_TEXT_MAIN)




   # Chargement asynchrone des plongements vectoriels (embeddings) et instanciation de l'index de recherche FAISS
    def load_embeddings (self) :
        def worker() :
            try :
                if not os.path.exists(FICHIER_PTH) :
                    self.statut_text.set("Indexation of the photos...")
                    # extraction des vecteurs embeddings
                    bib.extraction_vecteur_img()
                self.embeddings = torch.load(FICHIER_PTH,weights_only=False)
                # Construction de la matrice par empilement les vecteurs , taille : n*512
                self.image_names = list(self.embeddings.keys())
                matrix = torch.stack([self.embeddings[name] for name in self.image_names])
                #Normalisation L2 de la matrice et conversion au format NumPy float32 requis par FAISS
                self.dim_embedding = matrix.shape[1]
                matrix = matrix.view(-1,self.dim_embedding)
                matrix_np = matrix.numpy().astype("float32")
                faiss.normalize_L2(matrix_np)
                # Instanciation de l'index de produit scalaire Flat IP (Inner Product) pour une recherche exhaustive et optimisée
                self.index = faiss.IndexFlatIP ( self.dim_embedding)
                self.index.add(matrix_np)
                total=len(self.image_names)
                self.statut_text.set(f"system connected - we have {total} images")
                self.statut_lbl.config(fg=COLOR_PRIMARY)
            except Exception as err :
                self.statut_text.set(f"Err CLIP : {err}")
                self.statut_lbl.config(fg=COLOR_ERROR)




        threading.Thread(target=worker,daemon=True).start()
   
    # Récupération de la requête de l'utilisateur, validation du paramètre K et exécution asynchrone de la recherche de similarité
    def handle_search(self) :    
        query = self.search_entry.get().strip()
        if not query or query == "" :
            return
        if self.embeddings is None :
            self.statut_text.set("Just wait a few seconds ... ")
            return
        #Validation et conversion sécurisée de la valeur K (Par défaut k=1)
        k_input = self.k_entry.get().strip()
        try :
            k = int(k_input)
            if k <=0 :
                k = 1
        except ValueError :
            k = 1
        self.submit_bttn.config(state="disabled",text="Calcul...")
        self.statut_text.set("Matrix sorting and cosine similarity calculations...")




        def process() :
            try :
                #encodage de texte via CLIP
                text_vector = bib.get_embd_txt(query).view(1,-1)
                #Normalisation L2 du vecteur requête pour assurer une équivalence de calcul de la similarité cosinus via le produit scalaire
                text_np = text_vector.numpy().astype("float32")
                faiss.normalize_L2(text_np)
                #extraction des top k images
                k_final = min (k,len(self.image_names))
                top_scores , top_indices = self.index.search(text_np,k_final)
                self.last_results = []
                for score , idx in zip(top_scores[0],top_indices[0]) :
                    self.last_results.append({ "name" : self.image_names[idx] , "score" : score.item()})
                self.after(0,lambda : self.render_grid(query))
            except Exception as err :
                self.after(0,lambda : self.show_error_message(str(err)))




        threading.Thread(target=process,daemon=True).start()




    def render_grid ( self,query="") :
        if not self.last_results:
            return
        self.submit_bttn.config(state="normal",text="Discover")
        if query :
            self.statut_text.set(f"Results of << {query} >>")
            self.statut_lbl.config(fg=COLOR_PRIMARY)




        #Réinitialisation du conteneur graphique avant l'affichage des nouveaux résultats
        for child in self.results_grid.winfo_children() :
            child.destroy()
        self.cashed_photos.clear()
         
        #Calcul du nombre optimal de colonnes selon la largeur courante de la fenêtre
        width = self.canvas.winfo_width()
        if width < 450 :
            columns = 1
        elif width < 780 :
            columns = 2
        elif width < 1100 :
            columns = 3
        else :
            columns = 4




        for col in range(columns) :
            self.results_grid.columnconfigure(col,weight=1,uniform="col_uniform")
        #génération de la galerie des photos à renvoyer
        for index, item in enumerate(self.last_results) :
            filename=item["name"]
            score = item["score"]
            card_frame = tk.Frame(self.results_grid,bg=COLOR_BG,highlightbackground=COLOR_BORDER,highlightthickness=1,bd=0)
            row = index // columns
            col = index % columns
            card_frame.grid ( row=row , column=col,padx=14,pady=16,sticky="nswe")
            card_frame.bind("<Enter>" , lambda e , f = card_frame: f.config(highlightbackground=COLOR_BORDER_ACTIVE))
            card_frame.bind("<Leave>",lambda e , f = card_frame: f.config(highlightbackground=COLOR_BORDER) )
            #Traitement d'image : ouverture , redimensionnement et mise en cache mémoire
            img_path = os.path.join("Images",filename)
            try :
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((255,245),Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.cashed_photos.append(photo)
                img_lbl = tk.Label(card_frame,image=photo,bg=COLOR_CARD)
                img_lbl.pack(fill="x",padx=8,pady=(12,4))
                img_lbl.bind("<Enter>",lambda e , f=card_frame: f.config(highlightbackground=COLOR_BORDER_ACTIVE))
                img_lbl.bind("<Leave>",lambda e , f=card_frame: f.config(highlightbackground=COLOR_BORDER))
            except Exception :
                err_placeholder =tk.Label ( card_frame,text="Image unattainable",fg=COLOR_ERROR,bg=COLOR_CARD,font=("Segoe UI",10))
                err_placeholder.pack(pady=40)
           
            # Affichage du rang et du pourcentage
            pourcentage = int (score*100)
            text_color = COLOR_PRIMARY if score > 0.26 else COLOR_ACCENT
            rang = tk.Label(card_frame,text=f"RANK #{index+1}  -  Match : {pourcentage}%",bg=COLOR_ACCENT_BG,fg=text_color,font=("Segoe UI",9,"bold"),padx=14,pady=6)
            rang.pack(pady=(6,12))
        #Repositionnement automatique du défilement au sommet a chaque rafraichissement
        self.canvas.yview_moveto(0)




    # Affichage de la fenêtre dynamique lors du redimensionnement
    def on_window_resize (self,event) :
      if event.widget == self and self.last_results :
        if self.resize_debounce_id is not None :
             self.after_cancel(self.resize_debounce_id)
        self.resize_debounce_id = self.after(150,self.render_grid)
   
    # envoyer les messages d'erreurs
    def show_error_message(self,msg) :
      self.submit_bttn.config(state="normal",text="Discover")
      self.statut_text.set(f"Error : {msg}")
      self.statut_lbl.config(fg=COLOR_ERROR)




# Point d'entrée principale de l'application et le lancement en loop
if __name__ == "__main__" :
  app = ImageSearchAPP()
  app.mainloop()


