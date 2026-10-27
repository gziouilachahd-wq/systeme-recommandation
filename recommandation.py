import os
import psycopg2
import nltk
import numpy as np
from nltk.stem.snowball import FrenchStemmer
from nltk.corpus import stopwords
from datetime import date

nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

DB_CONFIG = {
    "dbname":   "recommandation",
    "user":     "postgres",
    "password": os.getenv("DB_PASSWORD", "votre_mot_de_passe"),        
    "host":     "localhost",
    "port":     "5432",
    "options":  "-c client_encoding=UTF8"
}



def initialiser_base():
    produits = [
        (1,  "Ordinateur portable",    "Ordinateur portable léger performant idéal pour le travail et les études avec écran HD"),
        (2,  "Tablette tactile",        "Tablette tactile légère avec grand écran idéale pour lire regarder des vidéos et naviguer"),
        (3,  "Smartphone",              "Téléphone intelligent avec appareil photo haute résolution et grande autonomie de batterie"),
        (4,  "Clavier mécanique",       "Clavier mécanique rétroéclairé pour ordinateur de bureau idéal pour les joueurs et développeurs"),
        (5,  "Souris sans fil",         "Souris ergonomique sans fil compatible avec ordinateur portable et bureau longue autonomie"),
        (6,  "Casque audio",            "Casque audio sans fil avec réduction de bruit idéal pour musique et appels téléphoniques"),
        (7,  "Écran PC",                "Écran plat haute définition pour ordinateur de bureau avec dalle IPS et résolution 4K"),
        (8,  "Disque dur externe",      "Disque dur externe portable grande capacité pour sauvegarder photos vidéos et documents"),
        (9,  "Clé USB",                 "Clé USB rapide grande capacité idéale pour transférer fichiers entre ordinateurs"),
        (10, "Imprimante",              "Imprimante laser rapide idéale pour imprimer documents et photos de haute qualité"),
        (11, "Webcam HD",               "Webcam haute définition pour vidéoconférence et appels vidéo avec micro intégré"),
        (12, "Haut-parleur Bluetooth",  "Haut-parleur portable Bluetooth avec son haute qualité idéal pour musique en extérieur"),
        (13, "Montre connectée",        "Montre intelligente avec suivi sportif notifications téléphone et autonomie longue durée"),
        (14, "Chargeur rapide",         "Chargeur rapide compatible avec tous les smartphones et tablettes avec câble USB inclus"),
        (15, "Support téléphone",       "Support réglable pour téléphone portable et tablette idéal pour bureau et voiture"),
        (16, "Batterie portable",       "Batterie externe portable grande capacité pour recharger smartphones et tablettes en déplacement"),
        (17, "Routeur WiFi",            "Routeur WiFi haute vitesse pour connexion internet rapide et stable dans toute la maison"),
        (18, "Lampe de bureau LED",     "Lampe LED pour bureau réglable avec luminosité variable idéale pour travailler et étudier"),
        (19, "Sac à dos laptop",        "Sac à dos imperméable avec compartiment pour ordinateur portable et tablette de voyage"),
        (20, "Tapis de souris XXL",     "Grand tapis de souris pour bureau idéal pour ordinateur gamer et travail confortable"),
    ]

    utilisateurs = [
        (1,"Malika"),(2,"Sarra"),(3,"Ahmed"),(4,"Yasmine"),(5,"Omar"),
        (6,"Nour"),(7,"Karim"),(8,"Lina"),(9,"Mehdi"),(10,"Ines"),
    ]

    interactions = [
        (1,1,5,"2024-01-10"),(1,2,4,"2024-01-15"),(1,5,3,"2024-02-01"),
        (1,8,4,"2024-02-20"),(1,9,5,"2024-03-05"),
        (2,1,4,"2024-01-12"),(2,2,5,"2024-01-20"),(2,5,4,"2024-02-05"),
        (2,6,3,"2024-02-15"),(2,11,4,"2024-03-01"),
        (3,6,5,"2024-01-08"),(3,12,5,"2024-01-25"),(3,3,4,"2024-02-10"),
        (3,13,3,"2024-02-28"),(3,14,2,"2024-03-10"),
        (4,6,4,"2024-01-18"),(4,12,5,"2024-02-02"),(4,3,5,"2024-02-18"),
        (4,16,4,"2024-03-08"),(4,15,3,"2024-03-20"),
        (5,4,5,"2024-01-05"),(5,5,5,"2024-01-22"),(5,7,4,"2024-02-08"),
        (5,20,5,"2024-02-25"),(5,18,3,"2024-03-15"),
        (6,4,4,"2024-01-14"),(6,7,5,"2024-02-03"),(6,20,4,"2024-02-22"),
        (6,1,3,"2024-03-02"),(6,9,4,"2024-03-18"),
        (7,8,5,"2024-01-09"),(7,9,4,"2024-01-28"),(7,10,5,"2024-02-12"),
        (7,17,3,"2024-03-01"),(7,11,4,"2024-03-22"),
        (8,8,4,"2024-01-16"),(8,10,5,"2024-02-06"),(8,9,3,"2024-02-24"),
        (8,2,4,"2024-03-12"),(8,18,5,"2024-03-28"),
        (9,17,5,"2024-01-11"),(9,14,4,"2024-01-30"),(9,16,5,"2024-02-14"),
        (9,3,4,"2024-03-04"),(9,13,3,"2024-03-25"),
        (10,17,4,"2024-01-19"),(10,16,5,"2024-02-07"),(10,14,4,"2024-02-26"),
        (10,15,3,"2024-03-14"),(10,19,4,"2024-03-30"),
    ]

    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS Interaction;")
    cursor.execute("DROP TABLE IF EXISTS Utilisateur;")
    cursor.execute("DROP TABLE IF EXISTS Produit;")

    cursor.execute("""
        CREATE TABLE Produit (
            id          INTEGER PRIMARY KEY,
            nomPdt      VARCHAR(100) NOT NULL,
            description TEXT         NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE Utilisateur (
            id  INTEGER PRIMARY KEY,
            nom VARCHAR(50) NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE Interaction (
            id_user          INTEGER NOT NULL,
            id_produit       INTEGER NOT NULL,
            note             INTEGER CHECK (note BETWEEN 1 AND 5),
            date_interaction DATE    NOT NULL
        );
    """)

    for p in produits:
        cursor.execute("INSERT INTO Produit VALUES (%s, %s, %s);", p)
    for u in utilisateurs:
        cursor.execute("INSERT INTO Utilisateur VALUES (%s, %s);", u)
    for i in interactions:
        cursor.execute("INSERT INTO Interaction VALUES (%s, %s, %s, %s);", i)

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ {len(produits)} produits, {len(utilisateurs)} utilisateurs, {len(interactions)} interactions insérés.\n")




def charger_produits():
    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id, nompdt, description FROM produit ORDER BY id;")
    Produits = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"✅ {len(Produits)} produits récupérés.")
    return Produits

def charger_utilisateurs():
    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id, nom FROM utilisateur ORDER BY id;")
    utilisateurs = cursor.fetchall()
    cursor.close()
    conn.close()
    return utilisateurs

def charger_interactions():
    conn   = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id_user, id_produit, note, date_interaction FROM interaction;")
    interactions = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"✅ {len(interactions)} interactions récupérées.")
    return interactions




def traitement_nlp(Produits):
    stemer   = FrenchStemmer()
    StopList = set(stopwords.words('french'))
    StopList = StopList | {".", ",", ";", ":", "!", "?", "'", '"', "-", "(", ")"}

    dictProduits     = {}
    ListTotaliteMots = set()

    print("=" * 60)
    print("PRÉTRAITEMENT NLP DES DESCRIPTIONS")
    print("=" * 60)

    for p in Produits:
        idPdt       = int(p[0])
        Description = p[2]

        Mots          = nltk.word_tokenize(Description, language='french')
        MotsStems     = [stemer.stem(m) for m in Mots]
        ListFinalMots = [m for m in MotsStems if m not in StopList]

        dictProduits[idPdt] = ListFinalMots
        for m in ListFinalMots:
            ListTotaliteMots.add(m)

        print(f"  Produit {idPdt:2d} | {p[1]:<25} → {ListFinalMots}")

    TotaliteMots = sorted(list(ListTotaliteMots))
    print(f"\n✅ Vocabulaire total : {len(TotaliteMots)} mots uniques.\n")
    return dictProduits, TotaliteMots




def construire_matrice_binaire(dictProduits, TotaliteMots):
    NbProduits     = len(dictProduits)
    NbMots         = len(TotaliteMots)
    matriceBinaire = np.zeros((NbProduits, NbMots))

    for i in range(NbProduits):
        j = 0
        for m in TotaliteMots:
            if m in dictProduits[(i + 1)]:
                matriceBinaire[i][j] = 1
            j += 1

    print("=" * 60)
    print(f"MATRICE BINAIRE ({NbProduits} x {NbMots})")
    print("=" * 60)
    print(matriceBinaire[:5, :10])
    print()
    return matriceBinaire




def SimilariteCosinus(i, j, matriceBinaire):
    a  = matriceBinaire[i]
    b  = matriceBinaire[j]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return np.dot(a, b) / (na * nb)

def distance_euclidienne(i, j, matriceBinaire):
    a  = matriceBinaire[i]
    b  = matriceBinaire[j]
    de = np.linalg.norm(a - b)
    return 1 / (de + 1)

def similarite_jaccard(i, j, matriceBinaire):
    a            = matriceBinaire[i]
    b            = matriceBinaire[j]
    intersection = np.dot(a, b)
    union        = np.sum(a) + np.sum(b) - intersection
    if union == 0:
        return 0.0
    return intersection / union




def construire_matrices_similarite(matriceBinaire):
    NbProduits         = matriceBinaire.shape[0]
    matriceCosinus     = np.zeros((NbProduits, NbProduits))
    matriceEuclidienne = np.zeros((NbProduits, NbProduits))
    matriceJaccard     = np.zeros((NbProduits, NbProduits))

    for i in range(NbProduits):
        for j in range(NbProduits):
            matriceCosinus[i][j]     = SimilariteCosinus(i, j, matriceBinaire)
            matriceEuclidienne[i][j] = distance_euclidienne(i, j, matriceBinaire)
            matriceJaccard[i][j]     = similarite_jaccard(i, j, matriceBinaire)

    print("=" * 60)
    print("MATRICES DE SIMILARITÉ (extrait 5x5)")
    print("=" * 60)
    print("Cosinus :")
    print(np.round(matriceCosinus[:5, :5], 2))
    print("\nEuclidienne :")
    print(np.round(matriceEuclidienne[:5, :5], 2))
    print("\nJaccard :")
    print(np.round(matriceJaccard[:5, :5], 2))
    print()
    return matriceCosinus, matriceEuclidienne, matriceJaccard




def recommander_contenu(idPdt_cible, N, Produits,
                        matriceCosinus, matriceEuclidienne, matriceJaccard):
    index_cible = idPdt_cible - 1
    nom_cible   = Produits[index_cible][1]

    print("=" * 60)
    print(f"CONTENT-BASED : « {nom_cible} » (ID={idPdt_cible})")
    print("=" * 60)

    resultats = {}
    for nom_mesure, matrice in [
        ("Cosinus",     matriceCosinus),
        ("Euclidienne", matriceEuclidienne),
        ("Jaccard",     matriceJaccard)
    ]:
        scores       = list(enumerate(matrice[index_cible]))
        scores_tries = sorted(scores, key=lambda x: x[1], reverse=True)
        top_n        = [(idx, s) for idx, s in scores_tries if idx != index_cible][:N]

        print(f"\n  → Méthode {nom_mesure} :")
        liste_resultats = []
        for rang, (idx, score) in enumerate(top_n, 1):
            nom = Produits[idx][1]
            if score > 0.5:
                explication = "Très similaire au produit sélectionné"
            elif score > 0.2:
                explication = "Recommandé car similaire à vos préférences de contenu"
            else:
                explication = "Légèrement similaire au produit sélectionné"
            print(f"    {rang}. [{score:.4f}] {nom} → {explication}")
            liste_resultats.append({"produit": nom, "score": round(score, 4), "explication": explication})
        resultats[nom_mesure] = liste_resultats

    print()
    return resultats



def construire_matrice_notes(interactions):
    nb_users      = max(row[0] for row in interactions)
    nb_prods      = max(row[1] for row in interactions)
    matrice_notes = np.zeros((nb_users, nb_prods))
    for id_user, id_produit, note, _ in interactions:
        matrice_notes[id_user - 1][id_produit - 1] = note
    print("=" * 60)
    print(f"MATRICE NOTES ({nb_users} utilisateurs x {nb_prods} produits)")
    print("=" * 60)
    print(np.round(matrice_notes, 1))
    print()
    return matrice_notes

def recommander_collaboratif(id_user_cible, N, matrice_notes, Produits):
    idx_cible = id_user_cible - 1
    vecteur   = matrice_notes[idx_cible]

    similarites = []
    for i in range(matrice_notes.shape[0]):
        if i != idx_cible:
            n1  = np.linalg.norm(vecteur)
            n2  = np.linalg.norm(matrice_notes[i])
            sim = 0.0 if n1 == 0 or n2 == 0 else np.dot(vecteur, matrice_notes[i]) / (n1 * n2)
            similarites.append((i, sim))

    similarites     = sorted(similarites, key=lambda x: x[1], reverse=True)
    top_users       = similarites[:3]
    deja_notes      = set(j for j in range(matrice_notes.shape[1]) if vecteur[j] > 0)
    scores_produits = {}

    for idx_user, sim in top_users:
        for j in range(matrice_notes.shape[1]):
            note = matrice_notes[idx_user][j]
            if j not in deja_notes and note > 0:
                if j not in scores_produits:
                    scores_produits[j] = 0
                scores_produits[j] += sim * note

    recommandations = sorted(scores_produits.items(), key=lambda x: x[1], reverse=True)[:N]

    print("=" * 60)
    print(f"COLLABORATIVE FILTERING — Utilisateur {id_user_cible}")
    print("=" * 60)
    print("  Voisins similaires :")
    for idx, sim in top_users:
        print(f"    → Utilisateur {idx+1} (similarité = {sim:.4f})")
    print(f"\n  Top {N} recommandations :\n")

    resultats = []
    for rang, (idx_produit, score) in enumerate(recommandations, 1):
        nom = Produits[idx_produit][1]
        explication = "Très apprécié par des utilisateurs similaires" if score > 3 else "Recommandé car des utilisateurs similaires ont aimé cet item"
        print(f"  {rang}. {nom} (score={score:.4f}) → {explication}")
        resultats.append({"produit": nom, "score": round(score, 4), "explication": explication})

    print()
    return resultats




def recommander_temps(id_user_cible, N, interactions, Produits):
    aujourd_hui  = date.today()
    deja_notes   = set(row[1] for row in interactions if row[0] == id_user_cible)
    scores_temps = {}

    for id_user, id_produit, note, date_inter in interactions:
        if id_user != id_user_cible and id_produit not in deja_notes:
            nb_jours = (aujourd_hui - date_inter).days
            poids    = 1.0 / (nb_jours + 1)
            if id_produit not in scores_temps:
                scores_temps[id_produit] = 0
            scores_temps[id_produit] += note * poids

    recommandations = sorted(scores_temps.items(), key=lambda x: x[1], reverse=True)[:N]

    print("=" * 60)
    print(f"TIME-AWARE — Utilisateur {id_user_cible}")
    print("=" * 60)

    resultats = []
    for rang, (id_produit, score) in enumerate(recommandations, 1):
        nom         = Produits[id_produit - 1][1]
        explication = "Recommandé car récent ou tendance"
        print(f"  {rang}. {nom} (score={score:.6f}) → {explication}")
        resultats.append({"produit": nom, "score": round(score, 6), "explication": explication})

    print()
    return resultats




def recommander_hybride(id_user_cible, idPdt_ref, N,
                        matriceCosinus, matrice_notes,
                        interactions, Produits,
                        w_contenu=0.4, w_collab=0.4, w_temps=0.2):
    aujourd_hui   = date.today()
    NbProduits    = len(Produits)
    scores_finaux = {}

    idx_ref = idPdt_ref - 1
    for j in range(NbProduits):
        if j != idx_ref:
            scores_finaux[j + 1] = {"contenu": matriceCosinus[idx_ref][j], "collab": 0.0, "temps": 0.0}

    idx_u   = id_user_cible - 1
    vecteur = matrice_notes[idx_u]
    sims    = []
    for i in range(matrice_notes.shape[0]):
        if i != idx_u:
            n1  = np.linalg.norm(vecteur)
            n2  = np.linalg.norm(matrice_notes[i])
            sim = 0.0 if n1 == 0 or n2 == 0 else np.dot(vecteur, matrice_notes[i]) / (n1 * n2)
            sims.append((i, sim))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)

    for idx_uu, sim in sims[:3]:
        for j in range(NbProduits):
            note = matrice_notes[idx_uu][j]
            if note > 0 and (j + 1) in scores_finaux:
                scores_finaux[j + 1]["collab"] += sim * (note / 5.0)

    scores_t = {}
    for _, id_p, note, date_inter in interactions:
        nb_jours = (aujourd_hui - date_inter).days
        if id_p not in scores_t:
            scores_t[id_p] = 0
        scores_t[id_p] += note / (nb_jours + 1)

    max_t = max(scores_t.values()) if scores_t else 1
    for id_p, s in scores_t.items():
        if id_p in scores_finaux:
            scores_finaux[id_p]["temps"] = s / max_t

    resultats_raw = sorted(
        [(id_p, w_contenu*s["contenu"] + w_collab*s["collab"] + w_temps*s["temps"]) for id_p, s in scores_finaux.items()],
        key=lambda x: x[1], reverse=True
    )

    print("=" * 60)
    print(f"MODÈLE HYBRIDE — Utilisateur {id_user_cible} | Produit ref: {idPdt_ref}")
    print("Poids : Contenu=0.4 | Collaboratif=0.4 | Temps=0.2")
    print("=" * 60)

    resultats = []
    for rang, (id_p, score) in enumerate(resultats_raw[:N], 1):
        nom         = Produits[id_p - 1][1]
        explication = "Combinaison : contenu + utilisateurs similaires + tendances récentes"
        print(f"  {rang}. {nom} (score={score:.4f}) → {explication}")
        resultats.append({"produit": nom, "score": round(score, 4), "explication": explication})

    print()
    return resultats



if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  SYSTÈME DE RECOMMANDATION - 2 LNBI - ISG TUNIS")
    print("=" * 60 + "\n")

    initialiser_base()

    Produits     = charger_produits()
    utilisateurs = charger_utilisateurs()
    interactions = charger_interactions()

    dictProduits, TotaliteMots = traitement_nlp(Produits)
    matriceBinaire             = construire_matrice_binaire(dictProduits, TotaliteMots)

    matriceCosinus, matriceEuclidienne, matriceJaccard = construire_matrices_similarite(matriceBinaire)

    matrice_notes = construire_matrice_notes(interactions)

    print("\n🎯 GÉNÉRATION DES RECOMMANDATIONS\n")

    recommander_contenu(1, N=3, Produits=Produits,
                        matriceCosinus=matriceCosinus,
                        matriceEuclidienne=matriceEuclidienne,
                        matriceJaccard=matriceJaccard)

    recommander_collaboratif(1, N=3, matrice_notes=matrice_notes, Produits=Produits)
    recommander_collaboratif(2, N=3, matrice_notes=matrice_notes, Produits=Produits)

    recommander_temps(1, N=3, interactions=interactions, Produits=Produits)

    recommander_hybride(1, 1, N=3,
                        matriceCosinus=matriceCosinus,
                        matrice_notes=matrice_notes,
                        interactions=interactions,
                        Produits=Produits)

    print("✅ Terminé avec succès !")