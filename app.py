import os 
import streamlit as st
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


st.markdown("""
<style>
/* ── Fond général blanc/gris très clair ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f0f4ff, #faf5ff, #f0fdf4);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Sidebar claire ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(10px);
    border-right: 2px solid #e2e8f0;
}
[data-testid="stSidebar"] * { color: #1e293b !important; }

/* ── Texte général sombre ── */
html, body, [class*="css"], p, span, div, label {
    color: #1e293b;
}

/* ── Titre principal ── */
h1 {
    background: linear-gradient(90deg, #7c3aed, #2563eb, #059669);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    text-align: center;
    padding: 1rem 0;
}

/* ── Sous-titres ── */
h2, h3 { color: #7c3aed !important; font-weight: 700 !important; }

/* ── Carte méthode ── */
.method-card {
    background: rgba(255,255,255,0.9);
    border: 1.5px solid #c4b5fd;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(124,58,237,0.08);
    transition: all 0.3s ease;
}
.method-card:hover {
    border-color: #7c3aed;
    box-shadow: 0 4px 20px rgba(124,58,237,0.15);
    transform: translateY(-2px);
}

/* ── Carte résultat ── */
.result-card {
    background: rgba(255,255,255,0.95);
    border-left: 4px solid #7c3aed;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: all 0.3s ease;
}
.result-card:hover {
    box-shadow: 0 4px 16px rgba(124,58,237,0.12);
    border-left-color: #2563eb;
}
.result-rank {
    font-size: 1.4rem;
    font-weight: 800;
    color: #7c3aed;
    margin-right: 0.5rem;
}
.result-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
}
.result-score {
    display: inline-block;
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 700;
    float: right;
}
.result-expl {
    font-size: 0.88rem;
    color: #64748b;
    margin-top: 0.4rem;
    font-style: italic;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.badge-content  { background: #ede9fe; color: #6d28d9; border: 1px solid #c4b5fd; }
.badge-collab   { background: #dbeafe; color: #1d4ed8; border: 1px solid #93c5fd; }
.badge-time     { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
.badge-hybrid   { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.badge-eval     { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.95) !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 0.8rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #7c3aed !important; font-weight: 700 !important; }

/* ── Boutons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.25) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.4) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.95) !important;
    border: 1.5px solid #c4b5fd !important;
    border-radius: 10px !important;
    color: #1e293b !important;
}

/* ── Radio ── */
[data-testid="stRadio"] label { color: #1e293b !important; }

/* ── Info / Warning ── */
[data-testid="stInfo"] {
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 12px !important;
    color: #1e40af !important;
}
[data-testid="stWarning"] {
    background: #fffbeb !important;
    border: 1px solid #fcd34d !important;
    border-radius: 12px !important;
    color: #92400e !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.9) !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
}

/* ── Divider ── */
hr { border-color: #e2e8f0 !important; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
    border-radius: 10px !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label { color: #1e293b !important; }

/* ── Caption ── */
[data-testid="stCaptionContainer"] { color: #64748b !important; }

/* ── Header banner ── */
.header-banner {
    background: rgba(255,255,255,0.9);
    border: 1.5px solid #c4b5fd;
    border-radius: 16px;
    padding: 1rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(124,58,237,0.1);
}
.header-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    margin-top: 0.2rem;
}

/* ── Voisin card ── */
.voisin-card {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    color: #1e3a8a;
}

/* ── Stats globales ── */
.stat-box {
    background: rgba(255,255,255,0.95);
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.stat-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def charger_produits():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id, nompdt, description FROM produit ORDER BY id;")
    Produits = cursor.fetchall()
    cursor.close(); conn.close()
    return Produits

@st.cache_data
def charger_utilisateurs():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id, nom FROM utilisateur ORDER BY id;")
    utilisateurs = cursor.fetchall()
    cursor.close(); conn.close()
    return utilisateurs

@st.cache_data
def charger_interactions():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT id_user, id_produit, note, date_interaction FROM interaction;")
    interactions = cursor.fetchall()
    cursor.close(); conn.close()
    return interactions

@st.cache_data
def construire_matrices(Produits):
    stemer   = FrenchStemmer()
    StopList = set(stopwords.words('french'))
    StopList = StopList | {".", ",", ";", ":", "!", "?", "'", '"', "-", "(", ")"}
    dictProduits     = {}
    ListTotaliteMots = set()
    for p in Produits:
        idPdt         = int(p[0])
        Mots          = nltk.word_tokenize(p[2], language='french')
        MotsStems     = [stemer.stem(m) for m in Mots]
        ListFinalMots = [m for m in MotsStems if m not in StopList]
        dictProduits[idPdt] = ListFinalMots
        for m in ListFinalMots:
            ListTotaliteMots.add(m)
    TotaliteMots = sorted(list(ListTotaliteMots))
    NbProduits   = len(Produits)
    NbMots       = len(TotaliteMots)
    matriceBinaire = np.zeros((NbProduits, NbMots))
    for i in range(NbProduits):
        j = 0
        for m in TotaliteMots:
            if m in dictProduits[(i + 1)]:
                matriceBinaire[i][j] = 1
            j += 1
    matriceCosinus     = np.zeros((NbProduits, NbProduits))
    matriceEuclidienne = np.zeros((NbProduits, NbProduits))
    matriceJaccard     = np.zeros((NbProduits, NbProduits))
    for i in range(NbProduits):
        for j in range(NbProduits):
            a   = matriceBinaire[i]
            b   = matriceBinaire[j]
            na  = np.linalg.norm(a)
            nb_ = np.linalg.norm(b)
            matriceCosinus[i][j] = 0.0 if na==0 or nb_==0 else np.dot(a,b)/(na*nb_)
            de = np.linalg.norm(a - b)
            matriceEuclidienne[i][j] = 1 / (de + 1)
            intersection = np.dot(a, b)
            union        = np.sum(a) + np.sum(b) - intersection
            matriceJaccard[i][j] = 0.0 if union==0 else intersection/union
    return matriceCosinus, matriceEuclidienne, matriceJaccard

@st.cache_data
def construire_matrice_notes(interactions):
    interactions  = list(interactions)
    nb_users      = max(row[0] for row in interactions)
    nb_prods      = max(row[1] for row in interactions)
    matrice_notes = np.zeros((nb_users, nb_prods))
    for id_user, id_produit, note, _ in interactions:
        matrice_notes[id_user - 1][id_produit - 1] = note
    return matrice_notes


def recommander_contenu(idPdt_cible, N, Produits, matrice, nom_mesure):
    index_cible  = idPdt_cible - 1
    scores       = list(enumerate(matrice[index_cible]))
    scores_tries = sorted(scores, key=lambda x: x[1], reverse=True)
    top_n        = [(idx, s) for idx, s in scores_tries if idx != index_cible][:N]
    resultats = []
    for idx, score in top_n:
        nom = Produits[idx][1]
        if score > 0.5:
            explication = f"Très similaire au produit sélectionné ({nom_mesure})"
        elif score > 0.2:
            explication = f"Recommandé car similaire à vos préférences ({nom_mesure})"
        else:
            explication = f"Légèrement similaire au produit sélectionné ({nom_mesure})"
        resultats.append({"produit": nom, "score": round(score, 4), "explication": explication})
    return resultats

def recommander_collaboratif(id_user_cible, N, matrice_notes, Produits):
    idx_cible = id_user_cible - 1
    vecteur   = matrice_notes[idx_cible]
    similarites = []
    for i in range(matrice_notes.shape[0]):
        if i != idx_cible:
            n1  = np.linalg.norm(vecteur)
            n2  = np.linalg.norm(matrice_notes[i])
            sim = 0.0 if n1==0 or n2==0 else np.dot(vecteur, matrice_notes[i])/(n1*n2)
            similarites.append((i, sim))
    similarites = sorted(similarites, key=lambda x: x[1], reverse=True)
    top_users   = similarites[:3]
    deja_notes  = set(j for j in range(matrice_notes.shape[1]) if vecteur[j] > 0)
    scores_produits = {}
    for idx_user, sim in top_users:
        for j in range(matrice_notes.shape[1]):
            note = matrice_notes[idx_user][j]
            if j not in deja_notes and note > 0:
                if j not in scores_produits:
                    scores_produits[j] = 0
                scores_produits[j] += sim * note
    recommandations = sorted(scores_produits.items(), key=lambda x: x[1], reverse=True)[:N]
    resultats = []
    for idx_produit, score in recommandations:
        nom = Produits[idx_produit][1]
        if score > 3:
            explication = "Très apprécié par des utilisateurs similaires"
        else:
            explication = "Recommandé car des utilisateurs similaires ont aimé cet item"
        resultats.append({"produit": nom, "score": round(score, 4), "explication": explication})
    return resultats, top_users

def recommander_temps(id_user_cible, N, interactions, Produits):
    aujourd_hui = date.today()
    deja_notes  = set(row[1] for row in interactions if row[0] == id_user_cible)
    scores_temps = {}
    for id_user, id_produit, note, date_inter in interactions:
        if id_user != id_user_cible and id_produit not in deja_notes:
            nb_jours = (aujourd_hui - date_inter).days
            poids    = 1.0 / (nb_jours + 1)
            if id_produit not in scores_temps:
                scores_temps[id_produit] = 0
            scores_temps[id_produit] += note * poids
    recommandations = sorted(scores_temps.items(), key=lambda x: x[1], reverse=True)[:N]
    resultats = []
    for id_produit, score in recommandations:
        resultats.append({
            "produit":     Produits[id_produit - 1][1],
            "score":       round(score, 6),
            "explication": "Recommandé car récent ou tendance"
        })
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
            scores_finaux[j + 1] = {
                "contenu": matriceCosinus[idx_ref][j],
                "collab":  0.0,
                "temps":   0.0
            }
    idx_u   = id_user_cible - 1
    vecteur = matrice_notes[idx_u]
    sims    = []
    for i in range(matrice_notes.shape[0]):
        if i != idx_u:
            n1  = np.linalg.norm(vecteur)
            n2  = np.linalg.norm(matrice_notes[i])
            sim = 0.0 if n1==0 or n2==0 else np.dot(vecteur, matrice_notes[i])/(n1*n2)
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
    resultats_raw = []
    for id_p, s in scores_finaux.items():
        sf = w_contenu*s["contenu"] + w_collab*s["collab"] + w_temps*s["temps"]
        resultats_raw.append((id_p, sf))
    resultats_raw = sorted(resultats_raw, key=lambda x: x[1], reverse=True)
    resultats = []
    for id_p, score in resultats_raw[:N]:
        resultats.append({
            "produit":     Produits[id_p - 1][1],
            "score":       round(score, 4),
            "explication": "Combinaison : contenu + utilisateurs similaires + tendances récentes"
        })
    return resultats


st.set_page_config(
    page_title="Système de Recommandation",
    page_icon="🛍️",
    layout="wide"
)


st.markdown("""
<div class="header-banner">
    <h1>🛍️ Système de Recommandation</h1>
    
</div>
""", unsafe_allow_html=True)


Produits      = charger_produits()
utilisateurs  = charger_utilisateurs()
interactions  = charger_interactions()
matriceCosinus, matriceEuclidienne, matriceJaccard = construire_matrices(Produits)
matrice_notes = construire_matrice_notes(interactions)


with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    st.markdown("---")
    st.markdown("**🔬 Méthode**")
    methode = st.selectbox("", [
        "📄 Content-Based Filtering",
        "👥 Collaborative Filtering",
        "⏱️ Time-Aware",
        "🔀 Hybride",
        "📊 Évaluation Bonus"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**⚖️ Poids Hybride**")
    w_contenu = st.slider("📄 Contenu",      0.0, 1.0, 0.4, 0.1)
    w_collab  = st.slider("👥 Collaboratif", 0.0, 1.0, 0.4, 0.1)
    w_temps   = st.slider("⏱️ Temps",        0.0, 1.0, 0.2, 0.1)
    st.markdown("---")
    st.markdown("**🎯 Top-N recommandations**")
    N = st.slider("", 1, 10, 3, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em;'>
            ISG Tunis · 2026
        </div>
        <div style='font-size:0.7rem; color:#94a3b8; margin-top:0.3rem;'>
            2 LNBI · Projet Final
        </div>
    </div>
    """, unsafe_allow_html=True)


c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{len(Produits)}</div>
        <div class="stat-label">Produits</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{len(utilisateurs)}</div>
        <div class="stat-label">Utilisateurs</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">{len(interactions)}</div>
        <div class="stat-label">Interactions</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-value">4</div>
        <div class="stat-label">Méthodes</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


if methode == "📄 Content-Based Filtering":
    st.markdown('<span class="badge badge-content">Content-Based Filtering</span>', unsafe_allow_html=True)
    st.markdown("### 📄 Recommandation par contenu")
    

    
    noms_produits = [f"{p[0]} - {p[1]}" for p in Produits]
    choix = st.selectbox("🛒 Choisir un produit", noms_produits)
    idPdt = int(choix.split(" - ")[0])
    mesure = st.radio(
        "📐 Mesure de similarité",
        ["Cosinus", "Euclidienne", "Jaccard"],
        horizontal=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Lancer la recommandation"):
        matrice_choisie = matriceCosinus if mesure == "Cosinus" else (
            matriceEuclidienne if mesure == "Euclidienne" else matriceJaccard
        )
        resultats = recommander_contenu(idPdt, N, Produits, matrice_choisie, mesure)
        st.markdown(f"### 🏆 Top {N} — Mesure {mesure}")
        for i, r in enumerate(resultats, 1):
            pct = int(min(float(r['score']), 1.0) * 100)
            st.markdown(f"""
            <div class="result-card">
                <span class="result-rank">#{i}</span>
                <span class="result-name">{r['produit']}</span>
                <span class="result-score">{r['score']}</span>
                <div class="result-expl">💡 {r['explication']}</div>
                <div style="margin-top:0.6rem; background:#f1f5f9;
                            border-radius:10px; height:6px; overflow:hidden;">
                    <div style="width:{pct}%; height:100%;
                                background:linear-gradient(90deg,#7c3aed,#2563eb);
                                border-radius:10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


elif methode == "👥 Collaborative Filtering":
    st.markdown('<span class="badge badge-collab">Collaborative Filtering</span>', unsafe_allow_html=True)
    st.markdown("### 👥 Filtrage collaboratif")
    
    noms_users = [f"{u[0]} - {u[1]}" for u in utilisateurs]
    choix      = st.selectbox("👤 Choisir un utilisateur", noms_users)
    id_user    = int(choix.split(" - ")[0])
    nom_user   = choix.split(" - ")[1]
    st.markdown('</div>', unsafe_allow_html=True)

    if st.checkbox("📋 Afficher l'historique des notes"):
        st.markdown("**🧾 Historique des interactions :**")
        hist = [(row[1], row[2]) for row in interactions if row[0] == id_user]
        cols = st.columns(min(len(hist), 3))
        for k, (id_p, note) in enumerate(hist):
            with cols[k % 3]:
                etoiles = "⭐" * int(note)
                st.markdown(f"""
                <div class="result-card" style="text-align:center;">
                    <div style="font-weight:700;color:#1e293b;">{Produits[id_p-1][1]}</div>
                    <div style="font-size:1.1rem;margin-top:0.3rem;">{etoiles}</div>
                    <div style="color:#64748b;font-size:0.82rem;">{note}/5</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("---")

    if st.button("🔍 Lancer la recommandation"):
        resultats, top_users = recommander_collaboratif(id_user, N, matrice_notes, Produits)
        col_v, col_r = st.columns([1, 2])
        with col_v:
            st.markdown("**🤝 Voisins similaires :**")
            for idx, sim in top_users:
                st.markdown(f"""
                <div class="voisin-card">
                    👤 Utilisateur {idx+1}
                    <span style="float:right; color:#1d4ed8; font-weight:700;">
                        {sim:.4f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
        with col_r:
            st.markdown(f"**🏆 Top {N} pour {nom_user} :**")
            for i, r in enumerate(resultats, 1):
                pct = int(min(float(r['score']) / 5, 1.0) * 100)
                st.markdown(f"""
                <div class="result-card" style="border-left-color:#2563eb;">
                    <span class="result-rank" style="color:#2563eb;">#{i}</span>
                    <span class="result-name">{r['produit']}</span>
                    <span class="result-score" style="background:linear-gradient(90deg,#2563eb,#059669);">
                        {r['score']}
                    </span>
                    <div class="result-expl">💡 {r['explication']}</div>
                    <div style="margin-top:0.6rem; background:#f1f5f9;
                                border-radius:10px; height:6px; overflow:hidden;">
                        <div style="width:{pct}%; height:100%;
                                    background:linear-gradient(90deg,#2563eb,#059669);
                                    border-radius:10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


elif methode == "⏱️ Time-Aware":
    st.markdown('<span class="badge badge-time">Time-Aware Recommendation</span>', unsafe_allow_html=True)
    st.markdown("### ⏱️ Recommandation temporelle")
    
    noms_users = [f"{u[0]} - {u[1]}" for u in utilisateurs]
    choix      = st.selectbox("👤 Choisir un utilisateur", noms_users)
    id_user    = int(choix.split(" - ")[0])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#f0fdf4; border:1px solid #86efac;
                border-radius:12px; padding:0.8rem 1.2rem; margin-bottom:1rem;
                font-size:0.88rem; color:#166534;">
        ⚡ Les interactions les plus récentes reçoivent un poids plus élevé.
        Les tendances actuelles sont ainsi mieux représentées.
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔍 Lancer la recommandation"):
        resultats = recommander_temps(id_user, N, interactions, Produits)
        st.markdown(f"### 🏆 Top {N} recommandations récentes")
        for i, r in enumerate(resultats, 1):
            st.markdown(f"""
            <div class="result-card" style="border-left-color:#059669;">
                <span class="result-rank" style="color:#059669;">#{i}</span>
                <span class="result-name">{r['produit']}</span>
                <span class="result-score" style="background:linear-gradient(90deg,#059669,#2563eb);">
                    {r['score']}
                </span>
                <div class="result-expl">💡 {r['explication']}</div>
            </div>
            """, unsafe_allow_html=True)


elif methode == "🔀 Hybride":
    st.markdown('<span class="badge badge-hybrid">Modèle Hybride</span>', unsafe_allow_html=True)
    st.markdown("### 🔀 Recommandation hybride")

    st.markdown(f"""
    <div style="display:flex; gap:1rem; margin-bottom:1rem; flex-wrap:wrap;">
        <div style="flex:1; background:#ede9fe; border:1.5px solid #c4b5fd;
                    border-radius:12px; padding:0.8rem; text-align:center;">
            <div style="font-size:1.4rem; font-weight:800; color:#6d28d9;">{int(w_contenu*100)}%</div>
            <div style="font-size:0.78rem; color:#7c3aed; text-transform:uppercase; font-weight:600;">Contenu</div>
        </div>
        <div style="flex:1; background:#dbeafe; border:1.5px solid #93c5fd;
                    border-radius:12px; padding:0.8rem; text-align:center;">
            <div style="font-size:1.4rem; font-weight:800; color:#1d4ed8;">{int(w_collab*100)}%</div>
            <div style="font-size:0.78rem; color:#2563eb; text-transform:uppercase; font-weight:600;">Collaboratif</div>
        </div>
        <div style="flex:1; background:#d1fae5; border:1.5px solid #6ee7b7;
                    border-radius:12px; padding:0.8rem; text-align:center;">
            <div style="font-size:1.4rem; font-weight:800; color:#065f46;">{int(w_temps*100)}%</div>
            <div style="font-size:0.78rem; color:#059669; text-transform:uppercase; font-weight:600;">Temps</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

   
    col1, col2 = st.columns(2)
    with col1:
        noms_users = [f"{u[0]} - {u[1]}" for u in utilisateurs]
        choix_user = st.selectbox("👤 Utilisateur", noms_users)
        id_user    = int(choix_user.split(" - ")[0])
    with col2:
        noms_prods = [f"{p[0]} - {p[1]}" for p in Produits]
        choix_prod = st.selectbox("🛒 Produit de référence", noms_prods)
        idPdt      = int(choix_prod.split(" - ")[0])
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Lancer la recommandation"):
        resultats = recommander_hybride(
            id_user, idPdt, N,
            matriceCosinus, matrice_notes,
            interactions, Produits,
            w_contenu, w_collab, w_temps
        )
        st.markdown(f"### 🏆 Top {N} recommandations hybrides")
        for i, r in enumerate(resultats, 1):
            pct = int(min(float(r['score']), 1.0) * 100)
            st.markdown(f"""
            <div class="result-card" style="border-left-color:#d97706;">
                <span class="result-rank" style="color:#d97706;">#{i}</span>
                <span class="result-name">{r['produit']}</span>
                <span class="result-score" style="background:linear-gradient(90deg,#d97706,#dc2626);">
                    {r['score']}
                </span>
                <div class="result-expl">💡 {r['explication']}</div>
                <div style="margin-top:0.6rem; background:#f1f5f9;
                            border-radius:10px; height:6px; overflow:hidden;">
                    <div style="width:{pct}%; height:100%;
                                background:linear-gradient(90deg,#d97706,#dc2626);
                                border-radius:10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


elif methode == "📊 Évaluation Bonus":
    st.markdown('<span class="badge badge-eval">Évaluation · Bonus</span>', unsafe_allow_html=True)
    st.markdown("### 📊 Évaluation du système")
   
    col1, col2 = st.columns(2)
    with col1:
        K     = st.slider("🎯 Valeur de K", 1, 5, 3)
    with col2:
        seuil = st.slider("⭐ Seuil de pertinence (note ≥)", 1, 5, 3)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Lancer l'évaluation"):
        train = []
        test  = []
        users = sorted(set(row[0] for row in interactions))

        for id_user in users:
            inter_user = sorted(
                [row for row in interactions if row[0] == id_user],
                key=lambda x: x[3]
            )
            split = max(1, int(len(inter_user) * 0.8))
            train.extend(inter_user[:split])
            test.extend(inter_user[split:])

        nb_users_global = max(row[0] for row in interactions)
        nb_prods_global = max(row[1] for row in interactions)
        matrice_train   = np.zeros((nb_users_global, nb_prods_global))
        for id_user, id_produit, note, _ in train:
            matrice_train[id_user - 1][id_produit - 1] = note

        precision_list  = []
        recall_list     = []
        resultats_users = []

        for id_user in users:
            pertinents = set(
                row[1] for row in test
                if row[0] == id_user and row[2] >= seuil
            )
            if not pertinents:
                continue
            idx     = id_user - 1
            vecteur = matrice_train[idx]
            if np.sum(vecteur) == 0:
                continue
            similarites = []
            for i in range(matrice_train.shape[0]):
                if i != idx:
                    n1  = np.linalg.norm(vecteur)
                    n2  = np.linalg.norm(matrice_train[i])
                    sim = 0.0 if n1==0 or n2==0 else np.dot(vecteur, matrice_train[i])/(n1*n2)
                    similarites.append((i, sim))
            similarites = sorted(similarites, key=lambda x: x[1], reverse=True)
            top_u = similarites[:3]
            deja_notes = set(j for j in range(matrice_train.shape[1]) if vecteur[j] > 0)
            scores = {}
            for idx_u, sim in top_u:
                for j in range(matrice_train.shape[1]):
                    note = matrice_train[idx_u][j]
                    if j not in deja_notes and note > 0:
                        if j not in scores:
                            scores[j] = 0
                        scores[j] += sim * note
            top         = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:K]
            recommandes = set(id_p + 1 for id_p, _ in top)
            vp          = len(recommandes & pertinents)
            precision   = vp / K if K > 0 else 0
            recall      = vp / len(pertinents) if pertinents else 0
            precision_list.append(precision)
            recall_list.append(recall)
            resultats_users.append({
                "id":          id_user,
                "pertinents":  pertinents,
                "recommandes": recommandes,
                "precision":   round(precision, 2),
                "recall":      round(recall, 2)
            })

        if not precision_list:
            st.warning("⚠️ Aucun utilisateur évaluable. Essaie de baisser le seuil.")
        else:
            p_moy = np.mean(precision_list)
            r_moy = np.mean(recall_list)
            f1    = 2 * p_moy * r_moy / (p_moy + r_moy + 0.0001)

            st.markdown("### 📈 Résultats globaux")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"Precision@{K}", f"{p_moy:.2%}")
            m2.metric(f"Recall@{K}",    f"{r_moy:.2%}")
            m3.metric("F1-Score",       f"{f1:.2%}")
            m4.metric("Utilisateurs évalués", f"{len(resultats_users)}/{len(users)}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🔍 Détail par utilisateur")
            for r in resultats_users:
                couleur_p = "#16a34a" if r['precision'] > 0 else "#dc2626"
                couleur_r = "#16a34a" if r['recall'] > 0 else "#dc2626"
                with st.expander(f"👤 Utilisateur {r['id']} — P@{K} = {r['precision']} | R@{K} = {r['recall']}"):
                    st.markdown(f"""
                    <div style="display:flex; gap:1rem; margin-bottom:0.8rem; flex-wrap:wrap;">
                        <div style="flex:1; background:#f8fafc; border:1.5px solid #e2e8f0;
                                    border-radius:10px; padding:0.8rem; text-align:center;">
                            <div style="font-size:1.4rem; font-weight:800; color:{couleur_p};">
                                {r['precision']}
                            </div>
                            <div style="font-size:0.75rem; color:#64748b;">Precision@{K}</div>
                        </div>
                        <div style="flex:1; background:#f8fafc; border:1.5px solid #e2e8f0;
                                    border-radius:10px; padding:0.8rem; text-align:center;">
                            <div style="font-size:1.4rem; font-weight:800; color:{couleur_r};">
                                {r['recall']}
                            </div>
                            <div style="font-size:0.75rem; color:#64748b;">Recall@{K}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write(f"**Produits pertinents :** {r['pertinents']}")
                    st.write(f"**Produits recommandés :** {r['recommandes']}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 💡 Interprétation")
            st.markdown(f"""
            <div style="background:#eff6ff; border:1.5px solid #bfdbfe;
                        border-radius:14px; padding:1.2rem 1.5rem; line-height:1.8;">
                <b style="color:#6d28d9;">Precision@{K} = {p_moy:.2%}</b> →
                Sur {K} recommandations, {p_moy:.2%} sont pertinentes.<br>
                <b style="color:#1d4ed8;">Recall@{K} = {r_moy:.2%}</b> →
                Le système retrouve {r_moy:.2%} des produits pertinents.<br>
                <b style="color:#065f46;">F1-Score = {f1:.2%}</b> →
                Score global combinant Precision et Recall.<br><br>
                <span style="color:#64748b; font-size:0.88rem;">
                ⚠️ Les scores sont faibles car nous avons seulement 50 interactions.
                Avec plus de données, les résultats seraient nettement meilleurs.
                </span>
            </div>
            """, unsafe_allow_html=True)