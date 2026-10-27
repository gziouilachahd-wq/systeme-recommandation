# 🛍️ Système de Recommandation en Python

Projet final **2 LNBI – ISG Tunis** (Moteur de recommandation en Python).

Application de recommandation de produits combinant plusieurs approches, avec une explication pour chaque recommandation et une interface web **Streamlit**.

## 🎯 Fonctionnalités

* **Content-Based Filtering** : similarité entre produits à partir de leurs descriptions (NLP : tokenisation, stemming français, stop-words, matrice binaire) avec 3 mesures : Cosinus, Euclidienne, Jaccard.
* **Collaborative Filtering** : recommandations basées sur les utilisateurs aux goûts similaires (similarité cosinus sur la matrice des notes).
* **Time-Aware** : les interactions récentes ont un poids plus élevé.
* **Modèle hybride** : moyenne pondérée des 3 méthodes (poids réglables).
* **Évaluation (bonus)** : Precision@K, Recall@K, F1-score.
* **Explications** simples pour chaque recommandation.

\## 📸 Aperçu



\### Content-Based Filtering

!\[Content-Based](images/content.png)



\### Collaborative Filtering

!\[Collaborative Filtering](images/collab.png)



\### Time-Aware

!\[Time-Aware](images/temps.png)



\### Modèle hybride

!\[Hybride](images/hybride.png)



\### Évaluation (Precision@K, Recall@K, F1-score)

!\[Évaluation](images/evaluation.png)

## 🧰 Technologies

Python · PostgreSQL · psycopg2 · NLTK · NumPy · Streamlit

## 📁 Structure

|Fichier|Rôle|
|-|-|
|`recommandation.py`|Script principal : création/remplissage de la base, NLP, matrices, méthodes de recommandation (affichage console)|
|`app.py`|Interface Streamlit|
|`requirements.txt`|Dépendances Python|

## ⚙️ Installation

1. Installer PostgreSQL et créer une base nommée `recommandation`.
2. Installer les dépendances :

```bash
   pip install -r requirements.txt
   ```

3. Adapter les identifiants de connexion (`DB\_CONFIG`) dans les fichiers Python à votre configuration locale (utilisateur, mot de passe). Ne publiez jamais votre vrai mot de passe.

## ▶️ Exécution

1. Créer les tables et insérer les données (20 produits, 10 utilisateurs, 50 interactions) :

```bash
   python recommandation.py
   ```

2. Lancer l'interface web :

```bash
   streamlit run app.py
   ```

## 📊 Exemple d'explications

* « Très similaire au produit sélectionné »
* « Recommandé car des utilisateurs similaires ont aimé cet item »
* « Recommandé car récent ou tendance »

## 👤 Auteur

Chahed – Étudiant en Business Intelligence, ISG Tunis

