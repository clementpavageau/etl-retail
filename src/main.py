import pandas as pd
import numpy as np
import logging

# CONFIG LOG
logging.basicConfig(
    filename="etl.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# EXTRACT
# =========================
try:
    df = pd.read_csv("jeux_de_données.csv")
    logging.info("Extraction réussie")
except Exception as e:
    logging.error(f"Erreur extraction : {e}")
    raise

print("=== AVANT TRAITEMENT ===")
print("Valeurs manquantes :")
print(df.isnull().sum())
print("Doublons :", df.duplicated().sum())


# =========================
# TRANSFORM
# =========================

# 1. Gestion valeurs manquantes

# Nom_produit → remplacer par "Inconnu"
df["Nom_produit"] = df["Nom_produit"].fillna("Inconnu")

# Quantité → médiane
df["Quantite_vendue"] = df["Quantite_vendue"].fillna(df["Quantite_vendue"].median())

# Prix → médiane
df["Prix_unitaire"] = df["Prix_unitaire"].fillna(df["Prix_unitaire"].median())

logging.info("Valeurs manquantes traitées")

# 2. Suppression quantités nulles ou négatives
df = df[df["Quantite_vendue"] > 0]

# 3. Suppression doublons
df = df.drop_duplicates()

logging.info("Doublons supprimés")

# 4. Transformation 1 : Ajout Chiffre d'affaires
df["Chiffre_affaires"] = df["Quantite_vendue"] * df["Prix_unitaire"]

# 5. Transformation 2 : Normalisation du prix
df["Prix_normalise"] = (
    (df["Prix_unitaire"] - df["Prix_unitaire"].min()) /
    (df["Prix_unitaire"].max() - df["Prix_unitaire"].min())
)

logging.info("Transformations ajoutées")


# =========================
# VALIDATION
# =========================

print("\n=== APRES TRAITEMENT ===")
print("Valeurs manquantes :")
print(df.isnull().sum())
print("Doublons :", df.duplicated().sum())
print("Nombre lignes final :", len(df))

assert df.isnull().sum().sum() == 0

# =========================
# LOAD
# =========================

df.to_csv("ventes_clean.csv", index=False)
logging.info("Chargement terminé")
