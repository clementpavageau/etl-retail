import pandas as pd
import logging
import os

# =============================
# CONFIGURATION LOG
# =============================
logging.basicConfig(
    filename="etl.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def extract(path):
    try:
        df = pd.read_csv(path)
        logging.info("Extraction réussie")
        return df
    except Exception as e:
        logging.error(f"Erreur extraction : {e}")
        raise

def transform(df):
    # --- Valeurs manquantes ---
    df["Nom_produit"] = df["Nom_produit"].fillna("Inconnu")
    df["Quantite_vendue"] = df["Quantite_vendue"].fillna(df["Quantite_vendue"].median())
    df["Prix_unitaire"] = df["Prix_unitaire"].fillna(df["Prix_unitaire"].median())

    # --- Suppression quantités invalides ---
    df = df[df["Quantite_vendue"] > 0]

    # --- Suppression doublons ---
    df = df.drop_duplicates()

    # --- Transformation 1 : chiffre d'affaires ---
    df["Chiffre_affaires"] = df["Quantite_vendue"] * df["Prix_unitaire"]

    # --- Transformation 2 : normalisation prix ---
    df["Prix_normalise"] = (
        (df["Prix_unitaire"] - df["Prix_unitaire"].min()) /
        (df["Prix_unitaire"].max() - df["Prix_unitaire"].min())
    )

    logging.info("Transformations appliquées")
    return df

def load(df, output_path):
    df.to_csv(output_path, index=False)
    logging.info("Chargement terminé")

def validate(df):
    print("Valeurs manquantes :", df.isnull().sum().sum())
    print("Doublons :", df.duplicated().sum())
    print("Nombre de lignes :", len(df))

def main():
    input_path = "../data/jeux_de_données.csv"
    output_path = "../data/ventes_clean.csv"

    df = extract(input_path)

    print("AVANT TRAITEMENT")
    validate(df)

    df = transform(df)

    print("\nAPRES TRAITEMENT")
    validate(df)

    assert df.isnull().sum().sum() == 0

    load(df, output_path)

if __name__ == "__main__":
    main()