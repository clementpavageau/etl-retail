"""
ETL Pipeline - Nettoyage et transformation de données
Jeu de données : ventes de produits (ID, Nom, Quantité, Prix, Date)
"""

import pandas as pd
import numpy as np
import logging
import unittest
from io import StringIO

# ─────────────────────────────────────────────
# 1. CONFIGURATION DU LOGGER
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("etl_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 2. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Charge le fichier CSV et parse les dates."""
    try:
        df = pd.read_csv(filepath, parse_dates=["Date_vente"])
        logger.info(f"Fichier chargé : {filepath} ({len(df)} lignes, {len(df.columns)} colonnes)")
        return df
    except FileNotFoundError:
        logger.error(f"Fichier introuvable : {filepath}")
        raise
    except Exception as e:
        logger.error(f"Erreur lors du chargement : {e}")
        raise


# ─────────────────────────────────────────────
# 3. ANALYSE INITIALE DES DONNÉES
# ─────────────────────────────────────────────
def analyze_data(df: pd.DataFrame) -> dict:
    """Identifie les problèmes : valeurs manquantes, aberrantes, doublons."""
    try:
        logger.info("=" * 50)
        logger.info("ETAPE 1 : ANALYSE DES DONNEES")
        logger.info("=" * 50)

        report = {}

        # Dimensions
        report["shape"] = df.shape
        logger.info(f"Dimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes")

        # Valeurs manquantes
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        report["missing"] = missing[missing > 0].to_dict()
        report["missing_pct"] = missing_pct[missing_pct > 0].to_dict()
        logger.info(f"Valeurs manquantes :\n{missing[missing > 0]}")

        # Doublons
        dupes = df.duplicated(subset=["Nom_produit", "Quantite_vendue", "Prix_unitaire", "Date_vente"]).sum()
        report["duplicates"] = int(dupes)
        logger.info(f"Doublons detectes : {dupes}")

        # Valeurs aberrantes via IQR
        outlier_info = {}
        for col in ["Quantite_vendue", "Prix_unitaire"]:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = ((df[col] < lower) | (df[col] > upper)).sum()
            outlier_info[col] = {"count": int(n_out), "borne_basse": round(lower, 2), "borne_haute": round(upper, 2)}
        report["outliers"] = outlier_info
        logger.info(f"Valeurs aberrantes (IQR) : {outlier_info}")

        # Quantités à 0
        zero_qty = (df["Quantite_vendue"] == 0).sum()
        report["zero_quantity"] = int(zero_qty)
        logger.info(f"Quantites a 0 (suspects) : {zero_qty}")

        return report

    except Exception as e:
        logger.error(f"Erreur analyse : {e}")
        raise


# ─────────────────────────────────────────────
# 4. TRAITEMENT DES VALEURS MANQUANTES
# ─────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategie :
      - Nom_produit manquant  -> remplace par 'Produit_Inconnu'
      - Quantite_vendue nulle -> remplacee par la mediane
      - Prix_unitaire nul     -> remplace par la mediane
    """
    try:
        logger.info("=" * 50)
        logger.info("ETAPE 2 : TRAITEMENT DES VALEURS MANQUANTES")
        logger.info("=" * 50)
        df = df.copy()

        nb = df["Nom_produit"].isnull().sum()
        df["Nom_produit"] = df["Nom_produit"].fillna("Produit_Inconnu")
        logger.info(f"  'Nom_produit' : {nb} valeur(s) remplacee(s) par 'Produit_Inconnu'")

        mediane_qty = df["Quantite_vendue"].median()
        nb = df["Quantite_vendue"].isnull().sum()
        df["Quantite_vendue"] = df["Quantite_vendue"].fillna(mediane_qty)
        logger.info(f"  'Quantite_vendue' : {nb} valeur(s) remplacee(s) par la mediane ({mediane_qty})")

        mediane_prix = df["Prix_unitaire"].median()
        nb = df["Prix_unitaire"].isnull().sum()
        df["Prix_unitaire"] = df["Prix_unitaire"].fillna(mediane_prix)
        logger.info(f"  'Prix_unitaire' : {nb} valeur(s) remplacee(s) par la mediane ({mediane_prix})")

        logger.info(f"  Valeurs manquantes restantes : {df.isnull().sum().sum()}")
        return df

    except Exception as e:
        logger.error(f"Erreur valeurs manquantes : {e}")
        raise


# ─────────────────────────────────────────────
# 5. GESTION DES VALEURS ABERRANTES
# ─────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ecrete les valeurs hors bornes IQR.
    Supprime aussi les quantites <= 0 (ventes impossibles).
    """
    try:
        logger.info("=" * 50)
        logger.info("ETAPE 3 : GESTION DES VALEURS ABERRANTES")
        logger.info("=" * 50)
        df = df.copy()

        nb_zero = (df["Quantite_vendue"] <= 0).sum()
        df = df[df["Quantite_vendue"] > 0].copy()
        logger.info(f"  Lignes avec Quantite_vendue <= 0 supprimees : {nb_zero}")

        for col in ["Quantite_vendue", "Prix_unitaire"]:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            avant = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower=lower, upper=upper)
            logger.info(f"  '{col}' : {avant} valeur(s) ecretee(s) dans [{lower:.2f}, {upper:.2f}]")

        return df

    except Exception as e:
        logger.error(f"Erreur valeurs aberrantes : {e}")
        raise


# ─────────────────────────────────────────────
# 6. SUPPRESSION DES DOUBLONS
# ─────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons sur les colonnes metier."""
    try:
        logger.info("=" * 50)
        logger.info("ETAPE 4 : SUPPRESSION DES DOUBLONS")
        logger.info("=" * 50)
        initial = len(df)
        df = df.drop_duplicates(
            subset=["Nom_produit", "Quantite_vendue", "Prix_unitaire", "Date_vente"]
        ).copy()
        removed = initial - len(df)
        logger.info(f"  {removed} doublon(s) supprime(s). Lignes restantes : {len(df)}")
        return df

    except Exception as e:
        logger.error(f"Erreur suppression doublons : {e}")
        raise


# ─────────────────────────────────────────────
# 7. VALIDATION CROISEE
# ─────────────────────────────────────────────
def validate(df: pd.DataFrame, step: str) -> bool:
    """Verifie la qualite du DataFrame apres chaque etape."""
    try:
        logger.info(f"  [VALIDATION apres '{step}']")
        ok = True

        missing = df.isnull().sum().sum()
        if missing > 0:
            logger.warning(f"    ! {missing} valeur(s) manquante(s) encore presentes")
            ok = False
        else:
            logger.info("    OK - Aucune valeur manquante")

        neg_qty = (df["Quantite_vendue"] <= 0).sum()
        if neg_qty > 0:
            logger.warning(f"    ! {neg_qty} quantite(s) <= 0")
            ok = False
        else:
            logger.info("    OK - Toutes les quantites sont positives")

        neg_price = (df["Prix_unitaire"] <= 0).sum()
        if neg_price > 0:
            logger.warning(f"    ! {neg_price} prix <= 0")
            ok = False
        else:
            logger.info("    OK - Tous les prix sont positifs")

        return ok

    except Exception as e:
        logger.error(f"Erreur validation : {e}")
        raise


# ─────────────────────────────────────────────
# 8. TRANSFORMATIONS
# ─────────────────────────────────────────────
def transform(df: pd.DataFrame):
    """
    Transformations appliquees :
      T1 - Ajout colonne 'Chiffre_affaires' (prix x quantite)
      T2 - Normalisation Min-Max du Prix_unitaire -> 'Prix_normalise'
      T3 - Agregation mensuelle des ventes par produit
    """
    try:
        logger.info("=" * 50)
        logger.info("ETAPE 6 : TRANSFORMATIONS")
        logger.info("=" * 50)
        df = df.copy()

        # T1 : Chiffre d'affaires
        df["Chiffre_affaires"] = (df["Quantite_vendue"] * df["Prix_unitaire"]).round(2)
        logger.info("  T1 OK - Colonne 'Chiffre_affaires' ajoutee")

        # T2 : Normalisation Min-Max
        min_p, max_p = df["Prix_unitaire"].min(), df["Prix_unitaire"].max()
        df["Prix_normalise"] = ((df["Prix_unitaire"] - min_p) / (max_p - min_p)).round(4)
        logger.info(f"  T2 OK - Colonne 'Prix_normalise' ajoutee (min={min_p}, max={max_p})")

        # T3 : Agregation mensuelle
        df["Date_vente"] = pd.to_datetime(df["Date_vente"], errors="coerce")
        df["Mois"] = df["Date_vente"].dt.to_period("M").astype(str)
        agg = (
            df.groupby(["Mois", "Nom_produit"])
            .agg(
                Total_quantite=("Quantite_vendue", "sum"),
                CA_mensuel=("Chiffre_affaires", "sum"),
                Nb_ventes=("ID_produit", "count")
            )
            .reset_index()
        )
        logger.info(f"  T3 OK - Agregation mensuelle : {len(agg)} lignes")

        return df, agg

    except Exception as e:
        logger.error(f"Erreur transformation : {e}")
        raise


# ─────────────────────────────────────────────
# 9. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def run_pipeline(filepath: str):
    """Execute toutes les etapes du pipeline ETL."""
    logger.info("DEMARRAGE DU PIPELINE ETL")

    df = load_data(filepath)
    analyze_data(df)

    df = handle_missing_values(df)
    validate(df, "handle_missing_values")

    df = handle_outliers(df)
    validate(df, "handle_outliers")

    df = remove_duplicates(df)
    validate(df, "remove_duplicates")

    df_clean, df_agg = transform(df)
    validate(df_clean, "transform")

    df_clean.to_csv("data_nettoyee.csv", index=False)
    df_agg.to_csv("aggregation_mensuelle.csv", index=False)
    logger.info("Pipeline termine. Fichiers sauvegardes.")

    return df_clean, df_agg


# ─────────────────────────────────────────────
# 10. TESTS UNITAIRES
# ─────────────────────────────────────────────
class TestETLPipeline(unittest.TestCase):

    def setUp(self):
        csv_data = """ID_produit,Nom_produit,Quantite_vendue,Prix_unitaire,Date_vente
1,Chemise,10,25.0,2022-01-05
2,Pantalon,8,35.0,2022-01-06
3,Chaussures,,50.0,2022-01-07
4,,12,15.0,2022-01-08
5,Pantalon,8,35.0,2022-01-06
6,Chemise,10,25.0,2022-01-05
7,Robe,0,45.0,2022-01-09
8,Pull,20,,2022-01-10
"""
        self.df = pd.read_csv(StringIO(csv_data), parse_dates=["Date_vente"])

    def test_missing_values_resolved(self):
        df = handle_missing_values(self.df)
        self.assertEqual(df.isnull().sum().sum(), 0, "Des valeurs manquantes subsistent")

    def test_outliers_no_zero_quantity(self):
        df = handle_missing_values(self.df)
        df = handle_outliers(df)
        self.assertTrue((df["Quantite_vendue"] > 0).all(), "Des quantites <= 0 subsistent")

    def test_duplicates_removed(self):
        df = handle_missing_values(self.df)
        df = handle_outliers(df)
        df = remove_duplicates(df)
        dupes = df.duplicated(subset=["Nom_produit", "Quantite_vendue", "Prix_unitaire", "Date_vente"]).sum()
        self.assertEqual(dupes, 0, "Des doublons subsistent")

    def test_transform_columns_exist(self):
        df = handle_missing_values(self.df)
        df = handle_outliers(df)
        df, _ = transform(df)
        self.assertIn("Chiffre_affaires", df.columns)
        self.assertIn("Prix_normalise", df.columns)

    def test_prix_normalise_range(self):
        df = handle_missing_values(self.df)
        df = handle_outliers(df)
        df, _ = transform(df)
        self.assertTrue((df["Prix_normalise"] >= 0).all() and (df["Prix_normalise"] <= 1).all(),
                        "Prix_normalise hors de [0, 1]")

    def test_chiffre_affaires_positif(self):
        df = handle_missing_values(self.df)
        df = handle_outliers(df)
        df, _ = transform(df)
        self.assertTrue((df["Chiffre_affaires"] > 0).all(), "Chiffre d'affaires negatif detecte")


# ─────────────────────────────────────────────
# 11. POINT D'ENTREE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # --- Lancer le pipeline sur ton fichier ---
    df_clean, df_agg = run_pipeline("jeux_de_données.csv")

    print("\nApercu des donnees nettoyees :")
    print(df_clean.head(10).to_string(index=False))

    print("\nAgregation mensuelle :")
    print(df_agg.to_string(index=False))

    # --- Lancer les tests unitaires ---
    print("\nTests unitaires :")
    unittest.main(argv=[""], exit=False, verbosity=2)
