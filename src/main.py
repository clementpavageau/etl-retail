"""
ETL Pipeline enrichi - Sauvegarde & Historisation
Jeu de données : ventes de produits (ID, Nom, Quantité, Prix, Date)

Nouveautés :
  - Sauvegarde locale horodatée (raw / cleaned / transformed)
  - Historisation : chaque exécution crée un dossier daté, un manifeste JSON
    et un registre global d'historique (history_registry.csv)
  - Modules optionnels Google Cloud Storage et Amazon S3
"""

import os
import json
import shutil
import hashlib
import pandas as pd
import numpy as np
import logging
import unittest
from io import StringIO
from datetime import datetime
from pathlib import Path

# ════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ════════════════════════════════════════════════════════════

# Répertoire racine de stockage (modifiable)
STORAGE_ROOT = Path("etl_storage")

# Optionnel – Google Cloud Storage
GCS_ENABLED      = False
GCS_BUCKET_NAME  = "mon-bucket-gcs"          # à remplacer
GCS_PROJECT_ID   = "mon-projet-gcp"          # à remplacer

# Optionnel – Amazon S3
S3_ENABLED       = False
S3_BUCKET_NAME   = "mon-bucket-s3"           # à remplacer
S3_REGION        = "eu-west-1"               # à remplacer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("etl_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# 2. GESTIONNAIRE DE SAUVEGARDE & HISTORISATION
# ════════════════════════════════════════════════════════════

class StorageManager:
    """
    Gère la sauvegarde locale et l'historisation horodatée.
    Structure créée :
        etl_storage/
          history_registry.csv          ← registre global de toutes les exécutions
          runs/
            20240315_143022/            ← un dossier par exécution
              manifest.json             ← métadonnées de l'exécution
              raw_data.csv
              cleaned_data.csv
              transformed_data.csv
              aggregation_mensuelle.csv
    """

    REGISTRY_FILE = "history_registry.csv"

    def __init__(self, root: Path = STORAGE_ROOT):
        self.root      = Path(root)
        self.runs_dir  = self.root / "runs"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir   = self.runs_dir / self.timestamp
        self._init_dirs()

    # ── Initialisation ──────────────────────────────────────
    def _init_dirs(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[STORAGE] Dossier d'exécution : {self.run_dir}")

    # ── Calcul checksum ─────────────────────────────────────
    @staticmethod
    def _md5(df: pd.DataFrame) -> str:
        return hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()

    # ── Sauvegarde d'un DataFrame ───────────────────────────
    def save(self, df: pd.DataFrame, stage: str) -> Path:
        """
        Sauvegarde un DataFrame dans le dossier de l'exécution courante.
        stage : 'raw' | 'cleaned' | 'transformed' | 'aggregation'
        """
        filename = f"{stage}_data.csv" if stage != "aggregation" else "aggregation_mensuelle.csv"
        filepath = self.run_dir / filename
        df.to_csv(filepath, index=False)
        size_kb = filepath.stat().st_size / 1024
        logger.info(f"[STORAGE] Sauvegarde '{stage}' → {filepath} ({size_kb:.1f} Ko, {len(df)} lignes)")
        return filepath

    # ── Manifeste JSON ──────────────────────────────────────
    def write_manifest(self, meta: dict):
        """Écrit un fichier JSON décrivant l'exécution."""
        meta["timestamp"]  = self.timestamp
        meta["run_dir"]    = str(self.run_dir)
        manifest_path = self.run_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info(f"[STORAGE] Manifeste écrit : {manifest_path}")

    # ── Registre global d'historique ────────────────────────
    def register_run(self, meta: dict):
        """
        Ajoute une ligne dans le fichier CSV global d'historique.
        Crée le fichier s'il n'existe pas encore.
        """
        registry_path = self.root / self.REGISTRY_FILE
        record = {
            "timestamp":       self.timestamp,
            "run_dir":         str(self.run_dir),
            "lignes_brutes":   meta.get("lignes_brutes", ""),
            "lignes_nettoyees":meta.get("lignes_nettoyees", ""),
            "doublons_suppr":  meta.get("doublons_supprimes", ""),
            "checksum_clean":  meta.get("checksum_cleaned", ""),
            "statut":          meta.get("statut", "OK"),
        }
        new_row = pd.DataFrame([record])

        if registry_path.exists():
            existing = pd.read_csv(registry_path)
            registry = pd.concat([existing, new_row], ignore_index=True)
        else:
            registry = new_row

        registry.to_csv(registry_path, index=False)
        logger.info(f"[STORAGE] Registre mis à jour : {registry_path}")

    # ── Copie "latest" pour accès rapide ────────────────────
    def update_latest(self):
        """Copie le dossier courant vers etl_storage/latest/."""
        latest_dir = self.root / "latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        shutil.copytree(self.run_dir, latest_dir)
        logger.info(f"[STORAGE] Dossier 'latest' mis à jour → {latest_dir}")

    # ── Lister l'historique ──────────────────────────────────
    def list_history(self) -> pd.DataFrame:
        """Retourne le registre complet des exécutions passées."""
        registry_path = self.root / self.REGISTRY_FILE
        if not registry_path.exists():
            return pd.DataFrame()
        return pd.read_csv(registry_path)

    # ── Restaurer une version précédente ────────────────────
    def restore_version(self, timestamp: str, stage: str = "cleaned") -> pd.DataFrame:
        """
        Charge un DataFrame d'une exécution passée.
        timestamp : ex. '20240315_143022'
        stage     : 'raw' | 'cleaned' | 'transformed' | 'aggregation'
        """
        filename  = f"{stage}_data.csv" if stage != "aggregation" else "aggregation_mensuelle.csv"
        filepath  = self.runs_dir / timestamp / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Version introuvable : {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"[STORAGE] Version restaurée : {filepath} ({len(df)} lignes)")
        return df


# ════════════════════════════════════════════════════════════
# 3. MODULE GOOGLE CLOUD STORAGE (optionnel)
# ════════════════════════════════════════════════════════════

class GCSUploader:
    """
    Upload les fichiers ETL vers Google Cloud Storage.
    Nécessite : pip install google-cloud-storage
    Authentification : variable d'environnement GOOGLE_APPLICATION_CREDENTIALS
    """

    def __init__(self, bucket_name: str, project_id: str):
        try:
            from google.cloud import storage
            self.client = storage.Client(project=project_id)
            self.bucket = self.client.bucket(bucket_name)
            self.bucket_name = bucket_name
            logger.info(f"[GCS] Connecté au bucket : gs://{bucket_name}")
        except ImportError:
            raise ImportError("Installez google-cloud-storage : pip install google-cloud-storage")
        except Exception as e:
            raise ConnectionError(f"[GCS] Connexion impossible : {e}")

    def upload(self, local_path: Path, timestamp: str, stage: str):
        """Upload un fichier local vers GCS avec préfixe horodaté."""
        destination = f"etl_runs/{timestamp}/{local_path.name}"
        blob = self.bucket.blob(destination)
        blob.upload_from_filename(str(local_path))
        gcs_uri = f"gs://{self.bucket_name}/{destination}"
        logger.info(f"[GCS] Fichier uploadé : {gcs_uri}")
        return gcs_uri

    def upload_run(self, run_dir: Path, timestamp: str):
        """Upload tous les fichiers d'une exécution."""
        uris = []
        for f in run_dir.glob("*.csv"):
            uris.append(self.upload(f, timestamp, f.stem))
        for f in run_dir.glob("*.json"):
            uris.append(self.upload(f, timestamp, f.stem))
        logger.info(f"[GCS] {len(uris)} fichier(s) uploadé(s) pour l'exécution {timestamp}")
        return uris


# ════════════════════════════════════════════════════════════
# 4. MODULE AMAZON S3 (optionnel)
# ════════════════════════════════════════════════════════════

class S3Uploader:
    """
    Upload les fichiers ETL vers Amazon S3.
    Nécessite : pip install boto3
    Authentification : variables AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    ou profil AWS configuré (~/.aws/credentials)
    """

    def __init__(self, bucket_name: str, region: str):
        try:
            import boto3
            self.s3 = boto3.client("s3", region_name=region)
            self.bucket_name = bucket_name
            logger.info(f"[S3] Connecté au bucket : s3://{bucket_name} (région: {region})")
        except ImportError:
            raise ImportError("Installez boto3 : pip install boto3")
        except Exception as e:
            raise ConnectionError(f"[S3] Connexion impossible : {e}")

    def upload(self, local_path: Path, timestamp: str):
        """Upload un fichier local vers S3 avec préfixe horodaté."""
        key = f"etl_runs/{timestamp}/{local_path.name}"
        self.s3.upload_file(str(local_path), self.bucket_name, key)
        s3_uri = f"s3://{self.bucket_name}/{key}"
        logger.info(f"[S3] Fichier uploadé : {s3_uri}")
        return s3_uri

    def upload_run(self, run_dir: Path, timestamp: str):
        """Upload tous les fichiers d'une exécution."""
        uris = []
        for f in run_dir.glob("*.csv"):
            uris.append(self.upload(f, timestamp))
        for f in run_dir.glob("*.json"):
            uris.append(self.upload(f, timestamp))
        logger.info(f"[S3] {len(uris)} fichier(s) uploadé(s) pour l'exécution {timestamp}")
        return uris


# ════════════════════════════════════════════════════════════
# 5. ÉTAPES ETL (inchangées, enrichies des appels storage)
# ════════════════════════════════════════════════════════════

def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, parse_dates=["Date_vente"])
        logger.info(f"Fichier chargé : {filepath} ({len(df)} lignes, {len(df.columns)} colonnes)")
        return df
    except FileNotFoundError:
        logger.error(f"Fichier introuvable : {filepath}")
        raise
    except Exception as e:
        logger.error(f"Erreur chargement : {e}")
        raise


def analyze_data(df: pd.DataFrame) -> dict:
    try:
        logger.info("=" * 50)
        logger.info("ETAPE 1 : ANALYSE DES DONNEES")
        logger.info("=" * 50)
        report = {}
        report["shape"] = df.shape
        logger.info(f"Dimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes")

        missing = df.isnull().sum()
        report["missing"] = missing[missing > 0].to_dict()
        logger.info(f"Valeurs manquantes :\n{missing[missing > 0]}")

        dupes = df.duplicated(subset=["Nom_produit", "Quantite_vendue", "Prix_unitaire", "Date_vente"]).sum()
        report["duplicates"] = int(dupes)
        logger.info(f"Doublons detectes : {dupes}")

        outlier_info = {}
        for col in ["Quantite_vendue", "Prix_unitaire"]:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = ((df[col] < lower) | (df[col] > upper)).sum()
            outlier_info[col] = {"count": int(n_out), "borne_basse": round(lower, 2), "borne_haute": round(upper, 2)}
        report["outliers"] = outlier_info
        logger.info(f"Valeurs aberrantes (IQR) : {outlier_info}")

        report["zero_quantity"] = int((df["Quantite_vendue"] == 0).sum())
        return report
    except Exception as e:
        logger.error(f"Erreur analyse : {e}")
        raise


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("=" * 50)
        logger.info("ETAPE 2 : TRAITEMENT DES VALEURS MANQUANTES")
        logger.info("=" * 50)
        df = df.copy()
        nb = df["Nom_produit"].isnull().sum()
        df["Nom_produit"] = df["Nom_produit"].fillna("Produit_Inconnu")
        logger.info(f"  'Nom_produit' : {nb} remplacee(s) par 'Produit_Inconnu'")

        m = df["Quantite_vendue"].median()
        nb = df["Quantite_vendue"].isnull().sum()
        df["Quantite_vendue"] = df["Quantite_vendue"].fillna(m)
        logger.info(f"  'Quantite_vendue' : {nb} remplacee(s) par mediane ({m})")

        m = df["Prix_unitaire"].median()
        nb = df["Prix_unitaire"].isnull().sum()
        df["Prix_unitaire"] = df["Prix_unitaire"].fillna(m)
        logger.info(f"  'Prix_unitaire' : {nb} remplacee(s) par mediane ({m})")

        logger.info(f"  Valeurs manquantes restantes : {df.isnull().sum().sum()}")
        return df
    except Exception as e:
        logger.error(f"Erreur valeurs manquantes : {e}")
        raise


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("=" * 50)
        logger.info("ETAPE 3 : GESTION DES VALEURS ABERRANTES")
        logger.info("=" * 50)
        df = df.copy()
        nb = (df["Quantite_vendue"] <= 0).sum()
        df = df[df["Quantite_vendue"] > 0].copy()
        logger.info(f"  Quantite_vendue <= 0 supprimees : {nb}")

        for col in ["Quantite_vendue", "Prix_unitaire"]:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            avant = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower=lower, upper=upper)
            logger.info(f"  '{col}' : {avant} valeur(s) ecretee(s) dans [{lower:.2f}, {upper:.2f}]")
        return df
    except Exception as e:
        logger.error(f"Erreur aberrantes : {e}")
        raise


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("=" * 50)
        logger.info("ETAPE 4 : SUPPRESSION DES DOUBLONS")
        logger.info("=" * 50)
        initial = len(df)
        df = df.drop_duplicates(
            subset=["Nom_produit", "Quantite_vendue", "Prix_unitaire", "Date_vente"]
        ).copy()
        logger.info(f"  {initial - len(df)} doublon(s) supprime(s). Lignes restantes : {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Erreur doublons : {e}")
        raise


def validate(df: pd.DataFrame, step: str) -> bool:
    try:
        logger.info(f"  [VALIDATION apres '{step}']")
        ok = True
        m = df.isnull().sum().sum()
        if m > 0:
            logger.warning(f"    ! {m} valeur(s) manquante(s)")
            ok = False
        else:
            logger.info("    OK - Aucune valeur manquante")

        if (df["Quantite_vendue"] <= 0).sum() > 0:
            logger.warning(f"    ! Quantites <= 0 presentes")
            ok = False
        else:
            logger.info("    OK - Quantites positives")

        if (df["Prix_unitaire"] <= 0).sum() > 0:
            logger.warning(f"    ! Prix <= 0 presents")
            ok = False
        else:
            logger.info("    OK - Prix positifs")
        return ok
    except Exception as e:
        logger.error(f"Erreur validation : {e}")
        raise


def transform(df: pd.DataFrame):
    try:
        logger.info("=" * 50)
        logger.info("ETAPE 5 : TRANSFORMATIONS")
        logger.info("=" * 50)
        df = df.copy()

        df["Chiffre_affaires"] = (df["Quantite_vendue"] * df["Prix_unitaire"]).round(2)
        logger.info("  T1 OK - 'Chiffre_affaires' ajoutee")

        min_p, max_p = df["Prix_unitaire"].min(), df["Prix_unitaire"].max()
        df["Prix_normalise"] = ((df["Prix_unitaire"] - min_p) / (max_p - min_p)).round(4)
        logger.info(f"  T2 OK - 'Prix_normalise' ajoutee (min={min_p}, max={max_p})")

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


# ════════════════════════════════════════════════════════════
# 6. PIPELINE PRINCIPAL (avec sauvegarde & historisation)
# ════════════════════════════════════════════════════════════

def run_pipeline(filepath: str):
    """
    Pipeline ETL complet avec :
      - sauvegarde locale à chaque étape (raw / cleaned / transformed)
      - manifeste JSON par exécution
      - registre global d'historique
      - upload cloud optionnel (GCS / S3)
    """
    storage = StorageManager()
    meta = {"statut": "OK", "source": filepath}

    try:
        logger.info("DEMARRAGE DU PIPELINE ETL")

        # ── Extract ──────────────────────────────────────────
        df_raw = load_data(filepath)
        meta["lignes_brutes"] = len(df_raw)
        storage.save(df_raw, "raw")          # sauvegarde données brutes

        report = analyze_data(df_raw)
        meta["analyse"] = report

        # ── Clean ────────────────────────────────────────────
        df = handle_missing_values(df_raw)
        validate(df, "handle_missing_values")

        df = handle_outliers(df)
        validate(df, "handle_outliers")

        df = remove_duplicates(df)
        validate(df, "remove_duplicates")

        meta["lignes_nettoyees"]  = len(df)
        meta["doublons_supprimes"] = meta["lignes_brutes"] - len(df)
        meta["checksum_cleaned"]  = StorageManager._md5(df)
        storage.save(df, "cleaned")          # sauvegarde données nettoyées

        # ── Transform ────────────────────────────────────────
        df_clean, df_agg = transform(df)
        validate(df_clean, "transform")

        meta["lignes_transformees"] = len(df_clean)
        storage.save(df_clean, "transformed")       # sauvegarde données transformées
        storage.save(df_agg,   "aggregation")       # sauvegarde agrégation

        # ── Historisation ────────────────────────────────────
        storage.write_manifest(meta)
        storage.register_run(meta)
        storage.update_latest()

        # ── Upload cloud (optionnel) ─────────────────────────
        if GCS_ENABLED:
            try:
                gcs = GCSUploader(GCS_BUCKET_NAME, GCS_PROJECT_ID)
                gcs.upload_run(storage.run_dir, storage.timestamp)
            except Exception as e:
                logger.warning(f"[GCS] Upload ignoré : {e}")

        if S3_ENABLED:
            try:
                s3 = S3Uploader(S3_BUCKET_NAME, S3_REGION)
                s3.upload_run(storage.run_dir, storage.timestamp)
            except Exception as e:
                logger.warning(f"[S3] Upload ignoré : {e}")

        logger.info(f"Pipeline termine avec succes. Run ID : {storage.timestamp}")
        return df_clean, df_agg, storage

    except Exception as e:
        meta["statut"] = f"ERREUR : {e}"
        storage.write_manifest(meta)
        storage.register_run(meta)
        logger.error(f"Pipeline echoue : {e}")
        raise


# ════════════════════════════════════════════════════════════
# 7. TESTS UNITAIRES
# ════════════════════════════════════════════════════════════

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
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_outliers_no_zero_quantity(self):
        df = handle_missing_values(self.df)
        df = handle_outliers(df)
        self.assertTrue((df["Quantite_vendue"] > 0).all())

    def test_duplicates_removed(self):
        df = handle_missing_values(self.df)
        df = handle_outliers(df)
        df = remove_duplicates(df)
        dupes = df.duplicated(subset=["Nom_produit", "Quantite_vendue", "Prix_unitaire", "Date_vente"]).sum()
        self.assertEqual(dupes, 0)

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
        self.assertTrue((df["Prix_normalise"] >= 0).all() and (df["Prix_normalise"] <= 1).all())

    def test_chiffre_affaires_positif(self):
        df = handle_missing_values(self.df)
        df = handle_outliers(df)
        df, _ = transform(df)
        self.assertTrue((df["Chiffre_affaires"] > 0).all())

    def test_storage_creates_run_dir(self):
        """Vérifie que StorageManager crée bien un dossier d'exécution."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            sm = StorageManager(root=Path(tmp))
            self.assertTrue(sm.run_dir.exists())

    def test_storage_save_and_restore(self):
        """Vérifie la sauvegarde et la restauration d'un DataFrame."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            sm = StorageManager(root=Path(tmp))
            df = handle_missing_values(self.df)
            sm.save(df, "cleaned")
            restored = sm.restore_version(sm.timestamp, "cleaned")
            self.assertEqual(len(restored), len(df))

    def test_registry_created(self):
        """Vérifie que le registre d'historique est bien créé."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            sm = StorageManager(root=Path(tmp))
            sm.register_run({"lignes_brutes": 10, "lignes_nettoyees": 8,
                             "doublons_supprimes": 2, "checksum_cleaned": "abc", "statut": "OK"})
            registry = sm.list_history()
            self.assertEqual(len(registry), 1)
            self.assertEqual(registry.iloc[0]["statut"], "OK")


# ════════════════════════════════════════════════════════════
# 8. POINT D'ENTRÉE
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Lancer le pipeline ───────────────────────────────────
    df_clean, df_agg, storage = run_pipeline("jeux_de_données.csv")

    print("\nApercu des donnees nettoyees et transformees :")
    print(df_clean.head(10).to_string(index=False))

    print("\nAgregation mensuelle :")
    print(df_agg.to_string(index=False))

    print("\nHistorique des executions :")
    print(storage.list_history().to_string(index=False))

    # ── Lancer les tests unitaires ──────────────────────────
    print("\nTests unitaires :")
    unittest.main(argv=[""], exit=False, verbosity=2)
