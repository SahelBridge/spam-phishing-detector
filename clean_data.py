import os
import pandas as pd

DATA_DIR = "data"
dataframes = []


def clean_and_normalize(df, filename):
    cols = {c.lower(): c for c in df.columns}

    # 1. Détection du texte (sujet + corps si dispo, sinon texte simple)
    text_series = None
    if "body" in cols and "subject" in cols:
        text_series = (
            df[cols["subject"]].fillna("").astype(str)
            + " "
            + df[cols["body"]].fillna("").astype(str)
        )
    elif "body" in cols:
        text_series = df[cols["body"]].astype(str)
    elif "text" in cols:
        text_series = df[cols["text"]].astype(str)
    elif "message" in cols:
        text_series = df[cols["message"]].astype(str)
    elif "v2" in cols:
        text_series = df[cols["v2"]].astype(str)

    # 2. Détection du label
    label_series = None
    if "label" in cols:
        label_series = df[cols["label"]]
    elif "v1" in cols:
        label_series = df[cols["v1"]]
    elif "category" in cols:
        label_series = df[cols["category"]]
    elif "email type" in cols:
        label_series = df[cols["email type"]]
    elif "spam" in cols:
        label_series = df[cols["spam"]]

    if text_series is None or label_series is None:
        print(f"⚠️ Ignoré (structure non reconnue) : {filename}")
        return None

    res = pd.DataFrame({"text": text_series, "label": label_series})

    # 3. Standardisation du label en 0 et 1
    mapping = {
        "spam": 1,
        "phishing": 1,
        "phishing email": 1,
        "fraud": 1,
        "smishing": 1,
        "1": 1,
        1: 1,
        1.0: 1,
        "ham": 0,
        "safe email": 0,
        "legitimate": 0,
        "normal": 0,
        "0": 0,
        0: 0,
        0.0: 0,
    }

    if res["label"].dtype == object:
        res["label"] = (
            res["label"].astype(str).str.strip().str.lower().map(mapping)
        )
    else:
        res["label"] = res["label"].map(mapping)

    res = res.dropna(subset=["text", "label"])
    res["label"] = res["label"].astype(int)
    return res


# Parcours de tous les fichiers du dossier data/
print("🔄 Traitement et fusion des datasets...")
for file in sorted(os.listdir(DATA_DIR)):
    if file.endswith(".csv"):
        path = os.path.join(DATA_DIR, file)
        try:
            # Essai de lecture avec différents encodages courants
            try:
                df = pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="latin-1")

            cleaned = clean_and_normalize(df, file)
            if cleaned is not None and not cleaned.empty:
                dataframes.append(cleaned)
                print(f"✅ Ajouté : {file} ({len(cleaned)} lignes)")
        except Exception as e:
            print(f"❌ Erreur sur {file} : {e}")

if dataframes:
    final_df = pd.concat(dataframes, ignore_index=True)
    # Nettoyage global : suppression des textes vides et des doublons
    final_df = final_df.drop_duplicates(subset=["text"])
    final_df = final_df[final_df["text"].str.strip().str.len() > 5]

    final_df.to_csv("dataset_clean.csv.gz", index=False, compression="gzip")
    print("\n🎉 FUSION RÉUSSIE !")
    print(f"📊 Total : {len(final_df)} exemples uniques dans dataset_clean.csv")
    print(f"📈 Répartition :\n{final_df['label'].value_counts()}")
    # Au lieu de: final_df.to_csv("dataset_clean.csv", index=False)
