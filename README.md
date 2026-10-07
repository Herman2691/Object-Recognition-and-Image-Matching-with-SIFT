# TP1 — Détection et reconnaissance d'objets avec SIFT

Projet de **vision par ordinateur** consacré à la détection, la localisation et la reconnaissance d'objets à l'aide des descripteurs locaux **SIFT (Scale-Invariant Feature Transform)**.

Le projet est réalisé sous forme de notebook Jupyter et utilise une partie de la base **COIL-100 (Columbia University Image Library)**.

---

## 📌 Présentation

Le TP est organisé en deux parties complémentaires :

1. **Localisation d'objets avec SIFT**
   - Déterminer si un objet de requête est présent dans une image scène.
   - Localiser l'objet lorsqu'il est présent.
   - Tester le cas d'une scène contenant plusieurs occurrences du même objet.

2. **Reconnaissance d'objets avec SIFT**
   - Construire une base d'apprentissage à partir des descripteurs SIFT.
   - Classifier des images inconnues parmi plusieurs classes.
   - Évaluer les performances avec une matrice de confusion, un rapport de classification et la précision globale.

---

## 🎯 Objectifs

Les principaux objectifs du projet sont :

- Extraire des points d'intérêt et des descripteurs SIFT.
- Comparer des descripteurs entre une image requête et une image scène.
- Utiliser le **BFMatcher** pour effectuer les correspondances.
- Filtrer les correspondances avec le **ratio test de Lowe**.
- Décider de la présence ou de l'absence d'un objet à partir d'un score de correspondance.
- Estimer une homographie avec **RANSAC**.
- Projeter les quatre coins de l'objet dans l'image scène.
- Détecter plusieurs occurrences d'un objet avec une approche de RANSAC itératif.
- Construire une procédure de reconnaissance par vote des meilleurs scores.
- Étudier l'influence du nombre de votes `N` sur la précision.

---

## 🗂️ Dataset

Le projet utilise **COIL-100**, une base d'images d'objets comprenant :

- 100 classes d'objets ;
- 72 images par classe ;
- des rotations de 0° à 355° avec un pas de 5°.

Dans ce TP, seules les **12 premières classes** sont utilisées :

```text
obj1
obj2
obj3
obj4
obj5
obj6
obj7
obj8
obj9
obj10
obj11
obj12
```

Chaque classe contient 72 images.

### Répartition utilisée

Pour chaque classe :

- **36 images pour l'apprentissage** ;
- **36 images pour le test**.

Au total :

| Ensemble | Nombre d'images |
|---|---:|
| Apprentissage | 432 |
| Test | 432 |
| Total | 864 |

La séparation est effectuée selon l'ordre des rotations :

- apprentissage : rotations 0°–175° ;
- test : rotations 180°–355°.

---

## 🔬 Méthode

### Pipeline général

```text
Image
  │
  ▼
Extraction SIFT
  │
  ├── Keypoints
  └── Descripteurs 128 dimensions
  │
  ▼
BFMatcher + Ratio de Lowe
  │
  ▼
Bonnes correspondances
  │
  ├── Score de similarité
  │
  └── Homographie RANSAC
          │
          ▼
      Localisation
```

---

# 1. Localisation d'objets avec SIFT

## 1.1 Extraction des caractéristiques

Pour chaque image, SIFT permet d'extraire :

- des **points d'intérêt (keypoints)** ;
- des **descripteurs SIFT de 128 dimensions**.

L'implémentation utilise :

```python
sift = cv2.SIFT_create()
```

---

## 1.2 Matching avec BFMatcher

Les descripteurs sont comparés avec :

```python
cv2.BFMatcher(cv2.NORM_L2)
```

Le matching utilise `knnMatch(..., k=2)` afin d'obtenir les deux correspondances les plus proches pour chaque descripteur.

---

## 1.3 Ratio test de Lowe

Les correspondances sont filtrées avec un seuil :

```python
RATIO_THRESH = 0.75
```

Une correspondance est conservée lorsque :

```text
distance_meilleure < 0.75 × distance_deuxième
```

Cette étape permet de réduire les fausses correspondances.

---

## 1.4 Score de présence

Le score utilisé pour la première partie est :

```text
score = nombre de bonnes correspondances
        ---------------------------------
        nombre de descripteurs de l'objet
```

Le seuil utilisé est :

```python
SEUIL_SCORE_P1 = 0.10
```

La décision est donc :

```text
score ≥ 0.10  →  OBJET PRÉSENT
score < 0.10  →  OBJET ABSENT
```

### Exemple enregistré dans le notebook

Pour le cas où `obj1__0.png` est recherché dans `obj2__0.png` :

```text
Descripteurs objet      : 84
Descripteurs scène      : 2
Bonnes correspondances  : 4

Score = 4 / 84 = 0.048
Seuil                    = 0.10

Décision                 : Objet ABSENT
```

---

## 1.5 Localisation par homographie

Lorsqu'un objet est considéré comme présent et qu'au moins quatre bonnes correspondances sont disponibles, le notebook estime une homographie avec :

```python
cv2.findHomography(
    src_pts,
    dst_pts,
    cv2.RANSAC,
    5.0
)
```

Le seuil de reprojection RANSAC utilisé est donc :

```text
5.0 pixels
```

L'homographie permet ensuite de projeter les quatre coins de l'image objet dans l'image scène.

Les quatre coins sont identifiés par :

```text
TL — Top Left
BL — Bottom Left
BR — Bottom Right
TR — Top Right
```

---

# 2. Tests de localisation

Trois situations sont étudiées.

### Cas 1 — Objet absent

L'image scène ne contient pas l'objet recherché.

```text
Objet : obj1__0.png
Scène : obj2__0.png
```

Le score obtenu dans le notebook est :

```text
0.048
```

Avec un seuil de `0.10`, la décision est :

```text
ABSENT
```

---

### Cas 2 — Une seule occurrence

L'objet recherché est présent dans la scène.

Le notebook teste notamment :

```text
obj1__0 → obj1__10
obj1__0 → obj1__30
```

La localisation est réalisée grâce à l'homographie estimée par RANSAC.

Le notebook souligne que les faibles rotations sont généralement plus favorables au matching, tandis que l'apparence visible et la quantité de texture peuvent influencer le nombre de points SIFT détectés.

---

### Cas 3 — Plusieurs occurrences

Une scène synthétique contenant deux copies du même objet est construite.

```text
Objet recherché : obj1__0.png
Nombre d'occurrences réelles : 2
```

Deux approches sont comparées :

1. **Homographie standard**
2. **RANSAC itératif avec masquage des inliers**

### RANSAC itératif

Après la détection d'une occurrence :

1. les correspondances considérées comme inliers sont mémorisées ;
2. les keypoints correspondants sont exclus ;
3. le matching est relancé ;
4. une nouvelle homographie est estimée ;
5. le processus continue jusqu'à atteindre le nombre maximal d'itérations ou jusqu'à ce que le score devienne insuffisant.

Cette stratégie permet de rechercher plusieurs transformations correspondant à plusieurs occurrences du même objet.

---

# 3. Reconnaissance d'objets

La deuxième partie transforme le problème en classification.

## 3.1 Préparation des données

Les 12 classes sélectionnées sont séparées en deux ensembles :

```text
36 images/classe → apprentissage
36 images/classe → test
```

Soit :

```text
432 images d'apprentissage
432 images de test
864 images au total
```

---

## 3.2 Cache des descripteurs SIFT

Afin d'éviter de recalculer les descripteurs à chaque exécution, le projet utilise un cache :

```text
sift_cache/
└── sift_descriptors.pkl
```

Si le fichier existe, les descripteurs sont directement chargés.

Sinon, ils sont calculés avec SIFT puis sauvegardés avec `pickle`.

---

## 3.3 Score de similarité

Pour comparer une image test avec une image du modèle, le projet applique également le ratio test de Lowe.

Le score est défini comme :

```text
score = nombre de bonnes correspondances
        ---------------------------------
        nombre de descripteurs de l'image modèle
```

Le même seuil de ratio est utilisé :

```python
ratio_thresh = 0.75
```

---

## 3.4 Classification par vote

Pour chaque image test :

1. elle est comparée aux images de la base d'apprentissage ;
2. les scores de similarité sont calculés ;
3. les meilleures correspondances sont sélectionnées ;
4. la classe est déterminée par vote majoritaire.

Le nombre de votes par défaut est :

```python
N_VOTES = 3
```

Le principe est similaire à une classification par voisins, mais les voisins sont sélectionnés selon le score de correspondance SIFT.

---

# 4. Évaluation

La reconnaissance est évaluée sur :

```text
432 images de test
```

Les métriques et visualisations utilisées sont :

- précision globale ;
- précision, rappel et F1-score par classe ;
- matrice de confusion ;
- matrice de confusion normalisée ;
- comparaison de plusieurs valeurs de `N`.

---

## 📊 Résultats

Avec la configuration par défaut :

```text
ratio_thresh = 0.75
N = 3
```

la précision globale obtenue dans le notebook est :

```text
11.1 %
```

Sur les :

```text
432 images testées
```

Le rapport de classification enregistré dans le notebook donne notamment :

```text
Accuracy       : 0.111
Macro avg      : Precision 0.229 | Recall 0.111 | F1 0.060
Weighted avg   : Precision 0.229 | Recall 0.111 | F1 0.060
```

Ces résultats montrent que, dans cette configuration expérimentale, la reconnaissance multiclasses basée uniquement sur le matching SIFT reste limitée.

---

# 5. Influence du nombre de votes

Le notebook compare trois valeurs de `N` :

| Nombre de votes `N` | Précision |
|---:|---:|
| 1 | 9.5 % |
| 3 | 11.1 % |
| 5 | 13.2 % |

### Observation

Dans cette expérimentation :

```text
N = 1  →  9.5 %
N = 3  → 11.1 %
N = 5  → 13.2 %
```

La précision augmente lorsque le nombre de votes passe de 1 à 5.

---

# ⚙️ Hyperparamètres principaux

| Paramètre | Valeur | Description |
|---|---:|---|
| `RATIO_THRESH` | `0.75` | Seuil du ratio de Lowe |
| `SEUIL_SCORE_P1` | `0.10` | Seuil de présence de l'objet |
| `N_VOTES` | `3` | Nombre de meilleures correspondances utilisées pour le vote |
| `RANSAC reprojThresh` | `5.0 px` | Seuil de reprojection pour les inliers |
| `min_matches` | `8` | Nombre minimal de matches pour la détection multi-occurrence |
| `max_iter` | `5` | Nombre maximal d'itérations du RANSAC multi-occurrence |

---

# 🛠️ Technologies utilisées

- **Python**
- **Jupyter Notebook**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Seaborn**
- **Pickle**
- **KaggleHub** *(optionnel pour le téléchargement automatique du dataset)*

---

# 📦 Installation

Créer un environnement Python puis installer les dépendances :

```bash
pip install opencv-python numpy matplotlib scikit-learn seaborn
```

Pour permettre le téléchargement automatique de COIL-100 avec KaggleHub :

```bash
pip install kagglehub
```

Jupyter Notebook peut être installé avec :

```bash
pip install notebook
```

---

# ▶️ Exécution

Lancer Jupyter :

```bash
jupyter notebook
```

Puis ouvrir :

```text
TP_complet_fixed.ipynb
```

Exécuter les cellules dans l'ordre.

---

# 📁 Structure recommandée

```text
.
├── TP_complet_fixed.ipynb
├── README.md
├── coil-100/
│   └── coil-100/
│       ├── obj1__0.png
│       ├── obj1__5.png
│       ├── ...
│       ├── obj12__350.png
│       └── obj12__355.png
│
└── sift_cache/
    └── sift_descriptors.pkl
```

Le notebook recherche automatiquement la base COIL-100. Si elle n'est pas trouvée, il est possible de définir manuellement :

```python
BASE_PATH = r"C:\Mon\Chemin\coil-100\coil-100"
```

---

# 🔎 Fonctions principales

## Détection et localisation

```python
detect_and_localize(
    img_object_path,
    img_scene_path,
    ratio_thresh=0.75,
    seuil_score=0.10
)
```

Cette fonction :

- charge les images ;
- extrait les descripteurs SIFT ;
- réalise le matching ;
- applique le ratio test ;
- calcule le score ;
- décide de la présence de l'objet ;
- estime l'homographie avec RANSAC ;
- projette les quatre coins de l'objet.

---

## Détection de plusieurs occurrences

```python
detect_multi_occurrences(
    obj_path,
    scene_path,
    ratio_thresh=0.75,
    seuil_score=0.05,
    min_matches=8,
    max_iter=5
)
```

Cette fonction utilise un RANSAC itératif avec exclusion des inliers déjà utilisés.

---

## Calcul du score de similarité

```python
compute_similarity_score(
    des_test,
    des_model,
    ratio_thresh=0.75
)
```

---

## Prédiction d'une classe

```python
predict_class(
    des_test,
    train_features,
    N=3
)
```

La classe est déterminée à partir des `N` meilleures correspondances.

---

# 📈 Visualisations

Le notebook produit notamment :

- visualisation des correspondances SIFT ;
- localisation de l'objet par projection des quatre coins ;
- détection des occurrences multiples ;
- matrice de confusion en valeurs absolues ;
- matrice de confusion normalisée ;
- rapport de classification ;
- graphique de l'influence de `N` sur la précision.

---

# ⚠️ Limites observées

Les résultats du notebook montrent plusieurs limites de cette approche.

### 1. Reconnaissance multiclasses limitée

La précision globale obtenue avec `N=3` est seulement de :

```text
11.1 %
```

sur 12 classes.

### 2. Sensibilité au contenu visuel

Le nombre et la qualité des points SIFT dépendent fortement de la texture et de l'apparence visible de l'objet.

### 3. Fausses correspondances

Même avec le ratio test de Lowe, certaines correspondances incorrectes peuvent subsister.

### 4. Plusieurs occurrences

Une homographie classique décrit une transformation unique. Elle n'est donc pas suffisante, à elle seule, pour localiser plusieurs occurrences ayant des transformations différentes.

Le projet traite ce problème avec une stratégie de RANSAC itératif et de masquage des inliers.

---

# 🧠 Conclusion

Ce TP montre l'utilisation d'une approche classique de vision par ordinateur fondée sur les caractéristiques locales **SIFT**.

Pour la **localisation**, la combinaison :

```text
SIFT
  +
BFMatcher
  +
Ratio de Lowe
  +
Homographie RANSAC
```

permet de détecter et localiser un objet lorsqu'un nombre suffisant de correspondances cohérentes est disponible.

Pour la **reconnaissance multiclasses**, le projet utilise les scores de similarité SIFT entre les images de test et les images d'apprentissage, puis effectue une classification par vote.

Dans les résultats enregistrés, l'augmentation du nombre de votes améliore la précision :

```text
N=1 → 9.5 %
N=3 → 11.1 %
N=5 → 13.2 %
```

L'expérimentation met ainsi en évidence à la fois l'intérêt des descripteurs locaux SIFT pour la mise en correspondance et les limites d'une stratégie reposant uniquement sur le matching de caractéristiques locales pour une reconnaissance multiclasses.

---

## 👤 Auteur

**Herman Kandolo**

Projet académique — Vision par ordinateur
