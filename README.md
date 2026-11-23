# Classification de scènes : Vision Transformer et CNN

Comparaison de trois approches de classification d'images sur le jeu de données Intel Image Classification, en PyTorch. Projet réalisé dans le cadre du DESS en science des données et intelligence artificielle de l'UQO.

## Approches comparées

| Modèle | Description |
|:---|:---|
| ViT entraîné de zéro | Vision Transformer codé à la main : découpage en patchs de 16 pixels, jeton de classification, encodage de position, 6 couches d'encodeur à 6 têtes d'attention. |
| ViT préentraîné | `vit_base_patch16_224` de la bibliothèque `timm`, affiné avec AdamW et un taux d'apprentissage cosinus. |
| ResNet18 | Réseau convolutif préentraîné sur ImageNet, dont seule la dernière couche est remplacée (apprentissage par transfert). |

## Jeu de données

[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) contient environ 14 000 images d'entraînement et 3 000 images de test, réparties en six classes : `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`.

Le jeu de données n'est pas inclus. Après téléchargement depuis Kaggle, placer `seg_train` et `seg_test` dans un dossier `data/` (la structure imbriquée de l'archive Kaggle est reconnue).

## Utilisation

```bash
pip install -r requirements.txt
python train_vit_cnn_comparison.py --data-dir data --output-dir resultats
python train_vit_only.py --data-dir data --output-dir resultats
```

Le GPU est utilisé automatiquement s'il est disponible.

`train_vit_cnn_comparison.py` entraîne le ViT de zéro et le ResNet18, puis produit dans `resultats/` les courbes d'apprentissage, les matrices de confusion, les rapports de classification (précision, rappel et F1 par classe), les prédictions sur le jeu de test en CSV et les poids des deux modèles. Il affiche aussi un tableau comparatif avec l'exactitude et la durée d'entraînement mesurée.

`train_vit_only.py` affine le ViT préentraîné, garde le meilleur modèle selon la validation et l'évalue sur le jeu de test.

## Méthode

Les deux scripts réservent 20 % des images d'entraînement à la validation, avec une graine fixée pour rendre la séparation reproductible. L'augmentation de données combine recadrage aléatoire, miroir horizontal, rotation et variation de couleur. La normalisation suit les statistiques ImageNet pour les modèles préentraînés.

## Résultats

Exactitude mesurée sur le jeu de test `seg_test` (3 000 images, jamais utilisées pendant l'entraînement ni pour choisir le meilleur modèle) :

| Modèle | Exactitude (test) |
|:---|---:|
| ViT préentraîné (`vit_base_patch16_224`) | 100 % |
| ResNet18 (apprentissage par transfert) | 98 % |
| ViT entraîné de zéro | 92 % |

Les deux modèles préentraînés font mieux que le ViT entraîné de zéro. Avec environ 14 000 images d'entraînement, un Transformer a peu de données pour apprendre seul ses représentations ; les poids appris sur ImageNet lui donnent un point de départ bien plus solide.

## Technologies

Python, PyTorch, torchvision, timm, scikit-learn, pandas, Matplotlib, seaborn.
