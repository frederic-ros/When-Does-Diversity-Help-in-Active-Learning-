# deepanalysis — instrumentation des indicateurs structurels

Objectif : tester la LOI PRÉDICTIVE — l'indicateur structurel a priori
(mesuré aux premiers rounds) prédit-il le gain de la diversité sur margin ?

## Pré-requis (dans alframework/core/)
- runner.py        : version instrumentée (paramètre opt-in log_indicators)
- structural_indicators.py : calcul des deux indicateurs

Ces deux fichiers REMPLACENT/COMPLÈTENT le core. Le runner est rétrocompatible :
sans log_indicators=True, comportement identique à l'ancien.

## Contenu
- bench_synthetic_deep.py : copie du bench synthétique, log_indicators=True
- bench_real_deep.py      : idem tabulaire
- bench_latent_deep.py    : idem latents
  -> produisent des history_*.json ENRICHIS avec les clés ind_* aux rounds 0,1,2
- analyze_indicators.py   : lit ces historiques, calcule Delta, la corrélation
  de Spearman indicateur<->Delta, et la figure indicator_vs_delta.png

## Workflow
1. Déposer runner.py + structural_indicators.py dans alframework/core/
2. Lancer un (ou les 3) bench_*_deep.py  -> historiques enrichis
3. Régler HIST_DIR dans analyze_indicators.py vers le dossier d'historiques
4. python analyze_indicators.py
   -> rho de Spearman + figure. Si rho fort et positif => la loi tient.

## Paramètres clés (analyze_indicators.py)
- INDICATOR_KEY   : "ind_uncertain_eff_dim" (multi-modalité) ou
                    "ind_low_margin_fraction" (volume incertitude)
- INDICATOR_ROUND : round précoce où lire l'indicateur (0 par défaut)
- METRIC          : f1_macro
- CLUSTERING_METHODS / MARGIN_METHOD : définir la famille vs la référence

## Rigueur (pour la review)
- Indicateurs calculés aux rounds 0-2 uniquement => a priori, pas de circularité.
- Delta vient des courbes complètes ; indicateur des premiers rounds. Séparés.
- Tester la robustesse au choix de INDICATOR_ROUND et du quantile incertain.
