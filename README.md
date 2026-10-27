# sf-dedup-poc — Dédoublonnage + enrichissement Salesforce avec LangGraph

POC pédagogique pour reprendre Python via [LangGraph](https://github.com/langchain-ai/langgraph),
sur un cas d'usage réaliste de data quality Salesforce (dédoublonnage de
Contacts/Leads). Pensé comme preuve technique pour des candidatures type
"GTM/RevOps Engineer" / plateforme interne.

> **🚧 Proof of Concept — ni produit, ni outil de production.**
> Rien n'est réellement exécuté côté Salesforce : `merge_duplicates` décrit
> l'action qu'on effectuerait (merge ou conversion), il ne l'exécute jamais
> contre un org réel. Le checkpointer `human_review` est en mémoire (perdu à
> chaque redémarrage du process), et la revue humaine est simulée par une
> question dans le terminal. Voir [Pistes pour la suite](#pistes-pour-la-suite)
> pour ce qui manquerait avant un vrai déploiement.

## Prérequis & installation

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) pour la gestion des dépendances

```bash
git clone <url-du-repo>
cd sf-dedup-poc
uv sync
```

Aucune dépendance externe n'est requise pour tourner en mode mock (pas de
clé API, pas de sandbox Salesforce). `ANTHROPIC_API_KEY` et un `.env`
(copié depuis `.env.example`) ne sont nécessaires que pour le jugement LLM
réel et le branchement sur un vrai sandbox — voir la section correspondante
plus bas.

## Le graphe

```
START -> block_and_match -> judge_pairs -> split_by_confidence
                                                   |
                                 (conditionnel, route_after_split)
                                    /                          \
                          human_review                    merge_duplicates
                                \                              /
                                 -----> merge_duplicates <-----
                                              |
                                      enrich_contacts -> log_and_monitor -> END
```

- **`block_and_match`** : **blocking** (regroupement par nom normalisé +
  initiale du prénom, pour éviter une comparaison O(n²) — indispensable à
  l'échelle d'un vrai org) puis **scoring** composite (rapidfuzz sur le
  nom, l'email, le téléphone) → paires candidates au-dessus d'un seuil bas
  (55/100), volontairement large pour inclure la zone ambiguë.
- **`judge_pairs`** : une décision *doublon / pas doublon* par paire.
  Score ≥ 92 → décision heuristique directe (pas besoin de LLM sur les cas
  évidents). Sinon → jugement LLM (`judge.py`), avec retries (`tenacity`,
  backoff exponentiel) et repli heuristique automatique si l'appel échoue
  après plusieurs tentatives — le graphe ne casse jamais à cause d'un
  souci réseau/rate limit sur le LLM. **Sans clé API**
  (`ANTHROPIC_API_KEY` non configurée), tout tourne en mode heuristique
  pur ("mode démo").
- **`split_by_confidence`** : sépare les jugements "doublon" en deux
  paniers selon la confiance — `high_confidence` (≥ 0.85) et
  `needs_review` (< 0.85).
- **edge conditionnel** (`route_after_split`) : s'il n'y a rien
  d'ambigu, on saute direct à `merge_duplicates` ; sinon on passe par
  `human_review`.
- **`human_review`** : point d'interruption **human-in-the-loop** via
  `interrupt()`. Le graphe se met en pause, renvoie les paires ambiguës à
  l'appelant (dans `run.py`, ça se traduit par une question posée dans le
  terminal), et reprend exactement où il s'était arrêté une fois
  `Command(resume=decisions)` envoyé. Ça nécessite un *checkpointer*
  (`MemorySaver`, en mémoire ici — en prod on prendrait un backend
  persistant type Postgres/SQLite pour survivre à un redémarrage pendant
  l'attente d'une validation).
- **`merge_duplicates`** : applique les rapprochements décidés (haute
  confiance + validés côté humain). Choisit le survivant avec une règle
  simple mais réaliste (Contact plutôt que Lead ; entre deux mêmes types,
  email pro plutôt que perso). Distingue deux cas : `MERGE` (même type
  d'objet, un vrai merge Salesforce serait possible) et
  `CONVERSION CANDIDATE` (Lead vs Contact — Salesforce ne fait pas de merge
  entre les deux types, il faudrait rapprocher/convertir le Lead à la
  place). Ne touche évidemment à aucune vraie donnée dans ce POC — décrit
  juste l'action qu'on exécuterait.
- **`enrich_contacts`** (`enrichment.py`) : pour chaque fiche **survivante**
  (jamais les fiches absorbées par un merge — inutile de gaspiller un appel
  réseau sur une fiche qui va disparaître), deux signaux dérivés du domaine
  email : (1) une **vérification MX** via l'API publique DNS-over-HTTPS de
  Google (gratuite, sans clé, retries limités + repli propre
  `mx_valid=None` si le réseau est indisponible — jamais interprété comme
  "domaine invalide") — un vrai signal de data quality (ce domaine peut-il
  recevoir des emails ?) ; (2) un **nom d'entreprise deviné** à partir du
  domaine (heuristique locale, sans réseau — le point d'extension évident
  pour brancher un vrai provider d'enrichissement firmographique type
  Clearbit/Apollo, suivant exactement le même pattern que
  `judge.get_chat_model()`). Mémoïse les appels par domaine pour ne
  requêter qu'une fois par entreprise, même si plusieurs fiches partagent
  le même domaine.
- **`log_and_monitor`** : résumé final (décisions par source, nombre de
  merges, fiches enrichies) — exactement le genre de payload qu'on
  pousserait vers un webhook Slack en fin de synchronisation.

## Lancer (données mock)

```bash
uv run python -m sf_dedup_poc.run
```

S'il y a des paires ambiguës, le script pose la question directement dans
le terminal (`Fusionner ces deux fiches ? [o/N]`) — c'est la démonstration
concrète du human-in-the-loop. Sans terminal interactif (ex: script lancé
dans un pipeline CI), la revue est rejetée par défaut plutôt que de
planter.

## Tests

```bash
uv run pytest
```

56 tests, aucun appel réseau ni clé API requise (le fixture `_no_llm` de
`tests/test_graph_smoke.py` force le mode heuristique même si
`ANTHROPIC_API_KEY` est configurée dans l'environnement, et
`_no_real_dns_calls` fait pareil pour l'enrichissement MX). Répartition :

- `test_pair_score.py` : le scoring composite, et en particulier les deux
  régressions trouvées en testant sur un vrai sandbox / en écrivant ces
  tests — champs vides qui matchaient entre eux (faux positifs), et
  numéros au format `+33 6...` qui ne matchaient pas leur équivalent
  `06...` (faux négatifs, corrigé dans `_digits`).
- `test_blocking.py` : le blocking garantit qu'on ne compare jamais deux
  contacts de blocks différents, même très similaires par ailleurs.
- `test_judge.py` : décision heuristique (mode démo) et repli automatique
  quand l'appel LLM échoue — sans jamais faire un vrai appel réseau.
- `test_merge_logic.py` : quel enregistrement survit (Contact > Lead, email
  pro > perso), et MERGE vs CONVERSION CANDIDATE.
- `test_mock_data.py` : déterminisme du générateur (seed fixe), utilisé
  comme vérité terrain dans `run.py`.
- `test_enrichment.py` / `test_enrich_contacts_node.py` : la vérification
  MX (domaine valide/invalide/générique/non vérifiable), la mémoïsation par
  domaine, et surtout la non-régression sur un `tenacity.RetryError` qui
  fuitait — voir "Bugs trouvés" ci-dessous.
- `test_graph_smoke.py` : le graphe compilé de bout en bout, y compris le
  cycle `interrupt()` / `Command(resume=...)` du human-in-the-loop —
  la partie la plus facile à casser silencieusement en retouchant le
  câblage (`graph.py`) sans s'en rendre compte.

### Bugs trouvés en écrivant ces tests

Trois régressions concrètes, dans l'ordre où elles ont été trouvées — un
bon résumé de "pourquoi tester contre des cas réels, pas juste des
happy paths" :

1. **Champs vides qui matchent entre eux** (`_pair_score`) : `"" == ""` et
   `fuzz.ratio("", "")` valent tous les deux "match parfait" → faux
   positifs sur des données sparse (vu sur un vrai sandbox Salesforce).
2. **Préfixe pays non normalisé** (`_digits`) : `+33 6...` et `06...` (le
   même numéro) ne matchaient pas → faux négatifs, trouvé en écrivant les
   tests de non-régression du bug n°1.
3. **`tenacity.RetryError` qui fuitait** (`_query_mx` dans
   `enrichment.py`, et par ricochet les deux fonctions retried de
   `salesforce_source.py`) : sans `reraise=True` sur `@retry`, une fois les
   tentatives épuisées, tenacity lève son propre `RetryError` au lieu de
   l'exception réseau d'origine — un type que le code appelant n'attrapait
   pas, donc un souci réseau persistant faisait planter le noeud avec une
   trace brute au lieu de dégrader proprement. Trouvé en testant
   l'enrichissement contre un réseau qui bloque réellement `dns.google`
   (proxy sortant restrictif) ; corrigé aux trois endroits concernés.

## Concepts LangGraph vus dans ce POC

- **State** (`state.py`) : le `TypedDict` qui circule entre les noeuds.
  Par défaut une clé est *remplacée* par le dernier noeud qui la
  retourne ; `log` utilise `Annotated[list[str], operator.add]` pour au
  contraire *accumuler* les entrées de plusieurs noeuds au fil du graphe.
- **Node** (`nodes.py`) : une fonction `(state) -> dict`. Pas de magie —
  même quand elle appelle un LLM ou qu'elle interrompt le graphe.
- **Graph** (`graph.py`) : `StateGraph`, `add_node`/`add_edge`,
  `add_conditional_edges` pour brancher selon une fonction de routage, et
  `.compile(checkpointer=...)` pour activer la persistance nécessaire à
  `interrupt()`.
- **Robustesse d'un appel LLM** (`judge.py`) : retries + repli heuristique
  encapsulés, le node appelant reste simple.
- **Human-in-the-loop** (`nodes.py::human_review`, `graph.py`) :
  `interrupt()` / `Command(resume=...)` pour suspendre puis reprendre le
  graphe avec une décision humaine — sans réécrire la logique en machine à
  états à la main.

## Brancher un vrai sandbox Salesforce

`raw_contacts` est une simple liste de dicts (`id, first_name, last_name,
email, phone, company, city, object_type, duplicate_of`). C'est ce
découplage entre le State et la source des données qui permet de brancher
`salesforce_source.py` à la place de `mock_data.py` sans toucher au reste
du graphe.

`salesforce_source.py` supporte deux flows OAuth (`SF_AUTH_FLOW` dans
`.env`) et récupère ensuite Contact + Lead via l'API REST (SOQL, avec
pagination).

### Flow par défaut : Client Credentials (`SF_AUTH_FLOW=client_credentials`)

C'est le flow le plus simple à mettre en place — pas de certificat, juste
un Consumer Key + Secret et un "Run As User" configuré côté Salesforce.

1. **Créer un External Client App** (Setup → External Client App Manager →
   New External Client App) : cocher "Enable OAuth Settings", scope OAuth
   au minimum `api`. Note le **Consumer Key** et le **Consumer Secret**.
2. **Configurer les policies** (bouton "Policies" / "Edit Policies" sur
   l'app — c'est séparé de la définition de l'app dans les External Client
   Apps, contrairement aux anciennes Connected Apps) : activer
   "Enable Client Credentials Flow" et renseigner un **Run As User** —
   c'est cet utilisateur qui détermine les droits avec lesquels tourneront
   les requêtes SOQL (permissions sur Contact/Lead, en particulier).
3. **Remplir `.env`** (copier `.env.example`) : `SF_CLIENT_ID`,
   `SF_CLIENT_SECRET`, et `SF_LOGIN_URL` = le **My Domain** de ton org
   (ex. `https://mon-org--nomdusandbox.sandbox.my.salesforce.com`) — **pas**
   `login.salesforce.com`/`test.salesforce.com` : sans étape de login
   utilisateur, Salesforce a besoin de l'URL spécifique à l'org pour savoir
   à quel External Client App s'adresser. Tu trouves cette URL dans Setup →
   My Domain, ou dans l'URL de ton sandbox une fois connecté dessus.
4. **Lancer** :
   ```bash
   uv run python -m sf_dedup_poc.run --source salesforce --limit 200
   ```

### Flow alternatif : JWT Bearer (`SF_AUTH_FLOW=jwt`)

Si la politique de l'org impose une assertion signée plutôt qu'un secret
partagé : générer un certificat auto-signé (la clé privée ne quitte jamais
la machine) —

```bash
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -keyout certs/server.key -out certs/server.crt -days 365 -nodes
```

— cocher "Use digital signatures" sur l'External Client App et uploader
`certs/server.crt`, pré-autoriser l'utilisateur d'intégration dans les
policies ("Permitted Users" = *Admin approved users are pre-authorized* +
Permission Set), puis renseigner dans `.env` : `SF_AUTH_FLOW=jwt`,
`SF_CLIENT_ID`, `SF_USERNAME`, `SF_LOGIN_URL`
(`https://test.salesforce.com` pour un sandbox — ici on est dans le cas
"avec" étape de login, donc le domaine générique convient) et
`SF_PRIVATE_KEY_PATH=./certs/server.key`.

### Dans les deux cas

Sans `.env` valide, le script s'arrête proprement avec un message clair
(`SalesforceAuthError`) plutôt que de planter avec une trace Python. Même
pattern retries/tenacity que dans `judge.py` sur les appels HTTP
(authentification + pagination SOQL), mais seulement sur les erreurs
réseau/transitoires — jamais sur nos propres `SalesforceAuthError`
(identifiants invalides, token expiré...) : ça ne se répare pas en
retentant, autant échouer vite avec un message clair.

Comme les Lead et les Contact peuvent matcher entre eux (un classique du
dédoublonnage), `merge_duplicates` distingue les deux cas — voir plus haut.
Attention : sans vérité terrain connue sur de vraies données, il faut
regarder les résultats à l'oeil (ou passer davantage de paires par
`human_review` en abaissant `MERGE_CONFIDENCE_THRESHOLD` dans `nodes.py`
le temps de valider la logique sur ce sandbox précis).

## Pistes pour la suite

- Regrouper les paires en **clusters** (union-find) plutôt qu'en paires
  isolées, pour que trois doublons du même contact ne redonnent pas trois
  décisions indépendantes.
- Remplacer la validation terminal par un vrai canal (Slack, UI interne),
  en gardant `human_review` identique : il ne connaît que la forme du
  dict de décisions attendu au `resume`.
- Exécuter un vrai merge/rapprochement via l'API Salesforce (Bulk API
  `Database.merge` pour les `MERGE`, un flow de conversion pour les
  `CONVERSION CANDIDATE`) plutôt que de se contenter de les lister.
- Paralléliser `enrich_contacts` (aujourd'hui séquentiel, un appel réseau
  par domaine distinct) — avec un vrai provider d'enrichissement
  firmographique en plus de la vérification MX.
- Brancher un vrai provider d'enrichissement firmographique (Clearbit,
  Apollo...) derrière `enrichment.py`, sur le même point d'extension que
  `judge.get_chat_model()` — le nom d'entreprise deviné resterait le repli
  "mode démo" sans clé API.
