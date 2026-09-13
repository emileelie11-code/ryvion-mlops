# Rendu S3 LAB4 - retest 2026-09-13

Contexte de construction : commit `3619050` de `lab/slim`. Architecture linux/amd64, Docker 29.7.2.

| Variante | Taille | Identite | /health |
|---|---|---|---|
| v0 reference | 1.63 GB | root | 200 |
| v1 .dockerignore | 1.62 GB | root | 200 |
| v2 multi-etapes | 796 MB | root | 200 |
| v3 non root | 797 MB | uid=10001(ryvion) | 200 |
| v4 digest | 797 MB | uid=10001(ryvion) | 200 |

Base identifiee par digest : python@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84
