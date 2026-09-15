# Grandes modelos de lenguaje ajustados como subrogados cognitivos en la Iowa Gambling Task

Código del estudio empírico del Trabajo de Fin de Máster de David Alarcón Rubio (UNED, 2026). El estudio compara la
predicción ensayo a ensayo de la elección humana en la Iowa Gambling Task entre *Centaur*, su modelo base
*Llama-3.1-70B*, siete adaptadores LoRA propios (dos controles sin contenido conductual, cuatro especialistas y un control de diversidad), la base con los pesos reinicializados al azar y tres modelos cognitivos clásicos (*VSE*, *ORL* y *PVL-Δ*), sobre 1.041
sesiones de catorce estudios de origen.

| Carpeta | Contenido |
|---|---|
| `A1_modelos_cognitivos/` | Implementación de VSE, ORL y PVL-Δ y su ajuste por máxima verosimilitud con validación cruzada de cinco pliegues |
| `A2_configuracion_entrenamiento/` | Entrenamiento de los adaptadores LoRA: entrenador con pérdida enmascarada a los tokens de respuesta, ficheros de configuración, lanzadores y cuadernos de Colab |
| `A3_formato_corpus/` | Conversión de las sesiones al formato de texto de Psych-101 y construcción de los corpora de adaptación (subcorpus IGT, complemento sin IGT, ruido, texto irrelevante) |
| `A4_ciclo_entrenar_evaluar/` | Inferencia: servidor y cliente que obtienen la probabilidad de cada elección humana y evaluación de cada adaptador |
| `A5_analisis_estadisticos/` | Análisis estadísticos del estudio: contrastes pareados por sesión, bootstrap por estudios de origen, permutación del signo, exclusión sucesiva por estudio, d agregada, sensibilidad renorm4, masa de probabilidad sobre los cuatro mazos, verosimilitud absoluta por modelo y partición, verosimilitud sobre las mismas sesiones que los modelos cognitivos y contrastes entre los dos especialistas de cada par, junto con los cinco módulos comunes |

Entorno: Python 3.11 con torch 2.10.0, transformers 4.49.0, peft 0.19.1, bitsandbytes 0.49.2 y datasets 4.8.5 para
entrenar y evaluar los modelos de lenguaje en una GPU de 80 GB; numpy y scipy para los modelos cognitivos y los
análisis. Modelos y corpus: `meta-llama/Llama-3.1-70B`, `marcelbinz/Llama-3.1-Centaur-70B-adapter` y
`marcelbinz/Psych-101` en Hugging Face. Los datos generados y los pesos de los adaptadores entrenados se facilitan a
petición.

Licencia MIT (fichero `LICENSE`); dependencias en `requirements.txt`.
