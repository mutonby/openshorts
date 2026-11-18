# Plazos de conservación reales

**OpenShorts Cloud (openshorts.app) · TONVI TECH SL, B-19780394 · versión de 17 de noviembre de 2026**

Esta tabla es la versión larga del apartado 5 de la
[política de privacidad](https://www.openshorts.app/privacidad). Cada fila
nombra **el código que aplica el plazo**, porque un plazo que no está en el
código no es un plazo: es una intención. Si cambias una constante, cambia esta
tabla y la política en el mismo commit.

Las variables de entorno permiten acortar los plazos en un despliegue, nunca
alargarlos por encima de lo que dice la política sin actualizarla.

---

## 1. Archivos de trabajo en el servidor de procesado

| Qué | Plazo | Dónde se aplica |
|---|---|---|
| Directorio del trabajo (`output/<job_id>`): vídeo de origen descargado (o traído de vuelta de R2 por el editor, `.source_archive.*`), clips renderizados, checkpoint de transcripción, sidecars | **1 h** desde la última escritura | `app.cleanup_jobs`, `JOB_RETENTION_SECONDS` (3600 en cloud; 86400 en autoalojado) |
| Descarga de origen conservada de un trabajo por URL | Igual que el trabajo, o antes si se configura | `app._sweep_retained_sources`, `SOURCE_RETENTION_SECONDS` |
| Subidas pendientes de trabajo (huecos de subida para agentes) | **6 h** | `app._sweep_pending_uploads`, `UPLOAD_TTL_SECONDS` |
| Ficheros en `uploads/` | **1 h** (misma vuelta del barrido) | `app.cleanup_jobs` |
| **Miniaturas generadas y fotogramas muestreados** (`output/thumbnails/<sesión>`) — recortes de la cara del presentador | **24 h desde la última descarga**; servir un fichero reinicia el reloj | `app._sweep_thumbnail_sessions`, `THUMBNAIL_RETENTION_SECONDS` |
| **Fotos de actor subidas** (`output/actor_uploads/*.png`) — fotografías de personas | **1 h** (el reloj del trabajo) | `app._sweep_actor_uploads`, `ACTOR_UPLOAD_RETENTION_SECONDS` |
| Tope de disco: se purgan los trabajos más antiguos por encima del límite | Al superarlo, salvo los tocados en los últimos 30 minutos | `app._enforce_output_size_cap`, `OUTPUT_MAX_GB`, `OUTPUT_CAP_GRACE_SECONDS` |

Antes del 4-sep-2026 las dos filas en negrita **no tenían ningún plazo**: el
barrido horario salta deliberadamente `output/thumbnails`, y `actor_uploads`
solo desaparecía cuando el directorio entero envejecía como si fuese un
trabajo, es decir una hora después de la *última* subida de cualquier usuario.

## 2. Contenido del usuario en almacenamiento durable (Cloudflare R2)

| Qué | Plazo | Dónde se aplica |
|---|---|---|
| Clips del plan gratuito | **7 días**, con aviso por email el día anterior | `cloud/videos.py`, `FREE_CLIP_RETENTION_DAYS` |
| Clips y proyectos de plan de pago | Mientras la suscripción esté activa **+ 7 días** de gracia | `cloud/videos.py`, `VIDEO_RETENTION_GRACE_DAYS` |
| Metadatos de proyecto (incluye la transcripción) | Igual que los clips del mismo trabajo | `cloud/videos.archive_job` |
| **Vídeo de origen de un trabajo de plan de pago** (`users/<usuario>/<trabajo>/source/`), para poder seguir alargando cortes y reencuadrando cuando el servidor ya lo ha borrado | **30 días desde la subida** (barrido diario por fecha del objeto), o antes con la purga de la biblioteca o el borrado de la cuenta. Solo si el plan del dueño es de pago al terminar el trabajo; nunca en el gratuito | `cloud/source_archive.py` (`archive_source`, `purge_expired_sources`), `SOURCE_ARCHIVE_DAYS` (0 desactiva y el barrido borra lo que quede) |
| Todo lo anterior, si el usuario borra la cuenta (vídeo de origen incluido) | **Inmediato** (purga por prefijo antes de tocar la base de datos) | `cloud/account.py` |

## 3. Base de datos (PostgreSQL)

| Qué | Plazo | Dónde se aplica |
|---|---|---|
| `users`, `subscriptions`, `credit_topups`, `usage_ledger`, `projects`, `user_videos`, `api_keys`, `upload_post_profiles`, `signup_attribution`, `oauth_codes`, `clip_expiry_warnings` | Mientras exista la cuenta; **inmediato** al borrarla | `cloud/account.USER_OWNED_TABLES` |
| `magic_link_tokens` (incluye la IP que pidió el enlace) | **30 días** | `cloud/auth.py` |
| **Declaración de derechos** (IP truncada a /24, user-agent, fecha, origen) | Con el proyecto al que pertenece → hasta 5 años, y se borra con la cuenta | `app.py` (captura) → `cloud/videos.archive_job` → `Project.state.rights_attestation` |
| `account_deletions` (sha256 del email, sin la dirección) | **5 años** | `DELETION_LOG_RETENTION_DAYS` |
| `stripe_events` (deduplicación de webhooks) | Mientras dure la retención de eventos de Stripe; sin datos personales | `cloud/billing.py` |
| `proxy_usage` (host de la URL, id de trabajo; **sin usuario**) | Contabilidad de costes; sin datos personales | `cloud/proxy_ledger.py` |

## 4. Navegador del usuario (localStorage, nunca cookies de terceros)

| Qué | Plazo | Dónde |
|---|---|---|
| Token de sesión (JWT) | 30 días o hasta cerrar sesión | `dashboard/src/lib/api.js` |
| Token de medios (para descargar tus propios clips) | 12 h, se renueva solo | `dashboard/src/lib/mediaToken.js`, `MEDIA_TOKEN_TTL_SECONDS` |
| Decisión sobre cookies | Hasta que la cambies, o hasta que cambie la versión del banner | `dashboard/src/lib/consent.js` |
| Preferencias de interfaz, atribución de primera visita | Hasta que borres los datos del sitio | `dashboard/src/lib/attribution.js` |
| Claves BYOK de proveedores de IA | Hasta que las borres. **Ofuscadas, no cifradas** | `dashboard/src/App.jsx` |

## 5. Registros, alertas y copias

| Qué | Plazo | Nota |
|---|---|---|
| Log de aplicación (stdout del contenedor) | **7 días** en el log rotatorio de Docker | Requiere `max-size`/`max-file` en el runtime — ver "pendiente" abajo |
| Copia del log de la API en almacenamiento privado (`cloud/log_archive.py`, solo con `LOG_ARCHIVE_BUCKET`) | **30 días** | Trozos gzip en un bucket sin acceso público; el plazo lo aplica la regla de ciclo de vida del bucket, no el código. Cubre el mismo plazo que las copias de PostgreSQL, así que una supresión también desaparece de los logs en 30 días |
| Alertas operativas a Telegram | Las conserva Telegram | **No contienen email ni título de vídeo** desde el 4-sep-2026: identifican al usuario por los 8 primeros caracteres de su uuid (`cloud/alerts.user_ref`) |
| Copias de seguridad cifradas de PostgreSQL | **30 días** | `ops/pg_backup.sh` — la política promete que una supresión desaparece de toda copia en 30 días, y esto es lo único que lo hace cierto |
| Analítica (OpenPanel, solo con consentimiento) | Identificadores 13 meses; datos brutos 25 meses | Instancia propia |

## 6. Lo que sobrevive deliberadamente a una supresión de cuenta

| Qué | Plazo | Por qué |
|---|---|---|
| Cliente y facturas en Stripe | **6 años** | Art. 30 Código de Comercio |
| Una fila en `account_deletions` con `sha256(email)` | **5 años** | Art. 5.2 RGPD: poder probar que la supresión ocurrió sin conservar la identidad |

---

## Pendiente (no se puede hacer desde el repositorio)

- **Rotación del log del contenedor**: fijar `max-size=50m`, `max-file=3` (o el
  equivalente en Coolify) en el runtime del contenedor de la API. La fila de 7
  días de arriba describe la intención; hasta que esté configurado el plazo real
  depende del demonio de Docker del host.
- **Programar `ops/pg_backup.sh`** en el host de la base de datos y hacer **una
  restauración de prueba**.
- **Retención en OpenPanel**: confirmar en la instancia
  (`openpanel.fotoexamen.com`) que los perfiles caducan a 13 meses y los eventos
  a 25, y borrar el perfil por `profileId` cuando llega el evento
  `AccountDeleted`.
- **Purga de cuentas gratuitas inactivas** (p. ej. 24 meses sin acceso, con
  aviso previo): decidida como política, todavía no implementada.
