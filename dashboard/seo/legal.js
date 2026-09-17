/* Legal pages: Terms of Service, Privacy Policy and Legal Notice, in English
 * and Spanish. Spanish versions are not a nicety: Art. 60.1 TRLGDCU requires
 * pre-contractual information "at least in Castilian Spanish" for consumers in
 * Spain, and the AEPD has fined sites whose privacy policy existed only in a
 * foreign language. The prevalence clause (Spanish version wins for consumers
 * resident in Spain) lives in each document.
 *
 * These pages are emitted as static HTML like the marketing pages, and appear
 * in sitemap.xml and llms.txt, but stay OUT of the marketing interlinking
 * ring: they are reachable from the footer, which is where people (and
 * crawlers) expect them.
 *
 * Substantive choices are documented in the repo history: consumer-withdrawal
 * wording follows arts. 98.8/103.a/108 TRLGDCU as read after CJEU C-234/25
 * (SaaS is a digital service, not digital content — the right survives but is
 * paid pro-rata); the 15-day renewal reminder follows Ley 10/2025; the
 * moderation/takedown sections implement DSA arts. 11-17 for a hosting
 * service; AI labelling follows AI Act art. 50. The DMCA block inside section
 * 9 is not a duplicate of the DSA one: the DSA notice-and-action mechanism
 * does not grant the US safe harbor, and 17 U.S.C. 512(c) needs a designated
 * agent both published here AND registered with the US Copyright Office —
 * publishing this text is only half of it.
 */

const PUBLISHED = '2026-09-04'

const TERMS_EN = `
<h2>1. Who we are</h2>
<p>OpenShorts Cloud (openshorts.app) is operated by TONVI TECH SL, a company
incorporated in Spain, CIF B-19780394, with registered address at Calle Puerta
del Mar 18, 5th floor, 29005 Málaga, Spain ("OpenShorts", "we", "us"). You can
reach us at <a href="mailto:info@openshorts.app">info@openshorts.app</a>. These Terms of Service ("Terms") govern your
use of the hosted service at openshorts.app.</p>
<p>The open-source edition of OpenShorts that you can download and run on your
own hardware is licensed separately under the MIT License and is not covered by
these Terms: when you self-host, you are the operator of your instance and
these Terms do not apply to it.</p>
<p>These Terms are also available in <a href="/terminos">Spanish</a>. For
consumers resident in Spain, the Spanish version prevails in case of
discrepancy.</p>

<h2>2. The service</h2>
<p>OpenShorts turns long-form video into short vertical clips. On your
instruction, the service ingests a video you upload or a video located at a URL
you provide, transcribes it, uses AI models to select moments, reframes it to
vertical, and optionally adds subtitles, text overlays, AI dubbing, and
publishes the result to social media accounts you connect. The service is
available as a free plan with usage limits and a watermark, and as paid
subscriptions and top-ups billed through Stripe.</p>

<h2>3. Eligibility and accounts</h2>
<p>You must be at least 16 years old, or the age of digital consent in your
country if it is higher. You are responsible for everything done through your
account and your API keys. Keep your access credentials confidential and tell
us at <a href="mailto:info@openshorts.app">info@openshorts.app</a> if you believe your account has been compromised.</p>

<h2>4. Your content stays yours</h2>
<p>You keep all ownership of the videos you submit. By submitting content, you
grant us a worldwide, non-exclusive, royalty-free licence to host, store,
reproduce, transcode, modify and process that content, and to display it back
to you, strictly to the extent needed to operate and secure the service. This
licence ends when the content is deleted from our systems under the retention
rules in Section 13, except for backup copies kept for a limited period and
content we must keep to comply with the law.</p>
<p><strong>The clips are yours.</strong> As between you and us, you own the
output clips the service generates from your content, including for commercial
use, to the extent of our rights in them. On the free plan, clips carry an
OpenShorts watermark; you may use watermarked clips freely but you may not
remove or obscure the watermark by any means other than upgrading to a paid
plan.</p>
<p>We do not use your content to train AI models — ours or anyone else's. See
Section 7.</p>
<p>With your separate permission (for example, if you reply yes when we ask), we
may mention your name or showcase your public clips as customer references. We
never do this by default.</p>

<h2>5. Your promises about the content</h2>
<p>The service processes content on your instruction — both files you upload
and videos you direct us to fetch from a URL. You promise that, for every video
you submit by either route:</p>
<ul>
<li>you own it, or you hold the rights or permissions needed to download,
process, edit and republish it;</li>
<li>your use of it through the service does not infringe anyone's copyright,
trademark, image or publicity rights, or the terms of the platform it came
from;</li>
<li>it does not feature any identifiable person's image or voice without the
authorisation the law requires; and</li>
<li>if you use AI dubbing, you have the voice owner's consent to generate
synthetic speech from their voice.</li>
</ul>
<p>When you submit a job you tick a declaration confirming this. We record that
declaration together with the date, your IP address and browser identifier, and
we may produce that record if a dispute arises. Submitting other people's
content without permission is the thing most likely to get an account
terminated under Section 16.</p>

<h2>6. Personal data in your videos — our role as processor</h2>
<p>Videos often contain personal data of the people who appear in them. For
that data, you are the controller and we act as your processor under Article 28
GDPR: we process the footage only on your documented instructions (the jobs you
submit), we engage only the sub-processors listed in our
<a href="/privacy">Privacy Policy</a> (AI, storage and delivery providers), we
apply appropriate security measures, and we delete the content under the
retention rules in Section 13 or earlier on your instruction. If you need a
countersigned Data Processing Agreement for your business, request it at
<a href="mailto:info@openshorts.app">info@openshorts.app</a>. When you self-host OpenShorts with your own keys, your
content flows directly between your own instance and the AI provider under
your own agreement with them. On OpenShorts Cloud that is not the case even
when you supply your own provider key: the request is made by our servers,
which means the content passes through them and this Section 6 applies to it
in full.</p>

<h2>7. AI processing — what runs where</h2>
<p>The service uses AI systems to do its work, and we name them: transcription
runs on our own servers (Whisper-family and NVIDIA Parakeet models); moment
selection, titling, layout decisions and generated thumbnails use Google's
Gemini API; optional voice dubbing and AI voices use ElevenLabs; AI Shorts
uses fal.ai (Flux for actor images, Kling and Hailuo for the animated
presenter). Your video content, audio, transcript, sampled frames and any
actor photo you supply are sent to these providers only as needed to perform
the job you requested, under agreements that prohibit them from using
paid-API customer data to train their models. Section 4 of the
<a href="/privacy">Privacy Policy</a> says exactly what each one receives.</p>
<p><strong>We do not train AI models on your content, and we do not sell it or
share it for training.</strong></p>
<p>AI output is probabilistic: clip selections, transcripts, translations,
titles and dubbed audio may contain errors, and similar inputs can produce
similar outputs for different customers, so results are not guaranteed to be
unique. You are responsible for reviewing AI output before you use or publish
it. You may not use the service or its output to develop or train a competing
model or service.</p>
<p>Clips that use AI dubbing contain synthetic speech, and so do videos made
with an AI actor and thumbnails generated by the image model. We mark those
files as AI-generated in a machine-readable format — a metadata tag on video,
an XMP <code>DigitalSourceType: trainedAlgorithmicMedia</code> marking on
generated images — which is what Article 50(2) of Regulation (EU) 2024/1689
(the AI Act) requires of us as the provider; you must not remove or strip that
marking. Be aware that platforms commonly re-encode uploads and that a
re-encode can drop it. We do not burn a visible label into the picture. When you publish one of those
clips, revealing to your audience that it is artificially generated is your own
duty as the deployer under Article 50(4), and the AI-content label each
platform offers (TikTok, Instagram, YouTube and the rest) is the normal way to
discharge it.</p>

<h2>8. Acceptable use</h2>
<p>You may not use the service to:</p>
<ul>
<li>process or distribute content that is unlawful, defamatory, harassing,
hateful, sexually exploitative of minors, or that incites violence;</li>
<li>infringe intellectual property, image or privacy rights, including
submitting URLs of content you have no rights to;</li>
<li>impersonate any person, or produce media of an identifiable person's image
or voice without authorisation (deepfakes);</li>
<li>probe, disrupt or overload the service, scrape it, or access it by any
automated means other than the documented API and MCP endpoints within their
rate limits;</li>
<li>reverse engineer the hosted service or attempt to extract its models,
prompts or non-public components;</li>
<li>sell, transfer or share API keys, accounts or quota, or create multiple
accounts to evade free-plan limits;</li>
<li>artificially manipulate engagement metrics on any platform you publish
to.</li>
</ul>
<p>This section, together with Sections 5 and 9, is our content moderation
policy for the purposes of Article 14 of Regulation (EU) 2022/2065 (DSA):
content that breaches it may be removed and accounts that breach it may be
throttled, suspended or terminated, with reasons given to the affected user. We
may block re-registration by the same person after a termination for abuse.</p>

<h2>9. Copyright complaints and content takedown</h2>
<p>The full procedure, with what a notice needs and how to challenge a removal,
is on its own page: <a href="/report-content">Report Illegal Content</a>.</p>
<p>If you believe content processed or hosted through the service infringes
your rights, email <a href="mailto:info@openshorts.app">info@openshorts.app</a> (our point of contact for users and
authorities under Articles 11 and 12 DSA) with: your name and contact details,
the work you own, the exact URL or identifier of the infringing material, a
statement in good faith that the use is unauthorised, and your signature. We
review notices promptly, remove or disable access to content when the notice is
substantiated, and inform the affected user with reasons; the affected user may
reply with a substantiated counter-notice. We terminate the accounts of repeat
infringers. This procedure implements the notice-and-action mechanism of
Article 16 DSA.</p>
<h3>Notices under the US DMCA</h3>
<p>For rightsholders in the United States, TONVI TECH SL has designated the
following agent to receive notifications of claimed infringement under 17
U.S.C. § 512(c)(2):</p>
<p>Copyright Agent, TONVI TECH SL, Calle Puerta del Mar 18, 5th floor, 29005
Málaga, Spain. Email: <a href="mailto:info@upload-post.com">info@upload-post.com</a>.</p>
<p>That address is the one on file with the US Copyright Office for this
company, which operates openshorts.app and upload-post.com. Notices sent to
<a href="mailto:info@openshorts.app">info@openshorts.app</a> are acted upon too, but the address above is the formal
channel.</p>
<p>A notification must be in writing and include: a physical or electronic
signature of the owner or of a person authorised to act for them; identification
of the copyrighted work claimed to be infringed; identification of the material
you say is infringing and where it can be found on the service; your address,
telephone number and email; a statement that you have a good-faith belief that
the use is not authorised by the owner, its agent or the law; and a statement,
made under penalty of perjury, that the information in the notice is accurate
and that you are the owner or authorised to act on the owner's behalf.</p>
<p>If your material was removed and you believe the removal was a mistake or a
misidentification, you may send a counter-notification to the same agent under
17 U.S.C. § 512(g), including your signature, identification of the material
and where it appeared, a statement under penalty of perjury that you believe in
good faith it was removed by mistake or misidentification, and your consent to
the jurisdiction of the US federal district court for the district where you
live, or, if you live outside the United States, of any district in which we
may be found. We forward the counter-notification to the complainant and may
restore the material after 10 business days unless they tell us they have filed
a court action seeking to restrain you.</p>
<p>Under 17 U.S.C. § 512(f), anyone who knowingly and materially misrepresents
that material is infringing, or that it was removed by mistake, is liable for
the damages that misrepresentation causes.</p>

<h2>10. Publishing to your social accounts</h2>
<p>If you connect social accounts (via our publishing partner, Upload-Post),
you expressly authorise us and our partner to publish content to those accounts
on your instruction. You remain the publisher of everything posted: you are
responsible for the content, its scheduling and its compliance with each
platform's terms (YouTube, TikTok, Instagram and the rest). You can disconnect
your accounts at any time from the dashboard or from the platform's own
settings, which revokes our access.</p>

<h2>11. Plans, billing and renewals</h2>
<p>Paid subscriptions and minute top-ups are billed by Stripe; we never see or
store your card number. Prices are shown with any applicable taxes indicated at
checkout. Subscriptions renew automatically at the end of each billing period
until you cancel. You can cancel at any time
from your account page, as easily as you subscribed; cancellation takes effect
at the end of the current period, which you keep using in full. Unused quota
does not roll over unless the plan says otherwise. We will give you at least 30
days' notice by email before any price increase takes effect, and you may
cancel before it applies.</p>

<h2>12. Right of withdrawal (EU consumers)</h2>
<p>If you are a consumer in the EU, you have a statutory right to withdraw from
a distance contract within 14 days without giving a reason. When you subscribe,
you expressly request that the service start immediately, within the withdrawal
period. You keep your right of withdrawal during those 14 days, but if you
exercise it you will pay the proportional part of the service already provided,
and the right is lost once the service has been fully performed (Articles 98.8,
103.a and 108 of the Spanish Consumer Act, implementing Directive 2011/83/EU).
Automatic renewals of an ongoing subscription do not open a new withdrawal
period. To withdraw, email <a href="mailto:info@openshorts.app">info@openshorts.app</a> or use the model form we include
in your confirmation email; refunds are issued to the original payment method
within 14 days.</p>
<p>Separately from this right, our <a href="/refunds">refund policy</a> returns
your most recent charge in full if you have not used the service since it was
taken, which covers the renewal nobody noticed. The two do different jobs: the
statutory right above is the one that helps you when you did use some of your
minutes, because it is not conditional on leaving the service untouched.</p>

<h2>13. Content retention and deletion</h2>
<p>Generated clips on the free plan are stored for 7 days and then permanently
deleted; we email you before it happens. Download what you want to keep. On
paid plans, your projects and clips remain stored while your subscription is
active and are deleted 7 days after it ends unless you delete them earlier.
Deleting your account deletes your content on the same schedule, subject to
short-lived backups and to records we must keep by law (such as invoicing
data). We are not liable for the loss of content whose deletion these Terms
announced.</p>

<h2>14. Disclaimers</h2>
<p>The service is provided "as is" and "as available", without warranties of
uninterrupted availability, error-free operation, or fitness for a particular
purpose, to the extent such disclaimers are permitted by the law of your
residence. Nothing in these Terms limits warranties or rights that consumer law
grants you and that cannot be waived.</p>

<h2>15. Liability</h2>
<p>To the maximum extent permitted by law, our total aggregate liability for
claims arising out of the service is limited to the amounts you paid us in the
12 months before the event giving rise to the claim, and we are not liable for
indirect or consequential damages, loss of profits, or loss of data whose
deletion was announced in Section 13. This limitation does not apply to damages
caused by our wilful misconduct or gross negligence, to death or personal
injury, or to any liability that cannot be limited under Spanish or EU law.</p>

<h2>16. Indemnity, suspension and termination</h2>
<p>If a third party brings a claim against us because of content you submitted
or instructed us to fetch, or because of your breach of these Terms, you will
indemnify us — including our directors, employees and suppliers — for the
resulting damages, penalties and reasonable defence costs, except to the extent
the claim results from our own breach. If you are a consumer, this obligation
applies only to claims caused by your breach of these Terms or of the law.</p>
<p>We may suspend or terminate accounts for material breach. Where the breach
is curable we will tell you first and give you a reasonable period to fix it;
where it is not (illegal content, repeat infringement, abuse, fraud), we may
suspend immediately, giving you reasons. After termination you have 7 days to
download your remaining content unless the law requires us to remove it.
Sections 4 to 9 and 14 to 18 survive termination.</p>

<h2>17. Changes to the service and these Terms</h2>
<p>We improve the service continuously and may change or retire features. We
may amend these Terms; material changes will be announced by email or in-app at
least 15 days in advance, and continuing to use the service after they take
effect constitutes acceptance. If you do not agree, cancel before the new terms
apply and we will refund any unused prepaid period.</p>

<h2>18. Governing law and disputes</h2>
<p>These Terms are governed by Spanish law, without prejudice to the mandatory
consumer protection rules of your country of residence. Disputes are submitted
to the courts of Málaga, Spain — except that if you are a consumer, you keep
the right to sue and be sued in the courts of your own domicile, and you may
use the consumer mediation and arbitration bodies available in your country of
residence. Complaints route: write to <a href="mailto:info@openshorts.app">info@openshorts.app</a> first; most issues
are fixed in days.</p>
`

const TERMS_ES = `
<h2>1. Quiénes somos</h2>
<p>OpenShorts Cloud (openshorts.app) es un servicio prestado por TONVI TECH SL,
sociedad española con CIF B-19780394 y domicilio en Calle Puerta del Mar 18,
5ª planta, 29005 Málaga, España ("OpenShorts", "nosotros"). Puedes escribirnos
a <a href="mailto:info@openshorts.app">info@openshorts.app</a>. Estos Términos de Servicio ("Términos") regulan el uso
del servicio alojado en openshorts.app.</p>
<p>La edición de código abierto de OpenShorts que puedes descargar y ejecutar
en tu propio hardware se licencia por separado bajo la Licencia MIT y no está
cubierta por estos Términos: cuando te autoalojas, el operador de tu instancia
eres tú.</p>
<p>Estos Términos están también disponibles en <a href="/terms">inglés</a>.
Para consumidores residentes en España prevalece la versión en castellano en
caso de discrepancia.</p>

<h2>2. El servicio</h2>
<p>OpenShorts convierte vídeo largo en clips verticales cortos. Siguiendo tu
instrucción, el servicio ingiere un vídeo que subes o un vídeo situado en la
URL que indicas, lo transcribe, usa modelos de IA para seleccionar momentos, lo
reencuadra en vertical y, opcionalmente, añade subtítulos, rótulos, doblaje con
IA y publica el resultado en las cuentas de redes sociales que conectes. Existe
un plan gratuito con límites de uso y marca de agua, y suscripciones y recargas
de pago facturadas a través de Stripe.</p>

<h2>3. Requisitos y cuentas</h2>
<p>Debes tener al menos 16 años, o la edad de consentimiento digital de tu país
si es mayor. Eres responsable de todo lo que se haga con tu cuenta y tus claves
API. Mantén tus credenciales en secreto y avísanos en <a href="mailto:info@openshorts.app">info@openshorts.app</a> si
crees que tu cuenta está comprometida.</p>

<h2>4. Tu contenido sigue siendo tuyo</h2>
<p>Conservas la propiedad de los vídeos que envías. Al enviarlos nos concedes
una licencia mundial, no exclusiva y gratuita para alojarlos, almacenarlos,
reproducirlos, transcodificarlos, modificarlos y procesarlos, y para
mostrártelos, estrictamente en la medida necesaria para operar y proteger el
servicio. Esta licencia termina cuando el contenido se elimina de nuestros
sistemas según las reglas de conservación de la cláusula 13, salvo copias de
seguridad de duración limitada y lo que la ley nos obligue a conservar.</p>
<p><strong>Los clips son tuyos.</strong> Entre tú y nosotros, los clips que el
servicio genera a partir de tu contenido te pertenecen, incluido su uso
comercial, en la medida de nuestros derechos sobre ellos. En el plan gratuito
los clips llevan una marca de agua de OpenShorts; puedes usarlos libremente,
pero no puedes eliminarla ni ocultarla por otro medio que no sea pasar a un
plan de pago.</p>
<p>No usamos tu contenido para entrenar modelos de IA — ni nuestros ni de
terceros. Ver cláusula 7.</p>
<p>Con tu permiso expreso (por ejemplo, si nos dices que sí cuando te lo
pidamos) podremos citarte como cliente o mostrar tus clips públicos como
referencia. Nunca lo hacemos por defecto.</p>

<h2>5. Lo que garantizas sobre el contenido</h2>
<p>El servicio procesa contenido siguiendo tu instrucción — tanto archivos que
subes como vídeos que nos ordenas descargar desde una URL. Garantizas que, para
cada vídeo que envías por cualquiera de las dos vías:</p>
<ul>
<li>es tuyo, o cuentas con los derechos o permisos necesarios para
descargarlo, procesarlo, editarlo y volver a publicarlo;</li>
<li>tu uso a través del servicio no infringe derechos de autor, marcas,
derechos de imagen o publicidad de nadie, ni las condiciones de la plataforma
de origen;</li>
<li>no aparece la imagen o la voz de ninguna persona identificable sin la
autorización que exige la ley; y</li>
<li>si usas el doblaje con IA, cuentas con el consentimiento del titular de la
voz para generar voz sintética a partir de ella.</li>
</ul>
<p>Al enviar un trabajo marcas una declaración que lo confirma. Guardamos esa
declaración junto con la fecha, tu dirección IP y el identificador de tu
navegador, y podremos aportar ese registro si surge una disputa. Enviar
contenido ajeno sin permiso es lo que con más probabilidad termina en la
cancelación de una cuenta según la cláusula 16.</p>

<h2>6. Datos personales en tus vídeos — nuestro papel de encargado</h2>
<p>Los vídeos suelen contener datos personales de quienes aparecen en ellos.
Respecto de esos datos tú eres el responsable del tratamiento y nosotros
actuamos como tu encargado conforme al artículo 28 del RGPD: tratamos el
material solo según tus instrucciones documentadas (los trabajos que envías),
recurrimos únicamente a los subencargados listados en la
<a href="/privacidad">Política de Privacidad</a> (proveedores de IA,
almacenamiento y entrega), aplicamos medidas de seguridad apropiadas y
eliminamos el contenido según las reglas de la cláusula 13 o antes si nos lo
pides. Si tu empresa necesita un contrato de encargo de tratamiento firmado,
solicítalo en <a href="mailto:info@openshorts.app">info@openshorts.app</a>. Cuando autoalojas OpenShorts con tus
propias claves, tu contenido fluye directamente entre tu instancia y el
proveedor de IA bajo tu propio acuerdo con él. En OpenShorts Cloud eso no es
así ni siquiera si aportas tu propia clave de proveedor: la petición la hacen
nuestros servidores, es decir que el contenido pasa por ellos y esta cláusula
6 le aplica íntegramente.</p>

<h2>7. Procesamiento con IA — qué se ejecuta y dónde</h2>
<p>El servicio usa sistemas de IA y los nombramos: la transcripción se ejecuta
en nuestros propios servidores (modelos de la familia Whisper y NVIDIA
Parakeet); la selección de momentos, los títulos, las decisiones de encuadre y
las miniaturas generadas usan la API Gemini de Google; el doblaje de voz y las
voces de IA opcionales usan ElevenLabs; AI Shorts usa fal.ai (Flux para las
imágenes de actor, Kling y Hailuo para el presentador animado). Tu vídeo,
audio, transcripción, fotogramas muestreados y cualquier foto de actor que
aportes se envían a estos proveedores solo en la medida necesaria para
realizar el trabajo que pediste, bajo acuerdos que les prohíben usar los datos
de clientes de API de pago para entrenar sus modelos. El apartado 4 de la
<a href="/privacidad">Política de Privacidad</a> detalla qué recibe cada
uno.</p>
<p><strong>No entrenamos modelos de IA con tu contenido, ni lo vendemos ni lo
cedemos para entrenamiento.</strong></p>
<p>El resultado de la IA es probabilístico: las selecciones de clips,
transcripciones, traducciones, títulos y audio doblado pueden contener errores,
y entradas similares pueden producir salidas similares para distintos clientes,
por lo que los resultados no se garantizan únicos. Eres responsable de revisar
el resultado antes de usarlo o publicarlo. No puedes usar el servicio ni sus
resultados para desarrollar o entrenar un modelo o servicio competidor.</p>
<p>Los clips con doblaje de IA contienen voz sintética, y también los vídeos
hechos con un actor de IA y las miniaturas generadas por el modelo de imagen.
Marcamos esos ficheros como generados con IA en un formato legible por máquina
— una etiqueta de metadatos en el vídeo y un marcado XMP
<code>DigitalSourceType: trainedAlgorithmicMedia</code> en las imágenes
generadas —, que es lo que nos exige a nosotros como proveedor el artículo
50.2 del Reglamento (UE) 2024/1689 (Reglamento de IA); no debes eliminar esa
marca. Ten en cuenta que las plataformas recodifican habitualmente lo que se
sube y que una recodificación puede eliminarla. No quemamos una etiqueta
visible sobre la imagen. Al
publicar uno de esos clips, revelar a tu audiencia que es contenido generado
artificialmente es tu propia obligación como responsable del despliegue según
el artículo 50.4, y la etiqueta de contenido con IA que ofrece cada plataforma
(TikTok, Instagram, YouTube y las demás) es la forma normal de cumplirla.</p>

<h2>8. Uso aceptable</h2>
<p>No puedes usar el servicio para:</p>
<ul>
<li>procesar o distribuir contenido ilegal, difamatorio, acosador, de odio,
de explotación sexual de menores o que incite a la violencia;</li>
<li>infringir propiedad intelectual, derechos de imagen o privacidad,
incluido enviar URLs de contenido sobre el que no tienes derechos;</li>
<li>suplantar a nadie ni producir medios con la imagen o la voz de una persona
identificable sin autorización (deepfakes);</li>
<li>sondear, interrumpir o sobrecargar el servicio, hacer scraping, o acceder
por medios automatizados distintos de la API y el endpoint MCP documentados y
dentro de sus límites de uso;</li>
<li>hacer ingeniería inversa del servicio alojado o intentar extraer sus
modelos, prompts o componentes no públicos;</li>
<li>vender, transferir o compartir claves API, cuentas o cuota, o crear varias
cuentas para eludir los límites del plan gratuito;</li>
<li>manipular artificialmente métricas de interacción en las plataformas donde
publiques.</li>
</ul>
<p>Esta cláusula, junto con las cláusulas 5 y 9, constituye nuestra política de
moderación de contenidos a efectos del artículo 14 del Reglamento (UE)
2022/2065 (DSA): el contenido que la incumpla podrá retirarse y las cuentas
podrán limitarse, suspenderse o cancelarse, motivándolo ante el afectado.
Podemos bloquear el re-registro de la misma persona tras una cancelación por
abuso.</p>

<h2>9. Reclamaciones de derechos de autor y retirada de contenido</h2>
<p>El procedimiento completo, con lo que debe contener una notificación y cómo
impugnar una retirada, está en su propia página:
<a href="/reportar-contenido">Reportar Contenido Ilícito</a>.</p>
<p>Si crees que contenido procesado o alojado a través del servicio infringe
tus derechos, escribe a <a href="mailto:info@openshorts.app">info@openshorts.app</a> (nuestro punto de contacto para
usuarios y autoridades según los artículos 11 y 12 de la DSA) indicando: tu
nombre y datos de contacto, la obra de la que eres titular, la URL o el
identificador exacto del material infractor, una declaración de buena fe de que
el uso no está autorizado, y tu firma. Revisamos los avisos con diligencia,
retiramos o bloqueamos el contenido cuando el aviso está fundado e informamos
motivadamente al usuario afectado, que podrá responder con una contra-
notificación fundada. Cancelamos las cuentas de los infractores reincidentes.
Este procedimiento aplica el mecanismo de notificación y acción del artículo 16
de la DSA.</p>
<h3>Notificaciones bajo la DMCA estadounidense</h3>
<p>Para titulares de derechos en Estados Unidos, TONVI TECH SL ha designado al
siguiente agente para recibir notificaciones de presunta infracción conforme al
17 U.S.C. § 512(c)(2):</p>
<p>Copyright Agent, TONVI TECH SL, Calle Puerta del Mar 18, 5ª planta, 29005
Málaga, España. Email: <a href="mailto:info@upload-post.com">info@upload-post.com</a>.</p>
<p>Esa es la dirección que consta en el Copyright Office estadounidense para
esta empresa, que opera openshorts.app y upload-post.com. Las notificaciones
enviadas a <a href="mailto:info@openshorts.app">info@openshorts.app</a> también se atienden, pero el canal formal es la
dirección anterior.</p>
<p>La notificación debe hacerse por escrito e incluir: la firma física o
electrónica del titular o de quien esté autorizado a actuar en su nombre; la
identificación de la obra protegida presuntamente infringida; la identificación
del material que consideras infractor y dónde se encuentra en el servicio; tu
dirección, teléfono y email; una declaración de que crees de buena fe que el
uso no está autorizado por el titular, su agente o la ley; y una declaración,
bajo pena de perjurio, de que la información es exacta y de que eres el titular
o estás autorizado a actuar en su nombre.</p>
<p>Si retiramos tu material y crees que fue un error o una identificación
equivocada, puedes enviar una contra-notificación al mismo agente conforme al
17 U.S.C. § 512(g), con tu firma, la identificación del material y del lugar
donde aparecía, una declaración bajo pena de perjurio de que crees de buena fe
que se retiró por error, y tu sometimiento a la jurisdicción del tribunal
federal de distrito de tu domicilio o, si resides fuera de Estados Unidos, de
cualquier distrito en el que podamos ser emplazados. Trasladamos la
contra-notificación al reclamante y podemos restaurar el material pasados 10
días hábiles, salvo que nos comunique que ha iniciado acciones judiciales para
impedírtelo.</p>
<p>Conforme al 17 U.S.C. § 512(f), quien manifieste a sabiendas y de forma
sustancialmente falsa que un material infringe derechos, o que se retiró por
error, responde de los daños que esa manifestación cause.</p>

<h2>10. Publicación en tus cuentas sociales</h2>
<p>Si conectas cuentas sociales (a través de nuestro socio de publicación,
Upload-Post), nos autorizas expresamente a nosotros y a nuestro socio a
publicar contenido en esas cuentas siguiendo tu instrucción. Tú sigues siendo
el editor de todo lo publicado: respondes del contenido, de su programación y
de su conformidad con las condiciones de cada plataforma (YouTube, TikTok,
Instagram y las demás). Puedes desconectar tus cuentas en cualquier momento
desde el panel o desde los ajustes de la propia plataforma, lo que revoca
nuestro acceso.</p>

<h2>11. Planes, facturación y renovaciones</h2>
<p>Las suscripciones y recargas de pago se facturan a través de Stripe; nunca
vemos ni almacenamos tu número de tarjeta. Los precios indican en el proceso de
pago los impuestos aplicables. Las suscripciones se renuevan automáticamente al
final de cada periodo hasta que canceles.
Puedes cancelar en cualquier momento desde tu cuenta, con la misma facilidad
con la que te suscribiste; la cancelación surte efecto al final del periodo en
curso, que sigues disfrutando íntegro. La cuota no consumida no se acumula
salvo que el plan diga lo contrario. Te avisaremos por email con al menos 30
días de antelación antes de cualquier subida de precio, y podrás cancelar antes
de que se aplique.</p>

<h2>12. Derecho de desistimiento (consumidores UE)</h2>
<p>Si eres consumidor en la UE, tienes derecho a desistir del contrato a
distancia en un plazo de 14 días naturales sin necesidad de justificación. Al
suscribirte solicitas expresamente que el servicio comience de inmediato,
dentro del plazo de desistimiento. Conservas tu derecho durante esos 14 días,
pero si lo ejerces abonarás la parte proporcional del servicio ya prestado, y
el derecho se pierde una vez el servicio se ha ejecutado por completo
(artículos 98.8, 103.a y 108 del TRLGDCU, que transponen la Directiva
2011/83/UE). Las renovaciones automáticas de una suscripción en curso no abren
un nuevo plazo de desistimiento. Para desistir, escribe a <a href="mailto:info@openshorts.app">info@openshorts.app</a> o
usa el formulario modelo que incluimos en tu email de confirmación; los
reembolsos se emiten al medio de pago original en un máximo de 14 días.</p>
<p>Al margen de este derecho, nuestra <a href="/reembolsos">política de
reembolsos</a> devuelve íntegro tu último cargo si no has usado el servicio
desde que se cobró, que es el caso de la renovación que a nadie le sonó. Cada
una sirve para algo distinto: el derecho legal de arriba es el que te ayuda si
sí consumiste minutos, porque no exige que hayas dejado el servicio intacto.</p>

<h2>13. Conservación y eliminación del contenido</h2>
<p>Los clips generados en el plan gratuito se conservan 7 días y después se
eliminan de forma permanente; te avisamos por email antes de que ocurra.
Descarga lo que quieras conservar. En los planes de pago, tus proyectos y clips
permanecen almacenados mientras tu suscripción esté activa y se eliminan 7
días después de su fin, salvo que los borres antes. Eliminar tu cuenta elimina
tu contenido con el mismo calendario, sin perjuicio de copias de seguridad de
corta duración y de los registros que la ley nos obligue a conservar (como los
datos de facturación). No respondemos de la pérdida de contenido cuya
eliminación estos Términos anunciaban.</p>

<h2>14. Exclusión de garantías</h2>
<p>El servicio se presta "tal cual" y "según disponibilidad", sin garantía de
disponibilidad ininterrumpida, funcionamiento sin errores o idoneidad para un
fin concreto, en la medida en que la ley de tu residencia permita estas
exclusiones. Nada en estos Términos limita las garantías o derechos que la
normativa de consumo te reconoce con carácter irrenunciable.</p>

<h2>15. Responsabilidad</h2>
<p>En la máxima medida permitida por la ley, nuestra responsabilidad total
agregada por reclamaciones derivadas del servicio se limita a las cantidades
que nos hayas pagado en los 12 meses anteriores al hecho que origine la
reclamación, y no respondemos de daños indirectos, lucro cesante ni de la
pérdida de datos cuya eliminación anunciaba la cláusula 13. Esta limitación no
se aplica a los daños causados por dolo o negligencia grave, a los daños
personales, ni a ninguna responsabilidad que no pueda limitarse conforme al
Derecho español o de la UE.</p>

<h2>16. Indemnidad, suspensión y cancelación</h2>
<p>Si un tercero nos reclama por contenido que enviaste o nos ordenaste
descargar, o por tu incumplimiento de estos Términos, nos mantendrás indemnes —
incluidos nuestros administradores, empleados y proveedores — de los daños,
sanciones y costes razonables de defensa resultantes, salvo en la medida en que
la reclamación derive de nuestro propio incumplimiento. Si eres consumidor,
esta obligación se aplica únicamente a reclamaciones causadas por tu
incumplimiento de estos Términos o de la ley.</p>
<p>Podemos suspender o cancelar cuentas por incumplimiento grave. Si el
incumplimiento es subsanable te avisaremos primero y te daremos un plazo
razonable para corregirlo; si no lo es (contenido ilegal, reincidencia en
infracciones, abuso, fraude), podremos suspender de inmediato, motivándolo.
Tras la cancelación dispones de 7 días para descargar el contenido restante,
salvo que la ley nos obligue a retirarlo. Las cláusulas 4 a 9 y 14 a 18
sobreviven a la terminación.</p>

<h2>17. Cambios en el servicio y en estos Términos</h2>
<p>Mejoramos el servicio continuamente y podemos cambiar o retirar funciones.
Podemos modificar estos Términos; los cambios sustanciales se anunciarán por
email o en la aplicación con al menos 15 días de antelación, y seguir usando el
servicio tras su entrada en vigor supone su aceptación. Si no estás de acuerdo,
cancela antes de que se apliquen y te reembolsaremos el periodo prepagado no
disfrutado.</p>

<h2>18. Ley aplicable y disputas</h2>
<p>Estos Términos se rigen por la ley española, sin perjuicio de las normas
imperativas de protección de los consumidores de tu país de residencia. Las
disputas se someten a los juzgados y tribunales de Málaga, España — con la
salvedad de que, si eres consumidor, conservas el derecho a demandar y ser
demandado ante los tribunales de tu propio domicilio, y puedes acudir a los
sistemas de mediación y arbitraje de consumo de tu país de residencia. Vía de
reclamación: escribe primero a <a href="mailto:info@openshorts.app">info@openshorts.app</a>; la mayoría de los problemas
se resuelven en días.</p>
`

const PRIVACY_EN = `
<h2>1. Who is responsible</h2>
<p>The data controller is TONVI TECH SL (CIF B-19780394), Calle Puerta del Mar
18, 5th floor, 29005 Málaga, Spain — the company behind OpenShorts
(openshorts.app). For anything about your data, write to <a href="mailto:info@openshorts.app">info@openshorts.app</a>.
This policy is also available in <a href="/privacidad">Spanish</a>; for
residents of Spain, the Spanish version prevails in case of discrepancy.</p>

<h2>2. What we process, why, and on what legal basis</h2>
<h3>Your account</h3>
<p>Your email address (and, if you sign in with Google, your Google account
identifier), sign-up and last-login dates. We use passwordless magic links, so
we never store a password. Basis: performance of the contract (Art. 6.1.b
GDPR).</p>
<h3>Billing</h3>
<p>Payments run entirely on Stripe: we store your Stripe customer reference,
plan and subscription state, never your card number. Invoicing data is kept
because tax law requires it. Basis: contract (Art. 6.1.b) and legal obligation
(Art. 6.1.c).</p>
<h3>The videos you process</h3>
<p>The videos you upload or instruct us to fetch, their transcripts, and the
clips generated from them are processed to deliver exactly the job you
requested. Basis: contract (Art. 6.1.b). Where a video contains personal data
of other people (their image, their voice), you are the controller of that
data and we act as your processor on your instructions — see Section 6 of the
<a href="/terms">Terms of Service</a>.</p>
<h3>Your rights declaration</h3>
<p>Each time you submit a job you confirm you have the rights to the content.
We keep that declaration with its date, your browser identifier and your IP
address <em>truncated to its network</em> (the last part is discarded, so it
identifies a network and not a device), as evidence of diligence in case of
copyright or image-rights disputes. It is stored with the project it belongs
to, and is deleted with your account. Basis: legitimate interest in
establishing and defending legal claims (Art. 6.1.f).</p>
<h3>Service emails, and the one that is not</h3>
<p>We email you magic links, receipts, renewal reminders, clip-ready notices
and warnings before clips are deleted. These are part of the service, not
marketing. Basis: contract (Art. 6.1.b). One message is different: if you are
on the free plan and run out of minutes, we send a single email suggesting an
upgrade. That one is a commercial communication (Art. 21.2 of Spanish Law
34/2002, which permits it about our own similar products to our own
customers), it carries an unsubscribe link and a one-click unsubscribe header
in every send, and refusing it stops it permanently while leaving the service
messages untouched. We do not run a newsletter and we never sell or rent your
address.</p>
<h3>Product analytics</h3>
<p>We measure how the app is used with OpenPanel, running on an instance we
operate ourselves; the measurement script is served from this domain, not from
a third party. <strong>Nothing loads until you accept the "audience
measurement" category in the cookie banner</strong>, and rejecting it is one
click in the same banner — you can change your mind any time from the
"cookies" link in the footer. We send the account's internal identifier (a
random uuid) with these events so we can tell first-time from repeat use; we
do <em>not</em> send your email address. No advertising pixels, no cross-site
tracking, nothing shared with anyone. Basis: your consent (Art. 6.1.a GDPR and
Art. 22.2 of Law 34/2002); withdraw it at any time in the banner or by
emailing <a href="mailto:info@openshorts.app">info@openshorts.app</a>.</p>
<h3>Security and anti-abuse</h3>
<p>Server logs, IP-based rate limits and fraud signals, kept to protect the
service and its users. Basis: legitimate interest (Art. 6.1.f). Application
logs are kept in the container's rotating log for no more than 7 days and are
not aggregated elsewhere; they do not contain your email address.</p>
<p>We do not process special categories of data (Art. 9 GDPR) as part of the
service, we make no automated decisions with legal effects on you (Art. 22
GDPR), and <strong>we do not use your content or your data to train AI models
— ours or anyone else's — nor sell it.</strong></p>

<h2>3. Cookies and local storage</h2>
<p>We set no third-party cookies and load no third-party script on page load.
What the site stores falls into two groups.</p>
<p><strong>Strictly necessary — always on, no consent required.</strong> All of
it is first-party and lives in your browser's local storage, not in a cookie:
your session token (keeps you signed in), a short-lived media token (lets your
browser fetch your own clips and thumbnails, which are not public), your
interface preferences (last-used editor settings, tab, language), the
first-visit referrer we record once at sign-up, and — if you use
bring-your-own-key mode — your AI provider API keys. Those keys are
<em>obfuscated</em>, not encrypted: the value is scrambled so it is not
readable at a glance in developer tools, which is not the same as
cryptographic protection, and anyone with access to your browser profile could
recover them. They are sent only to perform your own jobs and are never stored
on our servers. Your consent record itself is also stored here.</p>
<p><strong>Audience measurement — always on, first-party only.</strong> OpenPanel
(see section 2) writes an anonymous visitor identifier. The script is served
from this domain; the instance is ours, the data is never shared and never used
cross-site, which under the AEPD/CNIL criteria exempts it from prior consent.
The consent banner is only shown to visitors in the EEA, the UK and Switzerland;
identifiers live no longer than 13 months and raw data no longer than 25
months. "Reject all" and "Accept all" sit side by side in the banner, and the
"cookies" link in the footer reopens it so you can withdraw consent as easily
as you gave it.</p>
<p><strong>Marketing.</strong> We use no advertising or remarketing trackers.
The category exists in the banner so that if we ever add one it starts
switched off.</p>

<h2>4. Who receives data (processors) and where</h2>
<p>Every provider we use, what each one receives, and where it is. The
machine-readable version of this list, with the transfer safeguard for each,
is kept in the public repository at
<a href="https://github.com/mutonby/openshorts/blob/main/docs/compliance/subprocessors.md">docs/compliance/subprocessors.md</a>
and changes are recorded there.</p>
<ul>
<li><strong>Google (Gemini API)</strong> — AI analysis to select moments,
layouts, titles and thumbnails. It receives your video's transcript and
sampled frames (which may show faces), the reference photo you pick for a
thumbnail, and — only when the "screencast" layout check runs — the source
video file itself, which we delete from Google's file store as soon as the
answer comes back rather than letting it expire there. Google's paid API terms
prohibit using customer data to train models. USA.</li>
<li><strong>ElevenLabs</strong> — AI dubbing and AI voices, only when you ask
for it. Receives the clip's audio, which means the voice of whoever is
speaking in it. USA.</li>
<li><strong>fal.ai</strong> — the image and video models behind AI Shorts
(Flux for actor images, Kling and Hailuo for the talking avatar). Receives the
actor photo you upload or generate — which may be a photograph of a real
person — and the audio to lip-sync it to. USA.</li>
<li><strong>Stripe</strong> — payments and invoicing. USA/EU.</li>
<li><strong>Cloudflare R2 and Amazon Web Services S3</strong> — storage of your
clips and project files. EU/USA regions.</li>
<li><strong>Cloudflare</strong> — DNS and CDN in front of the website, so it
sees the IP address and request metadata of every visitor. USA/global.</li>
<li><strong>Hetzner and Contabo</strong> — the servers the service runs on.
Germany.</li>
<li><strong>Download proxies (DataImpulse and static ISP proxies)</strong> —
when a video platform refuses our servers directly, the download is routed
through a commercial proxy. It sees the URL of the video and carries its
bytes; it receives no account data of yours.</li>
<li><strong>Upload-Post</strong> — publishing to social accounts you connect,
only on your instruction, and the custodian of the platform tokens those
connections produce. Operated by TONVI TECH SL under this same policy.</li>
<li><strong>Aikount</strong> — our own invoicing system, which issues the
Spanish legal invoice for your payment and therefore receives your billing
name, address and email from Stripe. Operated by TONVI TECH SL, servers in
Germany.</li>
<li><strong>OpenPanel</strong> — first-party audience measurement, exempt
from prior consent (section 3). The instance is ours and runs on our own servers in Germany;
the script is served from this domain.</li>
<li><strong>Namecheap Private Email</strong> — delivery of service emails.
USA.</li>
<li><strong>Google (sign in with Google)</strong> — only if you choose it, and
only the "openid email profile" scope, so we learn your address and Google
account id and nothing else. USA.</li>
</ul>
<p>Two things we deliberately do <em>not</em> send anywhere: our internal
operational alerts identify you by the first characters of your account
identifier, never by your email address or the title of your video; and the
face detection that decides where to crop runs on our own servers and
produces no biometric template.</p>
<p>Where a provider processes data outside the European Economic Area, the
transfer relies on the EU–US Data Privacy Framework where the provider is
certified and, in any case, on the European Commission's Standard Contractual
Clauses (Art. 46.2.c GDPR) as a fallback, together with the provider's
technical safeguards. We prefer EU regions where the provider offers them.</p>

<h2>5. How long we keep things</h2>
<ul>
<li>Free-plan clips: 7 days, then permanently deleted (we warn you by email
first).</li>
<li>Paid-plan projects and clips: while your subscription is active, plus 7
days.</li>
<li>Account data: while your account exists; after deletion, only what is
needed to meet legal obligations or resolve disputes.</li>
<li>Invoicing data: 6 years (Spanish commercial law).</li>
<li>Rights declarations: up to 5 years with the project they belong to,
matching the statute of limitations for civil claims, and deleted with your
account if you delete it sooner.</li>
<li>Working files on the processing server (the source download, the working
copies): 1 hour after the job finishes. Uploads waiting for a job: 6 hours.</li>
<li>Generated thumbnails and the video frames sampled to make them: 24 hours
after you last downloaded one.</li>
<li>Photos you upload to drive an AI actor: deleted with the job that used
them, within the hour.</li>
<li>Application logs: 7 days in the container's rotating log.</li>
<li>Encrypted database backups: 30 days, after which a deletion you made is
also gone from every copy.</li>
<li>Analytics identifiers: 13 months; raw analytics data: 25 months.</li>
</ul>
<p>The full, per-item version of this table, written against the actual code
that enforces it, is in the public repository at
<a href="https://github.com/mutonby/openshorts/blob/main/docs/compliance/retencion.md">docs/compliance/retencion.md</a>.</p>

<h2>6. Your rights</h2>
<p>You can ask us for access, rectification, erasure, restriction, portability
of your data, and object to processing based on legitimate interest — email
<a href="mailto:info@openshorts.app">info@openshorts.app</a> or <a href="mailto:info@openshorts.app">info@openshorts.app</a> from your account address; we answer
within one month. Two of these you can exercise yourself, right now, without
asking anyone: <strong>Account → Download my data</strong> gives you a JSON
file with everything our database holds about you plus a download link for
every clip in your library (Art. 20), and <strong>Account → Delete account</strong>
erases it (Art. 17). You do not need to ask us to erase your account: <strong>Account →
Delete account</strong> in the dashboard does it yourself, immediately and
permanently, taking your projects, clips, transcripts, API keys and social
connections with it and cancelling any active subscription. Two things survive
it, both listed in section 5: your invoices, which tax law requires us to keep,
and a record that the deletion happened, which identifies you by a one-way hash
of your email address rather than by the address itself.
If you believe we are mishandling your data, you can complain to the Spanish
supervisory authority (AEPD, aepd.es) or to the authority of your own EU
country.</p>

<h2>7. California residents (CCPA/CPRA)</h2>
<p>If you live in California, this section is your notice at collection. In the
last twelve months we have collected these categories of personal information:
identifiers (the email address you sign in with); commercial information (your
plan, your minute balance and the Stripe reference for your payments, never
your card number); internet activity (first-party, self-hosted analytics about
how the app is used); and audio and visual information (the videos you upload
or link, their audio, the clips we produce and their transcripts). We collect
them to run the service you asked for, to bill it, to keep it secure and to
meet our legal obligations, as detailed in section 2. The sources are you and,
where you paste a link, the platform you took the video from.</p>
<p><strong>We do not sell your personal information, and we do not share it for
cross-context behavioural advertising.</strong> We have not done so in the
preceding twelve months, and we do not do it with the data of anyone under 16.
We disclose data to service providers only, for the purposes and to the
recipients listed in section 4, under contracts that stop them using it for
anything else.</p>
<p>Our reframing detects faces and subjects inside the picture so it knows where
to crop. That happens on our own servers, is never used to recognise or
identify anyone, and produces no faceprint or other identifier that could match
a person across videos. We do not use sensitive personal information to infer
characteristics about you, so the right to limit its use does not arise.</p>
<p>You have the right to know what we collect and to receive a copy of it, to
have it corrected, to have it deleted, and not to be discriminated against for
exercising any of these rights: we do not offer a worse service, a higher price
or fewer features to anyone who does. Exercise them by emailing
<a href="mailto:info@openshorts.app">info@openshorts.app</a> from your account address, or delete everything yourself
with <strong>Account → Delete account</strong>. We verify a request by the
control you have over the account email, which is the same credential you sign
in with, and we answer within 45 days. An authorised agent may act for you if
they provide your written permission.</p>

<h2>8. Age</h2>
<p>The service is not directed at children. You must be at least 16, or the
age of digital consent in your country if higher, to create an account, and
you are asked to confirm that when you sign up. If you believe a minor has
created an account, write to <a href="mailto:info@openshorts.app">info@openshorts.app</a> and we will delete it.</p>

<h2>8 bis. AI-generated content</h2>
<p>Dubbing, AI Shorts and AI thumbnails produce synthetic content. We mark
what we generate in a machine-readable way — a metadata tag on video files and
an XMP <code>DigitalSourceType: trainedAlgorithmicMedia</code> marking on
generated images — as required by Art. 50(2) of Regulation (EU) 2024/1689 (the
AI Act). That marking does not always survive a platform re-encoding your
upload, so when you publish synthetic content you are the one who must turn on
the platform's own AI-content label (Art. 50(4)); the app reminds you at the
moment you post. Cloning a voice or animating a photograph of a real person
without their agreement is prohibited by the
<a href="/terms">Terms of Service</a>, and both features ask you to confirm
you have that agreement before they run.</p>

<h2>9. Changes</h2>
<p>If we change this policy in any meaningful way we will tell you by email or
in-app before the change takes effect, and the date on this page always
reflects the current version.</p>
`

const PRIVACY_ES = `
<h2>1. Responsable del tratamiento</h2>
<p>El responsable es TONVI TECH SL (CIF B-19780394), Calle Puerta del Mar 18,
5ª planta, 29005 Málaga, España — la empresa detrás de OpenShorts
(openshorts.app). Para cualquier cuestión sobre tus datos, escribe a
<a href="mailto:info@openshorts.app">info@openshorts.app</a>. Esta política está también disponible en
<a href="/privacy">inglés</a>; para residentes en España prevalece la versión
en castellano en caso de discrepancia.</p>

<h2>2. Qué tratamos, para qué y con qué base jurídica</h2>
<h3>Tu cuenta</h3>
<p>Tu dirección de email (y, si inicias sesión con Google, tu identificador de
cuenta de Google), fechas de alta y de último acceso. Usamos enlaces mágicos
sin contraseña, así que nunca almacenamos una contraseña. Base: ejecución del
contrato (art. 6.1.b RGPD).</p>
<h3>Facturación</h3>
<p>Los pagos se realizan íntegramente en Stripe: guardamos tu referencia de
cliente de Stripe, el plan y el estado de la suscripción, nunca tu número de
tarjeta. Los datos de facturación se conservan porque la normativa fiscal lo
exige. Base: contrato (art. 6.1.b) y obligación legal (art. 6.1.c).</p>
<h3>Los vídeos que procesas</h3>
<p>Los vídeos que subes o nos ordenas descargar, sus transcripciones y los
clips generados a partir de ellos se tratan para prestarte exactamente el
trabajo que solicitaste. Base: contrato (art. 6.1.b). Cuando un vídeo contiene
datos personales de otras personas (su imagen, su voz), el responsable de esos
datos eres tú y nosotros actuamos como tu encargado siguiendo tus
instrucciones — ver cláusula 6 de los <a href="/terminos">Términos de
Servicio</a>.</p>
<h3>Tu declaración de derechos</h3>
<p>Cada vez que envías un trabajo confirmas que tienes derechos sobre el
contenido. Conservamos esa declaración con su fecha, el identificador de tu
navegador y tu dirección IP <em>truncada a su red</em> (descartamos la parte
final, de modo que identifica una red y no un dispositivo), como prueba de
diligencia ante posibles disputas de derechos de autor o de imagen. Se guarda
junto al proyecto al que pertenece y se elimina con tu cuenta. Base: interés
legítimo en la formulación y defensa de reclamaciones (art. 6.1.f).</p>
<h3>Emails de servicio, y el que no lo es</h3>
<p>Te enviamos enlaces de acceso, recibos, avisos de renovación, avisos de
clips listos y avisos previos a la eliminación de clips. Forman parte del
servicio, no son marketing. Base: contrato (art. 6.1.b). Hay un mensaje
distinto: si estás en el plan gratuito y agotas tus minutos, te enviamos un
único email sugiriendo mejorar de plan. Ese sí es una comunicación comercial
(art. 21.2 de la Ley 34/2002, que la permite sobre productos propios similares
a clientes propios), lleva enlace de baja y cabecera de baja en un clic en
todos sus envíos, y darte de baja lo detiene de forma permanente sin afectar a
los mensajes de servicio. No tenemos boletín y jamás vendemos ni cedemos tu
dirección.</p>
<h3>Analítica de producto</h3>
<p>Medimos el uso de la aplicación con OpenPanel, en una instancia que
operamos nosotros mismos; el script de medición se sirve desde este dominio y
no desde un tercero. <strong>No se carga nada hasta que aceptas la categoría
"medición de audiencia" en el banner de cookies</strong>, y rechazarla es un
solo clic en ese mismo banner — puedes cambiar de opinión cuando quieras desde
el enlace "cookies" del pie de página. Con esos eventos enviamos el
identificador interno de la cuenta (un uuid aleatorio) para distinguir el
primer uso del uso recurrente; <em>no</em> enviamos tu dirección de email. Sin
píxeles publicitarios, sin seguimiento entre sitios, sin compartir nada con
nadie. Base: tu consentimiento (art. 6.1.a RGPD y art. 22.2 de la Ley
34/2002); puedes retirarlo cuando quieras en el banner o escribiendo a
<a href="mailto:info@openshorts.app">info@openshorts.app</a>.</p>
<h3>Seguridad y antiabuso</h3>
<p>Registros del servidor, límites de uso por IP y señales de fraude,
conservados para proteger el servicio y a sus usuarios. Base: interés legítimo
(art. 6.1.f). Los registros de aplicación se conservan como máximo 7 días en
el log rotatorio del contenedor, no se agregan en ningún otro sitio y no
contienen tu dirección de email.</p>
<p>No tratamos categorías especiales de datos (art. 9 RGPD) como parte del
servicio, no tomamos decisiones automatizadas con efectos jurídicos sobre ti
(art. 22 RGPD) y <strong>no usamos tu contenido ni tus datos para entrenar
modelos de IA — ni nuestros ni de terceros — ni los vendemos.</strong></p>

<h2>3. Cookies y almacenamiento local</h2>
<p>No instalamos cookies de terceros ni cargamos ningún script de terceros al
abrir la página. Lo que el sitio guarda se divide en dos grupos.</p>
<p><strong>Estrictamente necesario — siempre activo, sin consentimiento.</strong>
Todo es de primera parte y vive en el almacenamiento local de tu navegador, no
en una cookie: tu token de sesión (te mantiene identificado), un token de
medios de vida corta (permite que tu navegador descargue tus propios clips y
miniaturas, que no son públicos), tus preferencias de interfaz (últimos
ajustes del editor, pestaña, idioma), el referente de la primera visita que
registramos una sola vez al darte de alta y — si usas el modo con claves
propias — tus claves API de proveedores de IA. Esas claves están
<em>ofuscadas</em>, no cifradas: el valor se transforma para que no sea
legible de un vistazo en las herramientas de desarrollo, lo cual no es
protección criptográfica, y quien tenga acceso a tu perfil de navegador podría
recuperarlas. Solo se envían para ejecutar tus propios trabajos y nunca se
almacenan en nuestros servidores. El registro de tu decisión sobre cookies
también se guarda aquí.</p>
<p><strong>Medición de audiencia — siempre activa, solo de primera parte.</strong>
OpenPanel (ver apartado 2) escribe un identificador anónimo de visitante. El
script se sirve desde este dominio; la instancia es nuestra, los datos no se
ceden ni se usan entre sitios, lo que según el criterio de la AEPD y la CNIL la
exime de consentimiento previo. El banner solo se muestra a visitantes del EEE,
Reino Unido y Suiza; los identificadores no viven más de 13 meses y los datos brutos
no más de 25. "Rechazar todo" y "Aceptar todo" están uno al lado del otro en
el banner, y el enlace "cookies" del pie de página lo reabre para que retirar
el consentimiento cueste lo mismo que darlo.</p>
<p><strong>Marketing.</strong> No usamos rastreadores publicitarios ni de
remarketing. La categoría existe en el banner para que, si algún día
añadiéramos alguno, empiece desactivado.</p>

<h2>4. Quién recibe datos (encargados) y dónde</h2>
<p>Todos los proveedores que usamos, qué recibe cada uno y dónde está. La
versión detallada de esta lista, con la garantía de transferencia de cada uno,
se mantiene en el repositorio público en
<a href="https://github.com/mutonby/openshorts/blob/main/docs/compliance/subprocessors.md">docs/compliance/subprocessors.md</a>,
donde también se registran los cambios.</p>
<ul>
<li><strong>Google (API Gemini)</strong> — análisis con IA para seleccionar
momentos, encuadres, títulos y miniaturas. Recibe la transcripción de tu vídeo
y fotogramas muestreados (que pueden mostrar caras), la foto de referencia que
elijas para una miniatura y — solo cuando se ejecuta la comprobación de
encuadre "screencast" — el propio archivo de vídeo de origen, que borramos del
almacén de archivos de Google en cuanto llega la respuesta en lugar de dejar
que caduque allí. Las condiciones de la API de pago de Google prohíben usar
los datos de clientes para entrenar modelos. EE. UU.</li>
<li><strong>ElevenLabs</strong> — doblaje y voces con IA, solo cuando lo
solicitas. Recibe el audio del clip, es decir la voz de quien habla en él.
EE. UU.</li>
<li><strong>fal.ai</strong> — los modelos de imagen y vídeo que hay detrás de
AI Shorts (Flux para las imágenes de actor, Kling y Hailuo para el avatar que
habla). Recibe la foto de actor que subes o generas — que puede ser la
fotografía de una persona real — y el audio con el que se sincronizan sus
labios. EE. UU.</li>
<li><strong>Stripe</strong> — pagos y facturación. EE. UU./UE.</li>
<li><strong>Cloudflare R2 y Amazon Web Services S3</strong> — almacenamiento de
tus clips y archivos de proyecto. Regiones UE/EE. UU.</li>
<li><strong>Cloudflare</strong> — DNS y CDN delante del sitio web, por lo que
ve la dirección IP y los metadatos de petición de cada visitante.
EE. UU./global.</li>
<li><strong>Hetzner y Contabo</strong> — los servidores donde se ejecuta el
servicio. Alemania.</li>
<li><strong>Proxies de descarga (DataImpulse y proxies ISP estáticos)</strong> —
cuando una plataforma de vídeo rechaza a nuestros servidores directamente, la
descarga se enruta por un proxy comercial. Ve la URL del vídeo y transporta
sus bytes; no recibe ningún dato de tu cuenta.</li>
<li><strong>Upload-Post</strong> — publicación en las cuentas sociales que
conectes, solo siguiendo tu instrucción, y custodio de los tokens de
plataforma que esas conexiones generan. Operado por TONVI TECH SL bajo esta
misma política.</li>
<li><strong>Aikount</strong> — nuestro propio sistema de facturación, que
emite la factura legal española de tu pago y por tanto recibe de Stripe tu
nombre fiscal, dirección y email. Operado por TONVI TECH SL, servidores en
Alemania.</li>
<li><strong>OpenPanel</strong> — medición de audiencia propia, exenta de
consentimiento previo (apartado 3). La instancia es nuestra y funciona en nuestros propios
servidores en Alemania; el script se sirve desde este dominio.</li>
<li><strong>Namecheap Private Email</strong> — entrega de los emails de
servicio. EE. UU.</li>
<li><strong>Google (iniciar sesión con Google)</strong> — solo si lo eliges, y
solo con el ámbito "openid email profile", de modo que conocemos tu dirección
y tu identificador de cuenta de Google y nada más. EE. UU.</li>
</ul>
<p>Dos cosas que deliberadamente <em>no</em> enviamos a ningún sitio: nuestras
alertas operativas internas te identifican por los primeros caracteres del
identificador de tu cuenta, nunca por tu email ni por el título de tu vídeo; y
la detección de caras que decide dónde recortar se ejecuta en nuestros propios
servidores y no genera ninguna plantilla biométrica.</p>
<p>Cuando un proveedor trata datos fuera del Espacio Económico Europeo, la
transferencia se ampara en el Marco de Privacidad de Datos UE–EE. UU. si el
proveedor está certificado y, en todo caso, en las Cláusulas Contractuales
Tipo de la Comisión Europea (art. 46.2.c RGPD) como salvaguarda subsidiaria,
junto con las medidas técnicas del proveedor. Preferimos regiones de la UE
cuando el proveedor las ofrece.</p>

<h2>5. Cuánto tiempo conservamos cada cosa</h2>
<ul>
<li>Clips del plan gratuito: 7 días; después se eliminan de forma permanente
(te avisamos antes por email).</li>
<li>Proyectos y clips de planes de pago: mientras tu suscripción esté activa,
más 7 días.</li>
<li>Datos de cuenta: mientras exista tu cuenta; tras su eliminación, solo lo
necesario para cumplir obligaciones legales o resolver disputas.</li>
<li>Datos de facturación: 6 años (Código de Comercio).</li>
<li>Declaraciones de derechos: hasta 5 años junto al proyecto al que
pertenecen, en línea con el plazo de prescripción de las acciones civiles, y
se eliminan con tu cuenta si la borras antes.</li>
<li>Archivos de trabajo en el servidor de procesado (la descarga de origen,
las copias intermedias): 1 hora tras finalizar el trabajo. Subidas pendientes
de trabajo: 6 horas.</li>
<li>Miniaturas generadas y fotogramas muestreados para crearlas: 24 horas
desde la última vez que descargaste una.</li>
<li>Fotos que subes para animar un actor de IA: se eliminan con el trabajo que
las usó, dentro de la hora siguiente.</li>
<li>Registros de aplicación: 7 días en el log rotatorio del contenedor.</li>
<li>Copias de seguridad cifradas de la base de datos: 30 días, pasados los
cuales una eliminación que hayas hecho desaparece también de toda copia.</li>
<li>Identificadores de analítica: 13 meses; datos brutos de analítica: 25
meses.</li>
</ul>
<p>La versión completa y detallada de esta tabla, escrita contra el código que
realmente la aplica, está en el repositorio público en
<a href="https://github.com/mutonby/openshorts/blob/main/docs/compliance/retencion.md">docs/compliance/retencion.md</a>.</p>

<h2>6. Tus derechos</h2>
<p>Puedes pedirnos acceso, rectificación, supresión, limitación y portabilidad
de tus datos, y oponerte a los tratamientos basados en interés legítimo — basta
un email a <a href="mailto:info@openshorts.app">info@openshorts.app</a> o a <a href="mailto:info@openshorts.app">info@openshorts.app</a> desde la dirección de tu
cuenta; respondemos en el plazo de un mes. Dos de ellos puedes ejercerlos tú
mismo, ahora y sin pedir permiso a nadie: <strong>Cuenta → Descargar mis
datos</strong> te entrega un fichero JSON con todo lo que nuestra base de datos
guarda sobre ti más un enlace de descarga para cada clip de tu biblioteca
(art. 20), y <strong>Cuenta → Eliminar cuenta</strong> lo borra (art. 17). Para la supresión no necesitas pedírnosla:
<strong>Cuenta → Eliminar cuenta</strong> en el panel la ejecuta tú mismo, de
forma inmediata y permanente, y se lleva por delante tus proyectos, clips,
transcripciones, claves de API y conexiones con redes sociales, además de
cancelar cualquier suscripción activa. Solo sobreviven dos cosas, ambas
recogidas en el apartado 5: tus facturas, que la normativa fiscal nos obliga a
conservar, y un registro de que la eliminación se produjo, que te identifica
mediante un hash irreversible de tu dirección de email y no mediante la
dirección en sí. Si crees que tratamos mal tus datos, puedes reclamar
ante la Agencia Española de Protección de Datos (AEPD, aepd.es) o ante la
autoridad de tu país de la UE.</p>

<h2>7. Residentes en California (CCPA/CPRA)</h2>
<p>Si resides en California, este apartado es tu aviso en el momento de la
recogida. En los últimos doce meses hemos recogido estas categorías de
información personal: identificadores (la dirección de email con la que
accedes); información comercial (tu plan, tu saldo de minutos y la referencia
de Stripe de tus pagos, nunca el número de tu tarjeta); actividad en internet
(analítica propia y autoalojada sobre el uso de la aplicación); e información
audiovisual (los vídeos que subes o enlazas, su audio, los clips que
generamos y sus transcripciones). Las recogemos para prestarte el servicio que
has pedido, facturarlo, mantenerlo seguro y cumplir nuestras obligaciones
legales, según el detalle del apartado 2. Las fuentes eres tú y, cuando pegas
un enlace, la plataforma de la que tomas el vídeo.</p>
<p><strong>No vendemos tu información personal ni la compartimos para
publicidad conductual entre contextos.</strong> No lo hemos hecho en los doce
meses anteriores y no lo hacemos con datos de menores de 16 años. Solo la
comunicamos a proveedores de servicios, para los fines y a los destinatarios
del apartado 4, con contratos que les impiden usarla para otra cosa.</p>
<p>Nuestro reencuadre detecta caras y sujetos dentro de la imagen para saber
por dónde recortar. Ocurre en nuestros propios servidores, nunca se usa para
reconocer ni identificar a nadie, y no genera una plantilla facial ni ningún
identificador que permita relacionar a una persona entre vídeos. No usamos
información personal sensible para inferir características sobre ti, así que
el derecho a limitar su uso no llega a nacer.</p>
<p>Tienes derecho a saber qué recogemos y a recibir una copia, a que se
rectifique, a que se elimine y a no sufrir discriminación por ejercer
cualquiera de estos derechos: no damos peor servicio, ni precio más alto, ni
menos funciones a quien los ejerce. Se ejercen escribiendo a
<a href="mailto:info@openshorts.app">info@openshorts.app</a> desde la dirección de tu cuenta, o eliminándolo todo tú
mismo desde <strong>Cuenta → Eliminar cuenta</strong>. Verificamos la solicitud
por el control que tienes sobre el email de la cuenta, que es la misma
credencial con la que accedes, y respondemos en 45 días. Un agente autorizado
puede actuar por ti si aporta tu permiso por escrito.</p>

<h2>8. Edad</h2>
<p>El servicio no está dirigido a menores. Para crear una cuenta debes tener al
menos 16 años, o la edad de consentimiento digital de tu país si es mayor, y
se te pide confirmarlo al registrarte. Si crees que un menor ha creado una
cuenta, escribe a <a href="mailto:info@openshorts.app">info@openshorts.app</a> y la eliminaremos.</p>

<h2>8 bis. Contenido generado por IA</h2>
<p>El doblaje, AI Shorts y las miniaturas con IA producen contenido sintético.
Marcamos lo que generamos de forma legible por máquina — una etiqueta de
metadatos en los archivos de vídeo y un marcado XMP
<code>DigitalSourceType: trainedAlgorithmicMedia</code> en las imágenes
generadas — conforme al art. 50.2 del Reglamento (UE) 2024/1689 (Reglamento de
IA). Ese marcado no siempre sobrevive a la recodificación que hacen las
plataformas al subir un archivo, así que cuando publiques contenido sintético
eres tú quien debe activar la etiqueta de contenido de IA de la propia
plataforma (art. 50.4); la aplicación te lo recuerda en el momento de
publicar. Clonar una voz o animar la fotografía de una persona real sin su
consentimiento está prohibido por los <a href="/terminos">Términos de
Servicio</a>, y ambas funciones te piden confirmar que cuentas con ese
consentimiento antes de ejecutarse.</p>

<h2>9. Cambios</h2>
<p>Si cambiamos esta política de forma relevante te lo comunicaremos por email
o en la aplicación antes de que el cambio surta efecto, y la fecha de esta
página refleja siempre la versión vigente.</p>
`

const LEGAL_EN = `
<h2>Site owner</h2>
<p>In compliance with Article 10 of Spanish Law 34/2002 (LSSI-CE), the owner of
openshorts.app is:</p>
<ul>
<li><strong>Company:</strong> TONVI TECH SL</li>
<li><strong>Tax ID (CIF):</strong> B-19780394</li>
<li><strong>Registered address:</strong> Calle Puerta del Mar 18, 5th floor,
29005 Málaga, Spain</li>
<li><strong>Contact:</strong> <a href="mailto:info@openshorts.app">info@openshorts.app</a> (general and legal notices)
and <a href="mailto:info@openshorts.app">info@openshorts.app</a> (data protection)</li>
</ul>

<h2>Purpose of the site</h2>
<p>openshorts.app provides OpenShorts Cloud, an AI service that turns
long-form video into short vertical clips, under the <a href="/terms">Terms of
Service</a>. Personal data is handled as described in the
<a href="/privacy">Privacy Policy</a>. Prices shown on the pricing page
indicate whether taxes are included; where they are not, the applicable tax is
displayed at checkout before you pay.</p>

<h2>Intellectual property</h2>
<p>The OpenShorts source code is available under the MIT License in its public
repository. The openshorts.app site, its branding and its content are property
of TONVI TECH SL or its licensors. Videos and clips processed through the
service belong to their respective owners.</p>

<h2>Responsibility</h2>
<p>TONVI TECH SL acts as a hosting and processing provider for content
submitted by its users, within the liability regime of Articles 14–17 LSSI-CE
and Regulation (EU) 2022/2065 (DSA). Notices about allegedly unlawful content
can be sent to <a href="mailto:info@openshorts.app">info@openshorts.app</a>, which is our single point of contact for
users and authorities; see Section 9 of the <a href="/terms">Terms of
Service</a> for the procedure.</p>

<h2>Applicable law</h2>
<p>This legal notice is governed by Spanish law. It is also available in
<a href="/aviso-legal">Spanish</a>; for residents of Spain, the Spanish
version prevails.</p>
`

const LEGAL_ES = `
<h2>Titular del sitio</h2>
<p>En cumplimiento del artículo 10 de la Ley 34/2002 (LSSI-CE), el titular de
openshorts.app es:</p>
<ul>
<li><strong>Denominación social:</strong> TONVI TECH SL</li>
<li><strong>CIF:</strong> B-19780394</li>
<li><strong>Domicilio:</strong> Calle Puerta del Mar 18, 5ª planta, 29005
Málaga, España</li>
<li><strong>Contacto:</strong> <a href="mailto:info@openshorts.app">info@openshorts.app</a> (general y avisos legales)
y <a href="mailto:info@openshorts.app">info@openshorts.app</a> (protección de datos)</li>
</ul>

<h2>Objeto del sitio</h2>
<p>openshorts.app presta OpenShorts Cloud, un servicio de IA que convierte
vídeo largo en clips verticales cortos, conforme a los
<a href="/terminos">Términos de Servicio</a>. Los datos personales se tratan
según la <a href="/privacidad">Política de Privacidad</a>. Los precios de la
página de precios indican si incluyen impuestos; cuando no los incluyen, el
impuesto aplicable se muestra en el proceso de pago antes de pagar.</p>

<h2>Propiedad intelectual</h2>
<p>El código fuente de OpenShorts está disponible bajo Licencia MIT en su
repositorio público. El sitio openshorts.app, su marca y sus contenidos son
propiedad de TONVI TECH SL o de sus licenciantes. Los vídeos y clips
procesados a través del servicio pertenecen a sus respectivos titulares.</p>

<h2>Responsabilidad</h2>
<p>TONVI TECH SL actúa como prestador de servicios de alojamiento y
procesamiento del contenido remitido por sus usuarios, dentro del régimen de
responsabilidad de los artículos 14 a 17 de la LSSI-CE y del Reglamento (UE)
2022/2065 (DSA). Los avisos sobre contenido presuntamente ilícito pueden
dirigirse a <a href="mailto:info@openshorts.app">info@openshorts.app</a>, nuestro punto único de contacto para usuarios
y autoridades; el procedimiento se describe en la cláusula 9 de los
<a href="/terminos">Términos de Servicio</a>.</p>

<h2>Ley aplicable</h2>
<p>Este aviso legal se rige por la ley española. Está también disponible en
<a href="/legal-notice">inglés</a>; para residentes en España prevalece la
versión en castellano.</p>
`

const REPORT_EN = `
<h2>What this page is for</h2>
<p>This is the notice-and-action mechanism of Article 16 of Regulation (EU)
2022/2065 (the Digital Services Act). Use it to tell us that content processed,
stored or transmitted through openshorts.app is illegal, including content that
infringes copyright, trademarks, image rights or privacy. You do not need an
account and you do not need a lawyer.</p>
<p>If your complaint is about your own account, your billing or your data, this
is the wrong page: write to <a href="mailto:info@openshorts.app">info@openshorts.app</a> and say what you need.</p>

<h2>How to send a notice</h2>
<p>Email <strong><a href="mailto:info@openshorts.app">info@openshorts.app</a></strong> with the subject line
<strong>Illegal content report</strong>, in English or Spanish. So that we can
act on it without having to come back to you, the notice needs:</p>
<ul>
<li>the exact URL or identifier of the material, precise enough for us to find
it without searching;</li>
<li>an explanation of why you consider it illegal, and under which law;</li>
<li>your name and email address, unless the report concerns child sexual abuse
material or offences against sexual freedom, which may be sent anonymously;</li>
<li>a statement that you believe in good faith that the information in the
notice is accurate and complete.</li>
</ul>
<p>If you are the rightsholder or act for them, say which of the two you are.
A notice that contains all of the above gives us actual knowledge of the
content, which is what triggers our obligation to act.</p>

<h2>What happens next</h2>
<p>We confirm receipt of your notice without undue delay. We then review it
diligently, without arbitrary decisions, and with human judgement rather than
an automated rule. When the notice is substantiated we remove or disable access
to the material, and we tell you what we decided and why. If we decide not to
act, we tell you that too, and why.</p>
<p>Notices that are manifestly unfounded, or that come from someone who sends
them repeatedly in bad faith, get their processing suspended after we warn the
sender. Accounts that repeatedly upload infringing material are terminated.</p>

<h2>If your content was removed</h2>
<p>When we remove your content, restrict it, or suspend your account on these
grounds, you get a statement of reasons under Article 17 DSA: what we removed,
the facts we relied on, the legal or contractual ground, whether an automated
system was involved, and how to challenge the decision.</p>
<p>To challenge it, reply to that message or write to <a href="mailto:info@openshorts.app">info@openshorts.app</a>. A
person reviews it, not a filter. You also keep two routes that do not depend on
us at all: an out-of-court dispute settlement body certified under Article 21
DSA, and the courts. Nothing on this page limits either.</p>

<h2>Copyright notices from the United States</h2>
<p>The procedure above covers copyright complaints wherever you are. If you
prefer to send a formal notification under the US Digital Millennium Copyright
Act, section 9 of our <a href="/terms">Terms of Service</a> names our designated
agent and lists what section 512(c)(3) requires the notification to contain.</p>

<h2>Point of contact</h2>
<p><a href="mailto:info@openshorts.app">info@openshorts.app</a> is our single point of contact for users (Article 12
DSA) and for Member State authorities, the Commission and the European Board
for Digital Services (Article 11 DSA). We accept communications in Spanish and
in English. The service is operated by TONVI TECH SL, CIF B-19780394, Calle
Puerta del Mar 18, 5th floor, 29005 Málaga, Spain.</p>
`

const REPORT_ES = `
<h2>Para qué sirve esta página</h2>
<p>Este es el mecanismo de notificación y acción del artículo 16 del Reglamento
(UE) 2022/2065 (Reglamento de Servicios Digitales, DSA). Úsalo para avisarnos
de que un contenido procesado, almacenado o transmitido a través de
openshorts.app es ilícito, incluidos los que infringen derechos de autor,
marcas, derechos de imagen o privacidad. No necesitas tener cuenta ni abogado.</p>
<p>Si tu problema es con tu cuenta, tu facturación o tus datos, esta no es la
página: escribe a <a href="mailto:info@openshorts.app">info@openshorts.app</a> y cuéntanos qué necesitas.</p>

<h2>Cómo enviar una notificación</h2>
<p>Escribe a <strong><a href="mailto:info@openshorts.app">info@openshorts.app</a></strong> con el asunto
<strong>Reporte de contenido ilícito</strong>, en español o en inglés. Para que
podamos actuar sin tener que volver a preguntarte, la notificación necesita:</p>
<ul>
<li>la URL o el identificador exacto del material, con precisión suficiente
para localizarlo sin buscar;</li>
<li>una explicación de por qué lo consideras ilícito, y conforme a qué norma;</li>
<li>tu nombre y tu email, salvo que el reporte trate de material de abuso
sexual infantil o de delitos contra la libertad sexual, que puede enviarse de
forma anónima;</li>
<li>una declaración de que crees de buena fe que la información de la
notificación es exacta y completa.</li>
</ul>
<p>Si eres el titular de los derechos o actúas en su nombre, dinos cuál de las
dos cosas. Una notificación que incluya todo lo anterior nos da conocimiento
efectivo del contenido, que es lo que activa nuestra obligación de actuar.</p>

<h2>Qué pasa después</h2>
<p>Confirmamos la recepción de tu notificación sin dilaciones indebidas.
Después la revisamos de forma diligente y no arbitraria, con criterio humano y
no con una regla automática. Cuando la notificación está fundada, retiramos o
bloqueamos el acceso al material y te comunicamos qué hemos decidido y por qué.
Si decidimos no actuar, también te lo decimos, y por qué.</p>
<p>Las notificaciones manifiestamente infundadas, o las de quien las envía de
forma reiterada y de mala fe, ven suspendida su tramitación previa advertencia
al remitente. Las cuentas que suben material infractor de forma reiterada se
cancelan.</p>

<h2>Si te hemos retirado contenido</h2>
<p>Cuando retiramos tu contenido, lo restringimos o suspendemos tu cuenta por
estos motivos, recibes una declaración de motivos conforme al artículo 17 de la
DSA: qué hemos retirado, los hechos en que nos basamos, el fundamento legal o
contractual, si ha intervenido un sistema automatizado y cómo impugnar la
decisión.</p>
<p>Para impugnarla, responde a ese mensaje o escribe a <a href="mailto:info@openshorts.app">info@openshorts.app</a>. Lo
revisa una persona, no un filtro. Además conservas dos vías que no dependen de
nosotros: un órgano de resolución extrajudicial de litigios certificado
conforme al artículo 21 de la DSA, y los tribunales. Nada de esta página limita
ninguna de las dos.</p>

<h2>Notificaciones de copyright desde Estados Unidos</h2>
<p>El procedimiento anterior cubre las reclamaciones de derechos de autor estés
donde estés. Si prefieres enviar una notificación formal conforme a la Digital
Millennium Copyright Act estadounidense, el apartado 9 de nuestros
<a href="/terminos">Términos de Servicio</a> identifica a nuestro agente
designado y detalla lo que la sección 512(c)(3) exige que contenga.</p>

<h2>Punto de contacto</h2>
<p><a href="mailto:info@openshorts.app">info@openshorts.app</a> es nuestro punto único de contacto para los usuarios
(artículo 12 de la DSA) y para las autoridades de los Estados miembros, la
Comisión y la Junta Europea de Servicios Digitales (artículo 11 de la DSA).
Aceptamos comunicaciones en español y en inglés. El servicio lo opera TONVI
TECH SL, CIF B-19780394, Calle Puerta del Mar 18, 5ª planta, 29005 Málaga,
España.</p>
`

const REFUNDS_EN = `
<h2>The short version</h2>
<p>If you were charged in the last <strong>14 days</strong> and you have not
used the service since that charge, ask us and we refund it in full. No reason
needed and no form to fill in.</p>
<p>This is the "I did not notice the renewal" policy, and it is the case it is
built for: the payment went through, you never opened the app, and you should
not be out of pocket for it.</p>

<h2>What exactly is covered</h2>
<ul>
<li><strong>Your most recent charge</strong>, whether that is a first
subscription payment, a renewal or a minute top-up. Earlier charges are not
refundable under this policy.</li>
<li><strong>Within 14 days</strong> of that charge landing.</li>
<li><strong>Provided you have not used the service since it.</strong> Used
means minutes spent: a video processed, a clip generated, a dub or a thumbnail
made. Signing in and looking around is not using it.</li>
</ul>
<p>If you did use some of your minutes, this policy does not apply, but your
legal rights below may still get you money back.</p>

<h2>How to ask</h2>
<p>Email <strong><a href="mailto:info@openshorts.app">info@openshorts.app</a></strong> from the address your account
uses and say you want a refund. We check the date of the charge and whether any
minutes were spent after it, and we reply the same working day.</p>

<h2>When you get the money</h2>
<p>We issue the refund to the original payment method within 14 days of your
request, and normally the same day. Your bank or card issuer then takes 5 to 10
working days to show it, which is outside our control.</p>
<p>Refunding a subscription payment also cancels the subscription, so you are
not charged again. If you would rather stop future payments and keep using what
you already paid for, cancel from <strong>Account</strong> instead: your plan
runs to the end of the period you paid for.</p>

<h2>The one exception</h2>
<p>We do not refund an account terminated for abuse: uploading illegal content,
repeat copyright infringement, or trying to defraud the service.</p>

<h2>Your legal rights are separate, and they may be better</h2>
<p>If you are a consumer in the EU you have the statutory right of withdrawal
described in section 12 of our <a href="/terms">Terms of Service</a>. That
right lasts 14 days too, but it does <strong>not</strong> require that you left
the service unused: if you did use some of it, you can still withdraw and pay
only the proportional part of what you consumed. So if you have spent minutes,
ask for that instead. Nothing on this page takes it away from you, and we will
point you to it rather than let you lose it.</p>
`

const REFUNDS_ES = `
<h2>La versión corta</h2>
<p>Si te hemos cobrado en los últimos <strong>14 días</strong> y no has usado el
servicio desde ese cargo, pídelo y te lo devolvemos entero. Sin justificación y
sin formularios.</p>
<p>Esta es la política del "no me di cuenta de la renovación", y es justo el
caso para el que está pensada: el pago pasó, no llegaste a abrir la aplicación,
y no tienes por qué quedarte sin ese dinero.</p>

<h2>Qué cubre exactamente</h2>
<ul>
<li><strong>Tu último cargo</strong>, sea el primer pago de una suscripción, una
renovación o una recarga de minutos. Los cargos anteriores no entran en esta
política.</li>
<li><strong>Dentro de los 14 días</strong> siguientes a ese cargo.</li>
<li><strong>Siempre que no hayas usado el servicio desde entonces.</strong> Usar
significa gastar minutos: procesar un vídeo, generar un clip, doblar o crear una
miniatura. Entrar y mirar no cuenta como usarlo.</li>
</ul>
<p>Si sí gastaste minutos, esta política no se aplica, pero tus derechos legales
de más abajo pueden devolverte dinero igualmente.</p>

<h2>Cómo pedirlo</h2>
<p>Escribe a <strong><a href="mailto:info@openshorts.app">info@openshorts.app</a></strong> desde la dirección de tu
cuenta y dinos que quieres el reembolso. Comprobamos la fecha del cargo y si se
gastaron minutos después, y respondemos el mismo día laborable.</p>

<h2>Cuándo llega el dinero</h2>
<p>Emitimos el reembolso al medio de pago original en un máximo de 14 días desde
tu solicitud, y normalmente el mismo día. Después tu banco o tu emisora de
tarjeta tarda entre 5 y 10 días hábiles en reflejarlo, y ese tramo no está en
nuestra mano.</p>
<p>Reembolsar un pago de suscripción la cancela también, así que no se te vuelve
a cobrar. Si lo que quieres es dejar de pagar y seguir usando lo que ya pagaste,
cancela desde <strong>Cuenta</strong>: tu plan sigue hasta el final del periodo
abonado.</p>

<h2>La única excepción</h2>
<p>No reembolsamos una cuenta cancelada por abuso: subir contenido ilícito,
infringir derechos de autor de forma reiterada o intentar defraudar al
servicio.</p>

<h2>Tus derechos legales van aparte, y pueden ser mejores</h2>
<p>Si eres consumidor en la UE tienes el derecho de desistimiento que describe
el apartado 12 de nuestros <a href="/terminos">Términos de Servicio</a>. Ese
derecho dura también 14 días, pero <strong>no</strong> exige que hayas dejado el
servicio sin usar: si consumiste parte, puedes desistir igualmente y abonar solo
la parte proporcional de lo consumido. Así que si gastaste minutos, pide eso.
Nada de esta página te lo quita, y te lo recordaremos antes de dejar que lo
pierdas.</p>
`

export function legalPages() {
  return [
    {
      path: '/terms',
      title: 'Terms of Service | OpenShorts',
      description:
        'The terms that govern OpenShorts Cloud: your content stays yours, no AI training on your videos, EU consumer rights, retention periods and acceptable use.',
      h1: 'Terms of Service',
      breadcrumb: [{ name: 'Terms of Service' }],
      published: PUBLISHED,
      updated: PUBLISHED,
      tldr: [
        'You own your videos and the clips we generate from them. We never use your content to train AI models.',
        'You must hold the rights to every video you upload or link, and you are the publisher of everything you post through the service.',
        'Subscriptions renew automatically until you cancel, which you can do anytime from your account. EU consumers keep their 14-day withdrawal right.',
      ],
      body: TERMS_EN,
    },
    {
      path: '/privacy',
      title: 'Privacy Policy | OpenShorts',
      description:
        'What OpenShorts stores (email, billing reference, the videos you process), for how long, which providers touch it, and your GDPR rights.',
      h1: 'Privacy Policy',
      breadcrumb: [{ name: 'Privacy Policy' }],
      published: PUBLISHED,
      updated: PUBLISHED,
      tldr: [
        'We store your email, your billing reference and the videos you process — nothing beyond what the service needs.',
        'No advertising trackers: audience measurement is first-party, served from our own domain, and stays off until you accept it in the banner. BYOK API keys are obfuscated in your own browser and never reach our servers.',
        'Your content is never used to train AI models. Free-plan clips are deleted after 7 days.',
      ],
      body: PRIVACY_EN,
    },
    {
      path: '/legal-notice',
      title: 'Legal Notice | OpenShorts',
      description:
        'Legal identification of the operator of openshorts.app (TONVI TECH SL, Spain) as required by Spanish Law 34/2002 (LSSI-CE).',
      h1: 'Legal Notice',
      breadcrumb: [{ name: 'Legal Notice' }],
      published: PUBLISHED,
      updated: PUBLISHED,
      tldr: [
        'openshorts.app is operated by TONVI TECH SL, CIF B-19780394, Calle Puerta del Mar 18, 29005 Málaga, Spain — contact <a href="mailto:info@openshorts.app">info@openshorts.app</a>.',
      ],
      body: LEGAL_EN,
    },
    {
      path: '/terminos',
      title: 'Términos de Servicio | OpenShorts',
      description:
        'Los términos de OpenShorts Cloud: tu contenido sigue siendo tuyo, sin entrenar IA con tus vídeos, derechos del consumidor UE y plazos de conservación.',
      h1: 'Términos de Servicio',
      breadcrumb: [{ name: 'Términos de Servicio' }],
      published: PUBLISHED,
      updated: PUBLISHED,
      lang: 'es',
      tldr: [
        'Tus vídeos y los clips que generamos a partir de ellos son tuyos. Nunca usamos tu contenido para entrenar modelos de IA.',
        'Debes tener derechos sobre cada vídeo que subes o enlazas, y eres el editor de todo lo que publiques a través del servicio.',
        'Las suscripciones se renuevan automáticamente hasta que canceles, y puedes cancelar cuando quieras desde tu cuenta. Los consumidores de la UE conservan su derecho de desistimiento de 14 días.',
      ],
      body: TERMS_ES,
    },
    {
      path: '/privacidad',
      title: 'Política de Privacidad | OpenShorts',
      description:
        'Qué guarda OpenShorts (email, referencia de facturación, los vídeos que procesas), cuánto tiempo, qué proveedores intervienen y tus derechos RGPD.',
      h1: 'Política de Privacidad',
      breadcrumb: [{ name: 'Política de Privacidad' }],
      published: PUBLISHED,
      updated: PUBLISHED,
      lang: 'es',
      tldr: [
        'Guardamos tu email, tu referencia de facturación y los vídeos que procesas — nada más de lo que el servicio necesita.',
        'Sin rastreadores publicitarios: la medición de audiencia es propia, se sirve desde nuestro dominio y no se activa hasta que la aceptas en el banner. Las claves BYOK viven ofuscadas en tu navegador y nunca llegan a nuestros servidores.',
        'Tu contenido nunca se usa para entrenar modelos de IA. Los clips del plan gratuito se eliminan a los 7 días.',
      ],
      body: PRIVACY_ES,
    },
    {
      path: '/aviso-legal',
      title: 'Aviso Legal | OpenShorts',
      description:
        'Identificación legal del titular de openshorts.app (TONVI TECH SL, España) conforme al artículo 10 de la Ley 34/2002 (LSSI-CE).',
      h1: 'Aviso Legal',
      breadcrumb: [{ name: 'Aviso Legal' }],
      published: PUBLISHED,
      updated: PUBLISHED,
      lang: 'es',
      tldr: [
        'openshorts.app es un servicio de TONVI TECH SL, CIF B-19780394, Calle Puerta del Mar 18, 29005 Málaga, España — contacto <a href="mailto:info@openshorts.app">info@openshorts.app</a>.',
      ],
      body: LEGAL_ES,
    },
    {
      path: '/report-content',
      title: 'Report Illegal Content | OpenShorts',
      description:
        'How to report illegal or infringing content on openshorts.app: what a notice must contain, what we do with it, and how to challenge a removal. Article 16 DSA.',
      h1: 'Report Illegal Content',
      breadcrumb: [{ name: 'Report Illegal Content' }],
      published: PUBLISHED,
      updated: PUBLISHED,
      tldr: [
        'Email <a href="mailto:info@openshorts.app">info@openshorts.app</a> with the subject "Illegal content report", the exact URL, why it is illegal, and your contact details.',
        'We confirm receipt without undue delay, review it with human judgement, and tell you what we decided and why.',
        'If we removed your content you get a statement of reasons and can challenge it, with the courts and an Article 21 dispute body always open to you.',
      ],
      body: REPORT_EN,
    },
    {
      path: '/reportar-contenido',
      title: 'Reportar Contenido Ilícito | OpenShorts',
      description:
        'Cómo reportar contenido ilícito o infractor en openshorts.app: qué debe contener la notificación, qué hacemos con ella y cómo impugnar una retirada.',
      h1: 'Reportar Contenido Ilícito',
      breadcrumb: [{ name: 'Reportar Contenido Ilícito' }],
      published: PUBLISHED,
      updated: PUBLISHED,
      lang: 'es',
      tldr: [
        'Escribe a <a href="mailto:info@openshorts.app">info@openshorts.app</a> con el asunto "Reporte de contenido ilícito", la URL exacta, por qué es ilícito y tus datos de contacto.',
        'Confirmamos la recepción sin dilaciones indebidas, lo revisa una persona y te comunicamos qué decidimos y por qué.',
        'Si te hemos retirado contenido recibes una declaración de motivos y puedes impugnarla, con los tribunales y un órgano del artículo 21 siempre abiertos.',
      ],
      body: REPORT_ES,
    },
    {
      path: '/refunds',
      title: 'Refund Policy | OpenShorts',
      description:
        'Charged in the last 14 days and have not used the service since? OpenShorts refunds it in full. What counts as used, and how to ask.',
      h1: 'Refund Policy',
      breadcrumb: [{ name: 'Refund Policy' }],
      published: PUBLISHED,
      updated: PUBLISHED,
      tldr: [
        'Charged in the last 14 days and have not used the service since? We refund it in full, no reason needed.',
        'It covers your most recent charge only, and it is built for the renewal you did not notice. One email to <a href="mailto:info@openshorts.app">info@openshorts.app</a>.',
        'If you did spend minutes, the EU withdrawal right in section 12 of the Terms may still refund the part you did not use.',
      ],
      body: REFUNDS_EN,
    },
    {
      path: '/reembolsos',
      title: 'Política de Reembolsos | OpenShorts',
      description:
        '¿Te han cobrado en los últimos 14 días y no has usado el servicio? OpenShorts te lo devuelve entero. Qué cuenta como usarlo y cómo pedirlo.',
      h1: 'Política de Reembolsos',
      breadcrumb: [{ name: 'Política de Reembolsos' }],
      published: PUBLISHED,
      updated: PUBLISHED,
      lang: 'es',
      tldr: [
        '¿Te hemos cobrado en los últimos 14 días y no has usado el servicio desde entonces? Te lo devolvemos entero, sin justificación.',
        'Cubre solo tu último cargo, y está pensada para la renovación que no te sonó. Basta un email a <a href="mailto:info@openshorts.app">info@openshorts.app</a>.',
        'Si sí gastaste minutos, el desistimiento del apartado 12 de los Términos puede devolverte la parte no consumida.',
      ],
      body: REFUNDS_ES,
    },
  ]
}
