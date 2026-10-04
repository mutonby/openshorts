# OpenShorts.app  →  # 🎬 VANTCLIP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://opensource.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![GitHub stars](https://img.shields.io/github/stars/mutonby/openshorts?style=social)](https://github.com/mutonby/openshorts)  →  [![GitHub stars](https://img.shields.io/github/stars/feispla/VANTCLIP?style=social)](https://github.com/feispla/VANTCLIP)

**Open source AI video platform** with 3 tools in one...  →  
**🇪🇸 El AI clip generator open source en español** — Convierte podcasts y videos largos en shorts virales para TikTok, Instagram Reels y YouTube Shorts automáticamente.

![Your podcast, and the vertical clip OpenShorts makes of it...]  →  
![Demo VANTCLIP](screenshots/split-before-after.gif)  # Mantén el mismo GIF

**Two ways to run it, same software either way:**  →  
**Dos formas de usar VANTCLIP:**

|  | Self-hosted (este repo) | Hosted en [vantclip.app](https://vantclip.app) (próximamente) |
|---|---|---|
| **Precio** | Free forever, MIT | Free plan, paid desde $12/mes |
| **Velocidad** | 5-8 min por video de 8 min en CPU | ~50s en GPU NVIDIA |
| **API keys** | Trae las tuyas (Gemini, ElevenLabs, fal.ai) | Gemini incluido, sin configuración |
| **Marca de agua** | Nunca | Free: 20 min/mes con watermark, Paid: sin límites |
| **Setup** | Docker, 8GB+ RAM | Sign in y pega un link |
| **Tus datos** | Tu servidor | Nuestra infraestructura |

Self-hosting es genuinamente free y siempre lo será. Los planes hosted existen para cubrir hardware y API keys, no para desbloquear features.

## 3 Herramientas en 1 Plataforma

### 1. Clip Generator (Generador de Clips)
Convierte tus videos largos — podcasts, webinars, livestreams, vlogs, entrevistas — en shorts listos para volverse virales en TikTok, Instagram Reels y YouTube Shorts.

### 2. AI Shorts (Creador de Videos UGC)
Genera videos de marketing con actores AI para **cualquier producto o negocio**. Sin cámara, sin estudio, sin presupuesto de influencers. Solo describe tu producto o pega una URL.

### 3. YouTube Studio
Toolkit completo de YouTube con IA: thumbnails, títulos, descripciones y publicación directa.


---

## 🆚 VANTCLIP vs Competidores

| Feature | VANTCLIP | OpenShorts | Opus Clip | Vizard | Submagic |
|---------|:---:|:---:|:---:|:---:|:---:|
| **Precio** | **Free self-hosted**<br>desde $12/mo hosted | Free self-hosted | $15-29/mo | $15-20/mo | $12-41/mo |
| **Self-hosted** | **Sí** | Sí | No | No | No |
| **Open source** | **Sí** | Sí | No | No | No |
| **Marca de agua** | **Nunca self-hosted** | Nunca | Free tier | Free tier | Free tier |
| **Español nativo** | **✅ Optimizado** | ⚠️ Regular | ❌ No | ⚠️ Regular | ⚠️ Regular |
| **Límites de upload** | **Ilimitado self-hosted** | Ilimitado | 10-30GB | 60min-10hr | Por videos/mes |
| **Detección de clips AI** | Sí | Sí | Sí | Sí | Sí |
| **Smart 9:16 reframing** | Sí | Sí | Sí | Sí | Sí |
| **Subtítulos automáticos** | Sí | Sí | Sí | Sí | Sí |
| **Doblaje de voz (30+ langs)** | Sí | Sí | No | No | Pro only |
| **Actores AI UGC** | **Sí** | Sí | No | No | No |
| **Auto-publishing social** | Sí | Sí | Pro only | Paid only | Paid only |
| **Programar uploads** | Sí | Sí | Pro only | Paid only | No |
| **Privacidad de datos** | **Tu servidor** | Tu servidor | Su cloud | Su cloud | Su cloud |
| **Funciona con LLM local (Ollama)** | **Sí** | Sí | No | No | No |
| **Monetización lista (Stripe)** | **✅ Sí** | ❌ No | N/A | N/A | N/A |

---

## 💰 Monetización para Creadores

VANTCLIP no solo es una herramienta, es una **oportunidad de negocio**:

### Para Agencies y Freelancers
- **Plan Agency**: $299/mes por 2000 minutos + 10 team seats
- **White-label**: Sin branding de VANTCLIP, tu logo en clips
- **API access**: Integra en tus dashboards para clientes

### Para Creadores de Contenido
- **Affiliate program**: 30% de comisión recurrente
- **Lifetime deal**: Código único para tus followers (20% off de por vida)
- **Early access**: Features nuevas antes que nadie

### Para Desarrolladores
- **API desde $50/mes**: 500 minutos procesados, 100 requests/hora
- **Enterprise**: Custom pricing para volúmenes altos
- **On-premise deployment**: Para empresas que requieren privacidad total

[Ver planes completos →](https://vantclip.app/pricing)


| Servicio | Free Tier | Costo Paid | Usado Para |
|---------|-----------|-----------|------------|
| **Google Gemini** | Free trial generoso | < $0.01 por video de 10 min | Detección de momentos virales |
| **Local LLM (Ollama)** | **Free, tu hardware** | $0 | Moment picker sin Google |
| **fal.ai** | Pay-per-use | ~$0.50-1.50 por AI Short | Generación de actores |
| **ElevenLabs** | Free tier disponible | Pay-per-use | Voiceover, doblaje |
| **Upload-Post** | **10 uploads gratis/mes** | Pay-per-use | Auto-publishing a redes |
| **AWS S3** | Opcional | ~$0.023/GB | Backup en la nube |

**En resumen:** Puedes clippear videos casi gratis con Gemini, y publicar 10 videos/mes a todas las redes sin costo con Upload-Post.

**¿No quieres correr nada de eso?** [vantclip.app](https://vantclip.app) es el mismo software en nuestro hardware: nuestra GPU NVIDIA clippea un video de 8 min en ~50s en lugar de 5-8 min en CPU típica, la key de Gemini está incluida, y el auto-publishing ya está configurado. Free plan: 20 minutos/mes con watermark sin tarjeta; paid desde $12/mes por 100 minutos sin watermark.


# En el repo, busca estos archivos:
frontend/src/App.jsx
frontend/src/main.jsx
frontend/index.html
frontend/package.json


<!doctype html>
<html lang="en">  →  <html lang="es">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>OpenShorts</title>  →  <title>VANTCLIP | AI Clip Generator en Español</title>
    <meta name="description" content="Open source AI clip generator">  →  
    <meta name="description" content="Convierte podcasts en shorts virales automáticamente. Free, open source, en español.">
  </head>


  <header>
  <h1>OpenShorts</h1>  →  <h1>🎬 VANTCLIP</h1>
  <p>Open source AI clip generator</p>  →  
  <p>Convierte podcasts en shorts virales automáticamente</p>
</header>

<footer>
  <a href="https://github.com/mutonby/openshorts">OpenShorts</a>  →  
  <a href="https://github.com/feispla/VANTCLIP">VANTCLIP</a> · 
  <a href="https://github.com/mutonby/openshorts">Basado en OpenShorts</a>
</footer>


# En VS Code, usa Ctrl+Shift+H (Find and Replace in Files)
Buscar: OpenShorts
Reemplazar: VANTCLIP

Buscar: openshorts.app
Reemplazar: vantclip.app

Buscar: open source AI clip generator
Reemplazar: AI clip generator en español

# ===========================================
# Stripe endpoints para monetización
# ===========================================

import stripe
from fastapi import HTTPException, Request
from dotenv import load_dotenv

load_dotenv()

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

@app.post("/api/create-checkout-session")
async def create_checkout_session(plan: str = "starter"):
    """Crea sesión de checkout de Stripe"""
    
    prices = {
        "starter": os.getenv("STRIPE_PRICE_ID_STARTER"),    # $15/mes
        "pro": os.getenv("STRIPE_PRICE_ID_PRO"),            # $29/mes
        "business": os.getenv("STRIPE_PRICE_ID_BUSINESS"),  # $79/mes
    }
    
    if plan not in prices:
        raise HTTPException(status_code=400, detail="Plan inválido")
    
    if not prices[plan]:
        raise HTTPException(status_code=500, detail="Stripe no configurado")
    
    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price": prices[plan],
            "quantity": 1,
        }],
        mode="subscription",
        success_url=f"{os.getenv('APP_URL', 'http://localhost:5173')}/success?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{os.getenv('APP_URL', 'http://localhost:5173')}/pricing",
        metadata={
            "plan": plan,
        }
    )
    
    return {"url": session.url}

@app.post("/api/webhook")
async def stripe_webhook(request: Request):
    """Webhook para confirmar pagos de Stripe"""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.getenv("STRIPE_WEBHOOK_SECRET")
        )
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Webhook inválido")
    
    # Maneja el evento
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session.get("metadata", {}).get("user_id")
        plan = session.get("metadata", {}).get("plan")
        
        # Aquí actualizarías tu base de datos
        # Ejemplo: db.users.update(user_id, {"plan": plan, "status": "active"})
        print(f"Usuario {user_id} se suscribió al plan {plan}")
    
    return {"received": True}


    # ===========================================
# Monetización (Stripe)
# ===========================================
STRIPE_SECRET_KEY=sk_test_...  # https://stripe.com
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_ID_STARTER=price_1ABC...  # $15/mes
STRIPE_PRICE_ID_PRO=price_2DEF...      # $29/mes
STRIPE_PRICE_ID_BUSINESS=price_3GHI... # $79/mes

# ===========================================
# Dominio y branding
# ===========================================
APP_NAME=VANTCLIP
APP_URL=https://vantclip.app
SUPPORT_EMAIL=soporte@vantclip.app
DISCORD_URL=https://discord.gg/tu-invite

import { useState } from 'react'
import axios from 'axios'

function PricingTable() {
  const [loading, setLoading] = useState(null)

  const plans = [
    {
      name: 'Free',
      price: '$0',
      period: '/mes',
      features: [
        '60 minutos procesados/mes',
        'Sin marca de agua',
        'Subtítulos básicos',
        '3 clips máximos por video',
        'Soporte comunitario (Discord)',
      ],
      cta: 'Empezar gratis',
      highlighted: false,
    },
    {
      name: 'Starter',
      price: '$15',
      period: '/mes',
      features: [
        '150 minutos procesados/mes',
        'Subtítulos personalizados',
        '10 clips por video',
        'Auto-posting a 1 red social',
        'Doblaje en 5 idiomas',
        'Soporte por email (24-48h)',
      ],
      cta: 'Elegir Starter',
      highlighted: true,
    },
    {
      name: 'Pro',
      price: '$29',
      period: '/mes',
      features: [
        '400 minutos procesados/mes',
        'Resolución 4K',
        'Clips ilimitados',
        'Auto-posting a 3 redes',
        'Doblaje en 30+ idiomas',
        'AI video effects',
        'Analytics de viralidad',
        'Soporte prioritario (<12h)',
      ],
      cta: 'Elegir Pro',
      highlighted: false,
    },
    {
      name: 'Business',
      price: '$79',
      period: '/mes',
      features: [
        '1500 minutos procesados/mes',
        'API access',
        'Webhooks',
        '5 team seats',
        'Brand kits personalizados',
        'White-label',
        'SLA 99.9% uptime',
        'Soporte dedicado (Slack)',
      ],
      cta: 'Contactar ventas',
      highlighted: false,
    },
  ]

  const handleCheckout = async (plan) => {
    if (plan === 'Free') {
      window.location.href = '/signup'
      return
    }
    
    if (plan === 'Business') {
      window.location.href = '/contact'
      return
    }

    setLoading(plan)
    try {
      const res = await axios.post('/api/create-checkout-session', { plan })
      window.location.href = res.data.url
    } catch (error) {
      console.error('Error:', error)
      alert('Error al crear sesión de pago. Intenta de nuevo o contacta a soporte.')
    } finally {
      setLoading(null)
    }
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 max-w-7xl mx-auto px-4">
      {plans.map((plan) => (
        <div
          key={plan.name}
          className={`rounded-2xl p-8 border-2 flex flex-col ${
            plan.highlighted
              ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 shadow-xl scale-105'
              : 'border-gray-200 dark:border-gray-700'
          }`}
        >
          <h3 className="text-2xl font-bold">{plan.name}</h3>
          <div className="mt-4 flex items-baseline">
            <span className="text-4xl font-extrabold">{plan.price}</span>
            <span className="ml-1 text-gray-500 dark:text-gray-400">{plan.period}</span>
          </div>
          <ul className="mt-6 space-y-4 flex-grow">
            {plan.features.map((feature, idx) => (
              <li key={idx} className="flex items-start">
                <svg className="w-5 h-5 text-green-500 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                <span className="text-sm">{feature}</span>
              </li>
            ))}
          </ul>
          <button
            onClick={() => handleCheckout(plan.name.toLowerCase())}
            disabled={loading === plan.name.toLowerCase()}
            className={`mt-8 w-full py-3 px-6 rounded-lg font-semibold transition-colors ${
              plan.highlighted
                ? 'bg-blue-600 text-white hover:bg-blue-700 disabled:bg-blue-400'
                : 'bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 disabled:bg-gray-400'
            }`}
          >
            {loading === plan.name.toLowerCase() ? 'Procesando...' : plan.cta}
          </button>
        </div>
      ))}
    </div>
  )
}

export default PricingTable


// En frontend/src/App.jsx o donde tengas las rutas
import PricingTable from './components/PricingTable'

function App() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* ... resto de tu app ... */}
      
      <section id="pricing" className="py-20 bg-white dark:bg-gray-800">
        <div className="max-w-7xl mx-auto px-4">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-4">
            Planes simples y transparentes
          </h2>
          <p className="text-center text-gray-600 dark:text-gray-300 mb-12 max-w-2xl mx-auto">
            Empieza gratis, escala cuando necesites más. Sin sorpresas.
          </p>
          <PricingTable />
        </div>
      </section>
      
      {/* ... más secciones ... */}
    </div>
  )
}

STRIPE_SECRET_KEY=sk_test_51ABC...
STRIPE_WEBHOOK_SECRET=whsec_123...
STRIPE_PRICE_ID_STARTER=price_1ABC...
STRIPE_PRICE_ID_PRO=price_2DEF...
STRIPE_PRICE_ID_BUSINESS=price_3GHI...

## Branding
- [ ] README.md actualizado con "VANTCLIP" y descripción en español
- [ ] Tabla comparativa añadida (VANTCLIP vs OpenShorts vs competidores)
- [ ] Sección de monetización añadida al README
- [ ] Frontend: título de página cambiado a "VANTCLIP"
- [ ] Frontend: logo/header actualizado
- [ ] Frontend: colores de marca (azul + morado recomendado)

## Monetización
- [ ] Cuenta de Stripe creada
- [ ] Productos y precios configurados en Stripe Dashboard
- [ ] API keys de Stripe en `.env`
- [ ] Webhook endpoint configurado
- [ ] Componente PricingTable.jsx creado e integrado
- [ ] Endpoints de Stripe en backend (`/api/create-checkout-session`, `/api/webhook`)

## Legal
- [ ] Términos de servicio (https://termly.io o https://iubenda.com)
- [ ] Política de privacidad
- [ ] Página de contacto/soporte visible

## Marketing
- [ ] Landing page con email capture (opcional si ya lanzas)
- [ ] Demo video (60s) mostrando el flujo completo
- [ ] Discord server creado (https://discord.com)
- [ ] Twitter/X account para el proyecto

## Técnico
- [ ] Docker compose up funciona localmente
- [ ] Tests del pipeline corren sin errores
- [ ] API endpoints documentados
- [ ] 
