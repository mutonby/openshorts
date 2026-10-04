# vantclip.app →  # 🎬 VANTCLIP

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
Buscar: OpenShorts y lo cambia por vantclip
Reemplazar: VANTCLIP

Buscar: openshorts.app y lo cambia por vantclip.app
Reemplazar: vantclip.app

Buscar: open source AI clip generator
Reemplazar: AI clip generator en español

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
DISCORD_URL=https://discord.gg/kYDc6Jzp62

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

# ===========================================
# VANTCLIP - Variables de entorno
# Copia este archivo a .env y configura tus keys
# ===========================================

# ===========================================
# APIs de IA (requeridas para core features)
# ===========================================
GEMINI_API_KEY=sk-...  # https://aistudio.google.com/app/apikey
FAL_KEY=...  # https://fal.ai (AI Shorts, actores)
ELEVENLABS_API_KEY=...  # https://elevenlabs.io (voiceover, doblaje)
UPLOAD_POST_API_KEY=...  # https://upload-post.com (auto-publishing)

# ===========================================
# Monetización (Stripe)
# ===========================================
STRIPE_SECRET_KEY=sk_test_...  # https://stripe.com
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_ID_STARTER=price_1ABC...  # $15/mes
STRIPE_PRICE_ID_PRO=price_2DEF...      # $29/mes
STRIPE_PRICE_ID_BUSINESS=price_3GHI... # $79/mes

# ===========================================
# AWS S3 (opcional, para backup en la nube)
# ===========================================
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
AWS_S3_BUCKET=vantclip-backups-private
AWS_S3_PUBLIC_BUCKET=vantclip-gallery-public

# ===========================================
# Configuración de la app
# ===========================================
MAX_CONCURRENT_JOBS=5
DEFAULT_RESOLUTION=1080
ENABLE_WATERMARK=false
DEFAULT_LANGUAGE=es
APP_NAME=VANTCLIP
APP_URL=http://localhost:5173
SUPPORT_EMAIL=soporte@vantclip.app
DISCORD_URL=https://discord.gg/tu-invite

# ===========================================
# Base de datos (SQLite por defecto)
# ===========================================
DATABASE_URL=sqlite:///vantclip.db

# ===========================================
# LLM local (opcional, para no usar Gemini)
# ===========================================
# LLM_BASE_URL=http://host.docker.internal:11434/v1  # Ollama
# LLM_MODEL=qwen2.5:14b
# LLM_API_KEY=...  # solo si tu servidor requiere

version: '3.8'

services:
  backend:
    image: openshorts-backend
    build:
      context: .
      args:
        GPU: ${GPU:-0}
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY:-}
      - FAL_KEY=${FAL_KEY:-}
      - ELEVENLABS_API_KEY=${ELEVENLABS_API_KEY:-}
      - UPLOAD_POST_API_KEY=${UPLOAD_POST_API_KEY:-}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-}
      - AWS_REGION=${AWS_REGION:-us-east-1}
      - AWS_S3_BUCKET=${AWS_S3_BUCKET:-}
      - AWS_S3_PUBLIC_BUCKET=${AWS_S3_PUBLIC_BUCKET:-}
      - MAX_CONCURRENT_JOBS=${MAX_CONCURRENT_JOBS:-5}
      # Stripe para monetización
      - STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY:-}
      - STRIPE_WEBHOOK_SECRET=${STRIPE_WEBHOOK_SECRET:-}
      - APP_URL=${APP_URL:-http://localhost:5173}
    volumes:
      - ./output:/app/output
      - ./temp:/app/temp
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    image: openshorts-frontend
    build:
      context: ./frontend
    ports:
      - "5173:5173"
    environment:
      - VITE_API_URL=http://localhost:8000
    depends_on:
      - backend
    restart: unless-stopped

    # ===========================================
# Stripe endpoints para monetización
# ===========================================

import stripe
from fastapi import HTTPException, Request
from dotenv import load_dotenv
import os

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
        raise HTTPException(status_code=500, detail="Stripe no configurado. Contacta al admin.")
    
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
    
    if not os.getenv("STRIPE_WEBHOOK_SECRET"):
        raise HTTPException(status_code=500, detail="STRIPE_WEBHOOK_SECRET no configurado")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.getenv("STRIPE_WEBHOOK_SECRET")
        )
    except (ValueError, stripe.error.SignatureVerificationError) as e:
        raise HTTPException(status_code=400, detail=f"Webhook inválido: {str(e)}")
    
    # Maneja el evento
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session.get("metadata", {}).get("user_id")
        plan = session.get("metadata", {}).get("plan")
        
        # Aquí actualizarías tu base de datos de usuarios
        # Ejemplo con SQLite simple:
        # import sqlite3
        # conn = sqlite3.connect('vantclip.db')
        # c = conn.cursor()
        # c.execute("UPDATE users SET plan=?, status='active' WHERE id=?", (plan, user_id))
        # conn.commit()
        # conn.close()
        
        print(f"✅ Usuario {user_id} se suscribió al plan {plan}")
    
    elif event["type"] == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        user_id = subscription.get("metadata", {}).get("user_id")
        
        # Downgrade a free
        # c.execute("UPDATE users SET plan='free', status='cancelled' WHERE id=?", (user_id,))
        
        print(f"❌ Usuario {user_id} canceló su suscripción")
    
    return {"received": True}

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
      console.error('Error al crear checkout:', error)
      alert('Error al crear sesión de pago. Intenta de nuevo o contacta a soporte@vantclip.app')
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

import { useState, useEffect } from 'react'
import axios from 'axios'
import PricingTable from './components/PricingTable'  # ← AÑADE ESTA LÍNEA

function App() {
  // ... resto de tu código ...

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white">
      {/* Navbar */}
      <nav className="bg-white dark:bg-gray-800 shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold">🎬 VANTCLIP</h1>
            </div>
            <div className="flex items-center space-x-4">
              <a href="#features" className="hover:text-blue-600">Features</a>
              <a href="#pricing" className="hover:text-blue-600">Precios</a>
              <a href="https://github.com/feispla/VANTCLIP" className="hover:text-blue-600">GitHub</a>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-purple-600 text-white">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <h2 className="text-4xl md:text-6xl font-bold mb-6">
            Convierte podcasts en shorts virales automáticamente
          </h2>
          <p className="text-xl md:text-2xl mb-8 max-w-3xl mx-auto">
            El AI clip generator open source en español. Free, sin marca de agua, listo para monetizar.
          </p>
          <div className="flex justify-center gap-4">
            <a href="#pricing" className="bg-white text-blue-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition">
              Ver planes
            </a>
            <a href="https://github.com/feispla/VANTCLIP" className="bg-blue-800 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 transition">
              GitHub
            </a>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-white dark:bg-gray-800">
        <div className="max-w-7xl mx-auto px-4">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-12">
            3 Herramientas en 1 Plataforma
          </h2>
          {/* ... tus features existentes ... */}
        </div>
      </section>

      {/* Pricing Section - AÑADE ESTO */}
      <section id="pricing" className="py-20 bg-gray-50 dark:bg-gray-900">
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

      {/* Footer */}
      <footer className="bg-gray-800 text-gray-300 py-12">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p>
            <a href="https://github.com/feispla/VANTCLIP" className="hover:text-white">VANTCLIP</a> · 
            <a href="https://github.com/mutonby/openshorts" className="hover:text-white ml-2">Basado en OpenShorts</a>
          </p>
          <p className="mt-4 text-sm">
            Hecho con ❤️ para creators en español
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
# Términos de Servicio - VANTCLIP

**Última actualización:** Octubre 2026

## 1. Aceptación de los términos

Al usar VANTCLIP, aceptas estos términos. Si no estás de acuerdo, no uses el servicio.

## 2. Licencia

VANTCLIP está bajo licencia MIT para el código core. El uso comercial está permitido.

## 3. Uso aceptable

NO puedes usar VANTCLIP para:
- Contenido ilegal o infringing
- Spam o contenido engañoso
- Violar derechos de autor de terceros

## 4. Monetización

Los planes paid incluyen:
- Acceso a features premium según el plan
- Soporte prioritario
- Sin marca de agua

Reembolsos: 14 días garantía de satisfacción.

## 5. Privacidad

- No vendemos tus datos
- Los videos procesados son tuyos
- API keys se guardan encriptadas client-side

## 6. Limitación de responsabilidad

VANTCLIP se provee "AS IS" sin garantías.

## 7. Cambios a los términos

Podemos actualizar estos términos. Te notificaremos por email.

## 8. Contacto

Email: soporte@vantclip.app

# Política de Privacidad - VANTCLIP

**Última actualización:** Octubre 2026

## Datos que recopilamos

- Email (para cuenta y soporte)
- Videos que subes (procesamiento local o en tu servidor)
- API keys (encriptadas client-side, nunca en nuestro servidor si self-hosteas)

## Cómo usamos tus datos

- Procesar tus videos
- Mejorar el servicio
- Soporte técnico

## Compartir datos

NO compartimos tus datos con terceros excepto:
- Proveedores de infraestructura (AWS, si usas hosted)
- Requerimiento legal

## Tus derechos

- Acceder a tus datos
- Eliminar tu cuenta
- Exportar tus videos

## Contacto

Email: soporte@vantclip.app

import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

function Success() {
  const [searchParams] = useSearchParams()
  const sessionId = searchParams.get('session_id')
  const [status, setStatus] = useState('verifying')

  useEffect(() => {
    // Verifica el pago con el backend
    const verifyPayment = async () => {
      try {
        const res = await fetch(`/api/verify-session?session_id=${sessionId}`)
        const data = await res.json()
        
        if (data.valid) {
          setStatus('success')
        } else {
          setStatus('invalid')
        }
      } catch (error) {
        setStatus('error')
      }
    }

    if (sessionId) {
      verifyPayment()
    }
  }, [sessionId])

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
      <div className="text-center">
        {status === 'success' && (
          <>
            <div className="text-6xl mb-4">🎉</div>
            <h1 className="text-3xl font-bold mb-4">¡Pago exitoso!</h1>
            <p className="text-gray-600 dark:text-gray-300 mb-8">
              Tu suscripción está activa. Revisa tu email para los detalles.
            </p>
            <a href="/dashboard" className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700">
              Ir al dashboard
            </a>
          </>
        )}

        {status === 'verifying' && (
          <>
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600 dark:text-gray-300">Verificando tu pago...</p>
          </>
        )}

        {status === 'invalid' && (
          <>
            <div className="text-6xl mb-4">❌</div>
            <h1 className="text-3xl font-bold mb-4">Pago no válido</h1>
            <p className="text-gray-600 dark:text-gray-300 mb-8">
              No pudimos verificar tu pago. Contacta a soporte.
            </p>
            <a href="mailto:soporte@vantclip.app" className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700">
              Contactar soporte
            </a>
          </>
        )}

        {status === 'error' && (
          <>
            <div className="text-6xl mb-4">⚠️</div>
            <h1 className="text-3xl font-bold mb-4">Error de verificación</h1>
            <p className="text-gray-600 dark:text-gray-300 mb-8">
              Algo salió mal. Intenta de nuevo o contacta soporte.
            </p>
            <a href="/pricing" className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700">
              Volver a precios
            </a>
          </>
        )}
      </div>
    </div>
  )
}

export default Success

## Archivos de configuración
- [ ] `.env.example` completo con Stripe
- [ ] `docker-compose.yml` con variables de Stripe
- [ ] Endpoints de Stripe en `app.py`

## Frontend
- [ ] Crear `frontend/src/components/PricingTable.jsx`
- [ ] Integrar PricingTable en `App.jsx`
- [ ] Crear página `frontend/src/pages/Success.jsx`
- [ ] Actualizar `frontend/index.html` (título, descripción)
- [ ] Cambiar logo/header a "VANTCLIP"

## Legal
- [ ] Crear `docs/TERMS.md`
- [ ] Crear `docs/PRIVACY.md`

## Stripe (configuración real)
- [ ] Crear cuenta en https://stripe.com
- [ ] Crear productos y precios en Stripe Dashboard
- [ ] Configurar webhook endpoint
- [ ] Probar checkout localmente con Stripe CLI

## Testing
- [ ] Docker compose up funciona
- [ ] Checkout de Stripe redirige correctamente
- [ ] Webhook recibe eventos
- [ ] PricingTable se ve bien en mobile

## Marketing (pre-lanzamiento)
- [ ] Landing page con email capture
- [ ] Demo video (60s)
- [ ] Discord server
- [ ] Twitter/X account

