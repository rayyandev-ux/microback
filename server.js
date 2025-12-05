import 'dotenv/config'
import express from 'express'
import cors from 'cors'
import { OpenAI } from 'openai'
import { createClient } from 'redis'
import { query, initDb } from './db.js'

import { Groq } from 'groq-sdk'
import sharp from 'sharp'

const app = express()
app.use(cors())
app.use(express.json({ limit: '25mb' }))

// Initialize Database
initDb()
app.use(express.urlencoded({ extended: true, limit: '25mb' }))

function makeRedisUrl() {
  if (process.env.REDIS_URL) return process.env.REDIS_URL
  const scheme = String(process.env.REDIS_SSL).toLowerCase() === 'true' ? 'rediss' : 'redis'
  const host = process.env.REDIS_HOST || '127.0.0.1'
  const port = process.env.REDIS_PORT || '6379'
  const user = process.env.REDIS_USERNAME || ''
  const pass = process.env.REDIS_PASSWORD || ''
  const auth = user || pass ? `${user}:${pass}@` : ''
  return `${scheme}://${auth}${host}:${port}`
}
const redis = createClient({ url: makeRedisUrl() })
redis.on('error', (e) => {})
try {
  await redis.connect()
} catch (_) {}

const PATH = '/webhook'

function get(obj, path, def) {
  try {
    return path.split('.').reduce((o, k) => (o && k in o ? o[k] : undefined), obj) ?? def
  } catch (_) {
    return def
  }
}

function isEmptyNumber(v) {
  return v === undefined || v === null || v === ''
}

function trimOrEmpty(s) {
  if (typeof s !== 'string') return ''
  const t = s.trim()
  return t
}

function collapseSpaces(s) {
  return s.replace(/\s+/g, ' ').trim()
}

function isDataUrl(u) {
  return typeof u === 'string' && u.startsWith('data:')
}



async function bufferFromDataUrl(u) {
  const [, meta, data] = u.match(/^data:(.*?);base64,(.*)$/) || []
  const mime = meta || 'application/octet-stream'
  const buf = Buffer.from(data || '', 'base64')
  return { buf, mime }
}

async function fetchBuffer(u) {
  if (isDataUrl(u)) return bufferFromDataUrl(u)
  const res = await fetch(u)
  const ct = res.headers.get('content-type') || 'application/octet-stream'
  const buf = Buffer.from(await res.arrayBuffer())
  return { buf, mime: ct }
}

async function logUsage(model, usage, mastertext) {
  if (!usage) {
    console.warn('Warning: No usage data provided for model:', model)
    // Even if no usage is provided, we might want to log it as 0 tokens if it's an important event
    // But for now, let's stick to returning if no usage, OR handle specific cases.
    // However, Groq might return usage, so we expect it.
    return
  }
  try {
    const { prompt_tokens, completion_tokens, total_tokens } = usage
    
    // Ensure values are numbers
    const p_tokens_raw = Number(prompt_tokens) || 0
    const c_tokens_raw = Number(completion_tokens) || 0
    const t_tokens_raw = Number(total_tokens) || (p_tokens_raw + c_tokens_raw)
    
    // Round for DB integer columns
    const p_tokens = Math.ceil(p_tokens_raw)
    const c_tokens = Math.ceil(c_tokens_raw)
    const t_tokens = Math.ceil(t_tokens_raw)

    // Insert log
    await query(
      'INSERT INTO openai_usage (model, prompt_tokens, completion_tokens, total_tokens, mastertext) VALUES ($1, $2, $3, $4, $5)',
      [model, p_tokens, c_tokens, t_tokens, mastertext]
    )

    // Update daily summary
    const today = new Date().toISOString().split('T')[0]
    // Approximate cost calculation
    let cost = 0
    if (model === 'gpt-4o-mini') {
       // ~$0.15 / 1M input, ~$0.60 / 1M output
       cost = (p_tokens * 0.15 / 1000000) + (c_tokens * 0.60 / 1000000)
    } else if (model.includes('llava') || model.includes('llama')) {
       // Groq Vision Models
       // Llama 4 Scout (17B): ~$0.11 / 1M input, ~$0.34 / 1M output (Preview pricing may vary, using estimate)
       // Assuming similar to generic preview or low cost
       // Using values from search: $0.11 input, $0.34 output
       cost = (p_tokens * 0.11 / 1000000) + (c_tokens * 0.34 / 1000000)
    } else if (model === 'whisper-large-v3-turbo') {
       // Audio billed by time. 
       // We store DURATION (seconds) in total_tokens for this model.
       // Price: $0.111 / hour (Updated per user request)
       // Cost = (seconds / 3600) * 0.111
       // Use raw value for better cost precision
       cost = (t_tokens_raw / 3600) * 0.111
    }
    
    await query(
      `INSERT INTO daily_usage_summary (date, total_tokens, total_cost)
       VALUES ($1, $2, $3)
       ON CONFLICT (date)
       DO UPDATE SET
         total_tokens = daily_usage_summary.total_tokens + $2,
         total_cost = daily_usage_summary.total_cost + $3`,
      [today, t_tokens, cost]
    )
    
    return { cost, tokens: t_tokens }

  } catch (e) {
    console.error('Error logging usage:', e)
    return { cost: 0, tokens: 0 }
  }
}

async function analyzeImage(dataUrl, prompt) {
  // Ensure we have a valid Data URL (Base64) to avoid redirect (302) issues with Groq/OpenAI
  let finalImageUrl = dataUrl
  if (dataUrl && !isDataUrl(dataUrl)) {
    try {
      // console.log('Fetching image to convert to Base64...', dataUrl)
      const { buf, mime } = await fetchBuffer(dataUrl)
      
      // Convert to PNG using Sharp to ensure compatibility (e.g. WebP stickers)
      let finalBuf = buf
      let finalMime = mime
      try {
          finalBuf = await sharp(buf).png().toBuffer()
          finalMime = 'image/png'
      } catch (sharpErr) {
          console.warn('Sharp conversion failed, using original buffer:', sharpErr.message)
      }

      const b64 = finalBuf.toString('base64')
      finalImageUrl = `data:${finalMime};base64,${b64}`
    } catch (e) {
      console.error('Error converting image to Base64:', e.message)
      // Fallback to original URL if fetch fails
    }
  }

  const provider = process.env.AI_PROVIDER || 'openai'
  
  if (provider === 'groq') {
    if (!process.env.GROQ_API_KEY) {
      console.error('Groq API Key missing')
      return ''
    }
    try {
      const groq = new Groq({ apiKey: process.env.GROQ_API_KEY })
      const r = await groq.chat.completions.create({
        model: 'meta-llama/llama-4-scout-17b-16e-instruct',
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: prompt },
              { type: 'image_url', image_url: { url: finalImageUrl } }
            ]
          }
        ],
        max_tokens: 300
      })
      const c = r.choices?.[0]?.message?.content || ''
      
      // Log usage
      let usageData = { cost: 0, tokens: 0 }
      if (r.usage) {
        console.log('Groq Usage:', JSON.stringify(r.usage))
        usageData = await logUsage('meta-llama/llama-4-scout-17b-16e-instruct', r.usage, prompt) || { cost: 0, tokens: 0 }
      }

      return { content: trimOrEmpty(c), ...usageData }
    } catch (err) {
      console.error('\x1b[31m%s\x1b[0m', `Groq Image analysis failed: ${err.message}`)
      return { content: '', cost: 0, tokens: 0 }
    }
  } else {
    // Default to OpenAI
    if (!process.env.OPENAI_API_KEY) return { content: '', cost: 0, tokens: 0 }
    try {
      const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
      const r = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: prompt },
              { type: 'image_url', image_url: { url: finalImageUrl, detail: 'low' } }
            ]
          }
        ],
        max_tokens: 300
      })
      const c = r.choices?.[0]?.message?.content || ''
      
      // Log usage
      let usageData = { cost: 0, tokens: 0 }
      if (r.usage) {
        console.log('OpenAI Usage:', JSON.stringify(r.usage))
        usageData = await logUsage('gpt-4o-mini', r.usage, prompt) || { cost: 0, tokens: 0 }
      }

      return { content: trimOrEmpty(c), ...usageData }
    } catch (err) {
      console.error('\x1b[31m%s\x1b[0m', `OpenAI Image analysis failed: ${err.message}`)
      return { content: '', cost: 0, tokens: 0 }
    }
  }
}

function getExtensionFromMime(mime) {
  if (typeof mime !== 'string') return 'mp3'
  if (mime.includes('opus')) return 'ogg'
  if (mime.includes('ogg')) return 'ogg'
  if (mime.includes('webm')) return 'webm'
  if (mime.includes('wav')) return 'wav'
  if (mime.includes('mp4') || mime.includes('m4a')) return 'm4a'
  if (mime.includes('mpeg') || mime.includes('mp3')) return 'mp3'
  return 'mp3'
}

async function transcribeAudio(dataUrl) {
  // console.log('Starting audio transcription for URL:', dataUrl ? dataUrl.substring(0, 50) + '...' : 'null')
  try {
    const { buf, mime } = await fetchBuffer(dataUrl)
    // console.log('Audio buffer fetched: MIME type', mime, 'size', buf.length)
    
    const ext = getExtensionFromMime(mime)
    const filename = `audio.${ext}`
    const file = new File([buf], filename, { type: mime })
    
    const provider = process.env.AI_PROVIDER || 'openai'
    
    if (provider === 'groq') {
      if (!process.env.GROQ_API_KEY) {
        console.error('\x1b[31m%s\x1b[0m', 'Groq API key not found for audio transcription')
        return { content: '', cost: 0, tokens: 0 }
      }
      
      const groq = new Groq({ apiKey: process.env.GROQ_API_KEY })
      // console.log('Sending audio to Groq Whisper API for transcription...', filename)
      const r = await groq.audio.transcriptions.create({ 
        model: 'whisper-large-v3-turbo', 
        file,
        response_format: 'verbose_json'
      })
      const text = r.text || ''
      const duration = r.duration || 0
      
      // Log usage (Audio: total_tokens stores DURATION in seconds for cost calculation)
      const usageData = await logUsage('whisper-large-v3-turbo', { prompt_tokens: 0, completion_tokens: 0, total_tokens: duration }, 'AUDIO TRANSCRIPTION') || { cost: 0, tokens: 0 }
  
      // console.log('Audio transcription result:', text)
      return { content: trimOrEmpty(text), ...usageData }
    } else {
      // OpenAI
      if (!process.env.OPENAI_API_KEY) {
        console.error('\x1b[31m%s\x1b[0m', 'OpenAI API key not found for audio transcription')
        return { content: '', cost: 0, tokens: 0 }
      }
      
      const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
      // console.log('Sending audio to Whisper API for transcription...', filename)
      const r = await openai.audio.transcriptions.create({ model: 'whisper-1', file })
      const text = r.text || ''
      
      // Log usage (OpenAI Audio - also usually 0 tokens in response, but let's log it)
      const usageData = await logUsage('whisper-1', { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }, 'AUDIO TRANSCRIPTION') || { cost: 0, tokens: 0 }

      return { content: trimOrEmpty(text), ...usageData }
    }

  } catch (err) {
    console.error('\x1b[31m%s\x1b[0m', `Audio transcription failed: ${err.message}`)
    return { content: '', cost: 0, tokens: 0 }
  }
}

function buildReplyContext(body) {
  const m0 = get(body, 'conversation.messages.0', {})
  const v =
    get(body, 'content_attributes.in_reply_to') ||
    get(m0, 'content_attributes.in_reply_to') ||
    get(body, 'content_attributes.in_reply_to_external_id') ||
    get(m0, 'content_attributes.in_reply_to_external_id') ||
    get(m0, 'additional_attributes.context.message_id') ||
    get(m0, 'additional_attributes.quoted_message_id') ||
    ''
  return v
}

function getQuotedContent(body) {
  const m0 = get(body, 'conversation.messages.0', {})
  
  // 1. Chatwoot / Estructuras planas
  const qBody = get(body, 'content_attributes.quoted_content_body') || get(m0, 'content_attributes.quoted_content_body')
  if (qBody) return qBody

  // 2. Waha / Baileys (contextInfo)
  // A veces viene en 'additional_attributes.context' o directamente en el mensaje
  const context = get(m0, 'additional_attributes.context') || get(m0, 'context_info')
  if (context && context.quotedMessage) {
    const qm = context.quotedMessage
    return (
      qm.conversation ||
      qm.extendedTextMessage?.text ||
      qm.imageMessage?.caption ||
      qm.videoMessage?.caption ||
      ''
    )
  }
  
  return ''
}

function classify(body) {
  const attachRaw = get(body, 'conversation.messages.0.attachments', null)
  const attachments = Array.isArray(attachRaw) ? attachRaw : []
  const a0 = attachments[0]
  const fileType = a0?.file_type
  const content = trimOrEmpty(get(body, 'content', '')) || trimOrEmpty(get(body, 'conversation.messages.0.content', ''))
  const imageText = fileType === 'image' && content !== ''
  const imageOnly = fileType === 'image'
  const audioOnly = fileType === 'audio'
  const textOnly = attachments.length === 0
  if (imageText) return 'IMAGE-TEXT'
  if (imageOnly) return 'IMAGEN'
  if (audioOnly) return 'AUDIO'
  if (textOnly) return 'TEXT'
  return 'TEXT'
}

function ensureJid(body) {
  const direct = trimOrEmpty(get(body, 'sender.custom_attributes.waha_whatsapp_jid', ''))
  if (direct) return direct
  const phone = trimOrEmpty(get(body, 'sender.phone_number', ''))
  const digits = phone.replace(/\D+/g, '')
  return digits ? `${digits}@c.us` : ''
}

function shouldUseBot(body) {
  const incoming = trimOrEmpty(get(body, 'message_type')).toLowerCase() === 'incoming'
  const assigneeEmpty = isEmptyNumber(get(body, 'conversation.messages.0.conversation.assignee_id'))
  const notIntegration = get(body, 'sender.identifier') !== 'whatsapp.integration'
  return incoming && assigneeEmpty && notIntegration
}

function filterFlags(body) {
  const incoming = get(body, 'message_type') === 'incoming'
  const assigneeEmpty = isEmptyNumber(get(body, 'conversation.messages.0.conversation.assignee_id'))
  const senderId = get(body, 'sender.identifier')
  const notIntegration = senderId !== 'whatsapp.integration'
  const jid = ensureJid(body)
  const hasJid = jid !== ''
  return { incoming, assigneeEmpty, notIntegration, hasJid, senderId, jid }
}

function buildMasterText(text, audio, image, imagenText, replyText) {
  // Mantener formato del flujo original: espacios dobles si faltan partes
  const parts = [text || '', audio || '', image || '', imagenText || '']
  let joined = parts.join(' ')
  if (replyText) {
    joined = `[REPLY_TO: ${replyText}] ${joined}`
  }
  return joined
}

function nowIsoWithOffset() {
  const d = new Date()
  const pad = (n, l = 2) => String(n).padStart(l, '0')
  const y = d.getFullYear()
  const mo = pad(d.getMonth() + 1)
  const da = pad(d.getDate())
  const h = pad(d.getHours())
  const mi = pad(d.getMinutes())
  const s = pad(d.getSeconds())
  const ms = pad(d.getMilliseconds(), 3)
  const offMin = -d.getTimezoneOffset()
  const sign = offMin >= 0 ? '+' : '-'
  const abs = Math.abs(offMin)
  const oh = pad(Math.floor(abs / 60))
  const om = pad(abs % 60)
  return `${y}-${mo}-${da}T${h}:${mi}:${s}.${ms}${sign}${oh}:${om}`
}

async function cleanRedisBuffer(key, maxKeep) {
  const list = (await safeLRange(key, 0, -1)) || []
  const cleaned = []
  for (const s of list) {
    let o = null
    try {
      o = JSON.parse(s || '{}')
    } catch (_) {
      o = null
    }
    if (!o || typeof o !== 'object') continue
    const mt = collapseSpaces(trimOrEmpty(o.mastertext || ''))
    if (!mt) continue
    const obj = { ...o, mastertext: mt }
    cleaned.push(JSON.stringify(obj))
  }
  const start = cleaned.length > maxKeep ? cleaned.length - maxKeep : 0
  const sliced = cleaned.slice(start)
  let changed = sliced.length !== list.length
  if (!changed) {
    for (let i = 0; i < list.length; i++) {
      if (list[i] !== sliced[i]) { changed = true; break }
    }
  }
  if (changed) {
    await safeDel(key)
    for (const s of sliced) {
      await safeRPush(key, s)
    }
  }
  const final = await safeLRange(key, 0, -1)
  return final || []
}

const pendingRequests = {} // { jid: { res, timer } }

function buildMessageData(mastertext, messageId, replyContextId, replyContextText, body, queryQ, cost, tokens) {
  return {
    mastertext,
    cost: cost || 0,
    tokens: tokens || 0,
    timestamp: nowIsoWithOffset(),
    id: messageId,
    reply_to_id: replyContextId,
    reply_to_text: replyContextText,
    waha_whatsapp_jid: get(body, 'sender.custom_attributes.waha_whatsapp_jid'),
    conversation_id: get(body, 'conversation.messages.0.conversation_id'),
    _raw_conversation: get(body, 'conversation.messages.0', {}),
    _query: queryQ
  }
}

app.post(PATH, async (req, res) => {
  // console.log('--- INCOMING REQUEST ---')
  // console.log('Headers:', JSON.stringify(req.headers, null, 2))
  
  if (parseInt(req.headers['content-length'] || '0') === 0) {
    console.error('\x1b[31m%s\x1b[0m', '⚠️ WARNING: Request body is EMPTY (Content-Length: 0). Check n8n/sender configuration.')
  }

  try {
    // console.log('Body Preview:', JSON.stringify(req.body).substring(0, 500))
  } catch (_) {}

  const payload = req.body || {}
  let body
  if (Array.isArray(payload)) {
    const item = payload[0] || {}
    body = item.body && typeof item.body === 'object' ? item.body : item
  } else {
    body = payload.body && typeof payload.body === 'object' ? payload.body : payload
  }

  // --- FILTER MESSAGES ---
  const senderType = get(body, 'sender.type') || get(body, 'sender_type')
  const messageType = get(body, 'message_type')
  const senderIdentifier = get(body, 'sender.identifier')
  // Buscamos assignee_id en varios lugares por si acaso
  const assigneeId = get(body, 'conversation.assignee_id') || get(body, 'conversation.messages.0.conversation.assignee_id')
  
  // CONDICIONES PARA PROCESAR (Si no se cumplen, se ignora):
  // 1. message_type == 'incoming'
  // 2. assignee_id is empty
  // 3. sender.identifier != 'whatsapp.integration'
  
  if (
    messageType !== 'incoming' ||
    assigneeId || // Si tiene assignee, ignorar
    senderIdentifier === 'whatsapp.integration' ||
    // Filtros legacy/extra por seguridad
    senderType === 'Bot' || 
    senderType === 'agent_bot' || 
    senderType === 'AgentBot'
  ) {
    // console.log(`🚫 Ignoring message. Type: ${messageType}, Assignee: ${assigneeId}, Sender: ${senderIdentifier}`)
    return res.status(200).json({ status: 'ignored', reason: 'filtered_policy' })
  }
  // ---------------------------

  // console.log('Extracted Body keys:', body ? Object.keys(body) : 'body is null/undefined')
  
  const messageId = get(body, 'id', null)
  const replyContextId = buildReplyContext(body)
  const replyContextText = getQuotedContent(body)
  let type = classify(body)
  // console.log('Classified Type:', type)
  if (replyContextId) {} // console.log('Reply Context ID:', replyContextId)
  if (replyContextText) {} // console.log('Reply Context Text:', replyContextText)

  // --- ITERATE ALL ATTACHMENTS ---
  const attachRaw = get(body, 'conversation.messages.0.attachments', null)
  const attachments = Array.isArray(attachRaw) ? attachRaw : []

  let image = ''
  let text = ''
  const rawContent = get(body, 'content')
  if (typeof rawContent === 'string') {
    text = rawContent.trim()
  } else if (rawContent !== undefined && rawContent !== null) {
    try { text = String(rawContent).trim() } catch (_) { text = '' }
  }
  if (!text) {
    const raw2 = get(body, 'conversation.messages.0.content')
    if (typeof raw2 === 'string') text = raw2.trim()
  }
  let audio = ''
  let imagenText = ''

  let combinedImages = []
  let combinedAudios = []
  let totalCost = 0
  let totalTokens = 0

  if (attachments.length > 0) {
      for (const att of attachments) {
          const fType = att.file_type
          const dUrl = att.data_url
          if (!dUrl) continue

          if (fType === 'image') {
              const result = await analyzeImage(dUrl, '¿Analiza esta imagen profundamente?')
              if (result && result.content) {
                  combinedImages.push(result.content)
                  totalCost += (result.cost || 0)
                  totalTokens += (result.tokens || 0)
              }
          } else if (fType === 'audio') {
              const result = await transcribeAudio(dUrl)
              if (result && result.content) {
                  combinedAudios.push(result.content)
                  totalCost += (result.cost || 0)
                  totalTokens += (result.tokens || 0)
              }
          }
      }
  }

  // Construct final strings
  if (combinedImages.length > 0) {
      // If text exists, treat as IMAGE-TEXT, else IMAGEN
      // But actually we just append all analyses
      const joinedImages = combinedImages.join('\n---\n')
      if (text) {
          // IMAGE-TEXT scenario
          imagenText = `IMAGE: ${joinedImages} IMAGEN-FOOTER:${text}`
          type = 'IMAGE-TEXT' // force update type for logging if needed
      } else {
          image = `IMAGE: ${joinedImages}`
          type = 'IMAGEN'
      }
  }
  
  if (combinedAudios.length > 0) {
      audio = `AUDIO: ${combinedAudios.join('\n---\n')}`
      if (!image && !imagenText) type = 'AUDIO'
  }

  if (!image && !imagenText && !audio && text) {
      type = 'TEXT'
  }
  
  // Fallback if type was classified but no attachments found (weird edge case)
  // or if we need to preserve original single-attachment logic for compatibility?
  // The loop above handles single attachment too (length=1).
  
  const mastertextRaw = buildMasterText(text, audio, image, imagenText, replyContextText)
  const mastertext = collapseSpaces(mastertextRaw)

  try {
    let inputContent = ''
    // Store all dataUrls or text. For logs, we might want to show multiple images?
    // For now, let's store the text + first image/audio URL or a summary
    
    if (attachments.length > 0) {
        // Create a summary or JSON of inputs?
        // Or just comma separated URLs?
        const urls = attachments.map(a => a.data_url).filter(Boolean)
        if (urls.length > 0) {
             // If multiple, maybe JSON array?
             // For backward compatibility with frontend (which expects string or dataUrl),
             // let's stick to one input_content field.
             // If multiple images, we can put them in a JSON array string.
             if (urls.length === 1) inputContent = urls[0]
             else inputContent = JSON.stringify(urls)
        } else {
            inputContent = text
        }
    } else {
        inputContent = text
    }

    if (text && attachments.length > 0) {
         // Mixed content (Text + Image)
         // If we have a single image, inputContent is that image URL.
         // But we also have text. The logs table only has one input_content column.
         // We might lose the text visibility in the 'Input' column if we only show image.
         // But the frontend 'LogsViewer' shows text if type is TEXT.
         // Let's try to combine if possible or prioritize image for display.
         // The user wants "que salga todo el output". 'output_content' is mastertext which has everything.
         // 'input_content' is for the raw input.
         
         // Let's just keep inputContent as the media URL(s). The text is usually in the footer or separate.
    }

    // Limit input content length for logs if it's a huge data URL
    if (inputContent && inputContent.length > 1000) {
       inputContent = inputContent.substring(0, 1000) + '... (truncated)'
    }

    await query(
      'INSERT INTO message_logs (type, input_content, output_content, cost, tokens) VALUES ($1, $2, $3, $4, $5)',
      [type, inputContent, mastertext, totalCost, totalTokens]
    )
  } catch (e) {
    console.error('Error saving message log:', e)
  }

  const jid = ensureJid(body)
  // console.log('JID:', jid)
  const key = `${jid}_buffer`

  const qFromPayload = Array.isArray(payload) ? get(payload[0], 'query.q') : get(payload, 'query.q')
  const queryQ = req.query.q || qFromPayload || get(body, 'query.q') || 'gamersx8gmailcom-bot'

  // Check if query parameter 'flush' is present to clear the buffer
  const flush = req.query.flush || get(body, 'query.flush') || get(body, 'flush')
  if (flush === 'true' || flush === true) {
    // console.log(`Flushing buffer for key: ${key}`)
    await safeDel(key)
  }

  if (mastertext) {
    // console.log('Mastertext generated:', mastertext)
    
    // --- GLOBAL DEDUPLICATION ---
    // Verificar si el ID ya fue procesado recientemente (fuera del buffer actual)
    if (messageId) {
      const processedKey = `processed:${messageId}`
      const alreadyProcessed = await redis.get(processedKey)
      if (alreadyProcessed) {
        // console.log(`♻️ Skipping globally processed message ID: ${messageId}`)
        // Respondemos "éxito" falso al debounce para que no se quede colgado esperando
        // Aunque en realidad, si es duplicado, simplemente no lo añadimos al buffer.
        // El debounce se encargará de devolver lo que haya (o nada).
        
        // Si NO añadimos nada al buffer, y era el único mensaje, el timeout devolverá lista vacía.
        // Esto es correcto para un duplicado.
      } else {
        // Marcar como procesado con TTL de 1 hora
        await redis.set(processedKey, '1', { EX: 3600 })
        
        // --- BUFFER ADDITION ---
        // Verificar duplicados EN EL BUFFER ACTUAL (por si llegan varios iguales en la misma ráfaga)
        const currentList = await safeLRange(key, 0, -1)
        let isDuplicate = false
        
        if (messageId) {
          for (const item of currentList) {
            try {
              const parsed = JSON.parse(item)
              if (String(parsed.id) === String(messageId)) {
                isDuplicate = true
                break
              }
            } catch (_) {}
          }
        }

        if (!isDuplicate) {
          const messageData = buildMessageData(mastertext, messageId, replyContextId, replyContextText, body, queryQ, totalCost, totalTokens)
          // console.log('Adding new message to Redis:', JSON.stringify(messageData))
          await safeRPush(key, JSON.stringify(messageData))
          const maxKeepRaw = Number(process.env.REDIS_MAX_BUFFER || '20')
          const maxKeep = Number.isFinite(maxKeepRaw) && maxKeepRaw >= 1 ? maxKeepRaw : 20
          const curLen = await safeLLen(key)
          if (curLen > maxKeep) {
            await safeLTrim(key, curLen - maxKeep, -1)
          }
        } else {
          // console.log('⚠️ Skipping duplicate message in buffer with ID:', messageId)
        }
      }
    } else {
      // Si no tiene ID (es null), verificamos por contenido exacto en el buffer
      const currentList = await safeLRange(key, 0, -1)
      let isDuplicate = false
      if (currentList.length > 0) {
        try {
          const lastItem = JSON.parse(currentList[currentList.length - 1])
          if (lastItem.mastertext === mastertext) {
              isDuplicate = true
           }
        } catch (_) {}
      }
      
      if (!isDuplicate) {
          const messageData = buildMessageData(mastertext, messageId, replyContextId, replyContextText, body, queryQ, totalCost, totalTokens)
          // console.log('Adding new message to Redis (no-ID):', JSON.stringify(messageData))
          await safeRPush(key, JSON.stringify(messageData))
      }
    }
  } else {
  // console.log('⚠️ Mastertext is empty. Nothing to add to Redis. Type:', type)
}

  // --- DEBOUNCE / WAIT LOGIC ---
  // Cancelar temporizador anterior y responder vacio a la peticion previa
  if (pendingRequests[jid]) {
    // console.log(`Canceling previous request for JID ${jid} to debounce...`)
    clearTimeout(pendingRequests[jid].timer)
    
    // Verificar si la respuesta anterior aún es escribible antes de intentar responder
    const prevRes = pendingRequests[jid].res
    if (prevRes && !prevRes.headersSent && !prevRes.writableEnded) {
      try {
        prevRes.status(200).send('OK')
      } catch (e) {
        console.error('\x1b[31m%s\x1b[0m', `Error responding to cancelled request: ${e.message}`)
      }
    }
    delete pendingRequests[jid]
  }

  let debounceSeconds = Number(process.env.DEBOUNCE_SECONDS || '4')
  // Rich media takes longer to produce/send multiple. Extend wait time for audio/image.
  if (type === 'AUDIO' || type === 'IMAGEN' || type === 'IMAGE-TEXT') {
     debounceSeconds = Math.max(debounceSeconds, 10)
  }
  const debounceMs = debounceSeconds * 1000
  // console.log(`Waiting ${debounceSeconds}s for more messages from ${jid}...`)
  
  pendingRequests[jid] = {
    res,
    timer: setTimeout(async () => {
      // Verificar si esta petición sigue viva antes de procesar nada
      if (res.headersSent || res.writableEnded) {
        // console.log(`⚠️ Request for ${jid} already handled or closed. Skipping buffer processing.`)
        delete pendingRequests[jid]
        return
      }

      // console.log(`Timeout reached for ${jid}. Processing buffer...`)
      
      // Limpieza y obtencion del buffer final
      const maxKeep2Raw = Number(process.env.REDIS_MAX_BUFFER || '20')
      const maxKeep2 = Number.isFinite(maxKeep2Raw) && maxKeep2Raw >= 1 ? maxKeep2Raw : 20
      
      let list = await cleanRedisBuffer(key, maxKeep2)
      
      // Filtro final en memoria para eliminar duplicados por ID
      const uniqueMap = new Map()
      const uniqueList = []
      for (const itemStr of list) {
        try {
          const item = JSON.parse(itemStr)
          if (item.id) {
            if (!uniqueMap.has(String(item.id))) {
              uniqueMap.set(String(item.id), true)
              uniqueList.push(itemStr)
            }
          } else {
            uniqueList.push(itemStr)
          }
        } catch (_) {
          uniqueList.push(itemStr)
        }
      }
      list = uniqueList
      
      // Consumir (borrar) el buffer SIEMPRE para evitar duplicados en la siguiente llamada
      // console.log(`Clearing buffer for key: ${key} (auto-consume)`)
      await safeDel(key)
      delete pendingRequests[jid]

      // Doble chequeo final antes de enviar
      if (!res.headersSent && !res.writableEnded) {
        // console.log(`Processing ${list.length} messages for forwarding:`, JSON.stringify(list))

        if (list.length === 0) {
             // console.log('⚠️ List is empty. Skipping forward to EliteSeller Bot.')
             return res.status(200).send('OK')
        }
        
        // Extraemos los metadatos de conversación del último mensaje (el más reciente)
        // o de cualquiera, asumiendo que es la misma conversación.
        let conversationMeta = {}
        let rootQuery = null
        let rootWahaJid = null
        let rootConvId = null

        if (list.length > 0) {
            try {
                const lastItem = JSON.parse(list[list.length - 1])
                if (lastItem._raw_conversation) {
                    conversationMeta = lastItem._raw_conversation
                }
                rootQuery = lastItem._query
                rootWahaJid = lastItem.waha_whatsapp_jid
                rootConvId = lastItem.conversation_id
            } catch (_) {}
        }

        // Limpiamos el campo auxiliar _raw_conversation de la lista final para que el string JSON quede limpio
        const cleanedList = list.map(itemStr => {
             try {
                 const item = JSON.parse(itemStr)
                 delete item._raw_conversation
                 delete item._query
                 delete item.waha_whatsapp_jid
                 delete item.conversation_id
                 return JSON.stringify(item)
             } catch (_) {
                 return itemStr
             }
        })

        const payload = [
          {
            ...conversationMeta,
            query: { q: rootQuery },
            waha_whatsapp_jid: rootWahaJid,
            conversation_id: rootConvId,
            input: cleanedList || []
          }
        ]

        // Forward to EliteSeller Bot
        const targetQ = rootQuery || 'gamersx8gmailcom-bot'
        const webhookBaseUrl = process.env.WEBHOOK_URL || 'https://bot.eliteseller.app/webhook-test/41728f37-7ad6-4ca3-bba1-b046e05a112a'
        const targetUrl = `${webhookBaseUrl}?q=${targetQ}`
        
        try {
            // console.log(`Forwarding payload to: ${targetUrl}`)
            // Using global fetch (Node 18+)
            const forwardRes = await fetch(targetUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            // console.log(`Forwarding response status: ${forwardRes.status}`)
        } catch (err) {
            console.error('\x1b[31m%s\x1b[0m', `Error forwarding to EliteSeller: ${err.message}`)
        }

        return res.status(200).send('OK')
      }
    }, debounceMs)
  }
})

app.get('/api/logs', async (req, res) => {
  try {
    const result = await query('SELECT * FROM message_logs ORDER BY timestamp DESC LIMIT 50')
    res.json(result.rows)
  } catch (e) {
    res.status(500).json({ error: e.message })
  }
})

app.get('/api/usage', async (req, res) => {
  try {
    const result = await query('SELECT * FROM openai_usage ORDER BY timestamp DESC LIMIT 100')
    res.json(result.rows)
  } catch (e) {
    res.status(500).json({ error: e.message })
  }
})

app.get('/api/usage/daily', async (req, res) => {
  try {
    const result = await query('SELECT * FROM daily_usage_summary ORDER BY date DESC LIMIT 30')
    res.json(result.rows)
  } catch (e) {
    res.status(500).json({ error: e.message })
  }
})

app.get('/api/usage/stats', async (req, res) => {
  try {
    const result = await query(`
      SELECT 
        DATE(timestamp) as date, 
        model, 
        SUM(total_tokens) as total_tokens, 
        SUM(prompt_tokens) as prompt_tokens, 
        SUM(completion_tokens) as completion_tokens 
      FROM openai_usage 
      GROUP BY DATE(timestamp), model 
      ORDER BY date DESC
    `)
    res.json(result.rows)
  } catch (e) {
    res.status(500).json({ error: e.message })
  }
})

const port = process.env.PORT || 3000
app.listen(port, () => {
  console.log(`mini-back listening on port ${port}`)
})
async function safeRPush(key, value) {
  try {
    return await redis.rPush(key, value)
  } catch (_) {
    return 0
  }
}

async function safeLIndex(key, index) {
  try {
    return await redis.lIndex(key, index)
  } catch (_) {
    return null
  }
}

async function safeLRange(key, start, stop) {
  try {
    return await redis.lRange(key, start, stop)
  } catch (_) {
    return []
  }
}

async function safeLLen(key) {
  try {
    return await redis.lLen(key)
  } catch (_) {
    return 0
  }
}

async function safeLTrim(key, start, stop) {
  try {
    return await redis.lTrim(key, start, stop)
  } catch (_) {
    return 0
  }
}

async function safeDel(key) {
  try {
    return await redis.del(key)
  } catch (_) {
    return 0
  }
}
