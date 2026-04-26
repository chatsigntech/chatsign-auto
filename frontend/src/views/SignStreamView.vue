<template>
  <div class="sign-stream">
    <h2>Speech → Sign Language (test)</h2>

    <div class="row">
      <button @click="toggleMic" :class="{ active: listening }">
        {{ listening ? '⏹ stop listening' : '🎤 start listening (English)' }}
      </button>
      <span v-if="listening && interimText" class="interim">{{ interimText }}</span>
    </div>

    <div class="main-area">
      <div class="left">
        <h3>Transcript history</h3>
        <ul v-if="history.length" class="history">
          <li v-for="(h, i) in history" :key="i"
              :class="{ active: i === activeIdx }"
              @click="replay(i)">
            <div class="row1">
              <span class="time">{{ h.time }}</span>
              <span class="text">{{ h.text }}</span>
            </div>
            <div v-if="h.plan && h.plan.length" class="row2">
              <span v-for="(p, j) in h.plan" :key="j" :class="['gloss-tag', p.source]">
                {{ p.gloss }}<sup>{{ p.source[0] }}</sup>
              </span>
            </div>
          </li>
        </ul>
        <p v-else class="empty">click the mic and speak — finalized utterances appear here</p>

        <div v-if="status" class="status">{{ status }}</div>
      </div>

      <div class="right">
        <video ref="videoEl" autoplay muted playsinline class="video"></video>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onBeforeUnmount } from 'vue'

const history = ref([])           // [{time, text, plan}]
const activeIdx = ref(-1)
const interimText = ref('')
const status = ref('')
const listening = ref(false)
const videoEl = ref(null)

let recognition = null
let userStopped = false

let ws = null
let mediaSource = null
let sourceBuffer = null
let mediaUrl = null
let pendingChunks = []
let streaming = false

function nowHHMMSS() {
  const d = new Date()
  return `${d.getHours()}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`
}

function startRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition
  if (!SR) {
    status.value = 'Web Speech API not supported in this browser'
    return false
  }
  recognition = new SR()
  recognition.lang = 'en-US'
  recognition.interimResults = true
  recognition.continuous = true
  recognition.onresult = (e) => {
    let interim = ''
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i]
      if (r.isFinal) {
        const text = r[0].transcript.trim()
        if (text) onFinalUtterance(text)
      } else {
        interim += r[0].transcript
      }
    }
    interimText.value = interim
  }
  recognition.onerror = (e) => {
    if (e.error === 'no-speech' || e.error === 'aborted') return
    status.value = `mic error: ${e.error}`
  }
  recognition.onend = () => {
    if (!userStopped) {
      try { recognition.start() } catch (_) {}
    } else {
      listening.value = false
    }
  }
  try {
    recognition.start()
    listening.value = true
    userStopped = false
    status.value = 'listening...'
    return true
  } catch (e) {
    status.value = `start failed: ${e.message}`
    return false
  }
}

function toggleMic() {
  if (listening.value) {
    userStopped = true
    recognition && recognition.stop()
    return
  }
  startRecognition()
}

function onFinalUtterance(text) {
  history.value.push({ time: nowHHMMSS(), text, plan: [] })
  interimText.value = ''
  activeIdx.value = history.value.length - 1
  play(text, activeIdx.value)
}

function pumpBuffer() {
  if (!sourceBuffer || sourceBuffer.updating || pendingChunks.length === 0) return
  const total = pendingChunks.reduce((n, c) => n + c.byteLength, 0)
  const merged = new Uint8Array(total)
  let off = 0
  for (const c of pendingChunks) { merged.set(new Uint8Array(c), off); off += c.byteLength }
  pendingChunks = []
  try {
    sourceBuffer.appendBuffer(merged)
  } catch (e) {
    status.value = `appendBuffer failed: ${e.message}`
  }
}

function setupMediaSource() {
  return new Promise((resolve, reject) => {
    mediaSource = new MediaSource()
    mediaUrl = URL.createObjectURL(mediaSource)
    videoEl.value.src = mediaUrl
    mediaSource.addEventListener('sourceopen', () => {
      try {
        sourceBuffer = mediaSource.addSourceBuffer('video/mp4; codecs="avc1.42E01E"')
        sourceBuffer.addEventListener('updateend', pumpBuffer)
        resolve()
      } catch (e) {
        reject(e)
      }
    })
  })
}

async function play(text, histIdx = -1) {
  if (!text || streaming) {
    if (streaming) status.value = 'still streaming previous utterance — skipped'
    return
  }
  cleanupStream()
  streaming = true
  status.value = 'connecting...'

  try {
    await setupMediaSource()
  } catch (e) {
    status.value = `MSE init failed: ${e.message}`
    streaming = false
    return
  }

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  ws = new WebSocket(`${proto}//${location.host}/api/sign-stream/ws`)
  ws.binaryType = 'arraybuffer'
  ws.onopen = () => {
    ws.send(JSON.stringify({ text }))
    status.value = 'streaming...'
  }
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') {
      const msg = JSON.parse(ev.data)
      if (msg.error) { status.value = `error: ${msg.error}`; return }
      if (msg.plan && histIdx >= 0 && history.value[histIdx]) {
        history.value[histIdx].plan = msg.plan
      }
      if (msg.done) {
        status.value = 'stream complete'
        try { mediaSource && mediaSource.readyState === 'open' && mediaSource.endOfStream() } catch (_) {}
      }
      return
    }
    pendingChunks.push(ev.data)
    pumpBuffer()
  }
  ws.onerror = () => { status.value = 'ws error' }
  ws.onclose = () => { streaming = false }
}

function replay(i) {
  if (streaming) return
  activeIdx.value = i
  play(history.value[i].text, i)
}

function cleanupStream() {
  if (ws) { try { ws.close() } catch (_) {} ws = null }
  if (sourceBuffer) {
    try { sourceBuffer.removeEventListener('updateend', pumpBuffer) } catch (_) {}
    if (sourceBuffer.updating) { try { sourceBuffer.abort() } catch (_) {} }
    sourceBuffer = null
  }
  if (mediaSource && mediaSource.readyState === 'open') {
    try { mediaSource.endOfStream() } catch (_) {}
  }
  mediaSource = null
  if (mediaUrl) { URL.revokeObjectURL(mediaUrl); mediaUrl = null }
  pendingChunks = []
  streaming = false
}

onBeforeUnmount(() => {
  userStopped = true
  cleanupStream()
  recognition && recognition.stop()
})
</script>

<style scoped>
.sign-stream { padding: 20px; max-width: 1100px; margin: 0 auto; }
.row { display: flex; gap: 12px; align-items: center; margin: 12px 0; }
button { padding: 8px 16px; cursor: pointer; }
button.active { background: #f44; color: #fff; }
.interim { color: #888; font-style: italic; font-size: 14px; }
.main-area { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 12px; }
.left h3 { margin: 0 0 8px; font-size: 16px; }
.history { list-style: none; padding: 0; margin: 0; max-height: 360px; overflow-y: auto; border: 1px solid #ddd; border-radius: 6px; background: #fff; }
.history li { padding: 8px 10px; border-bottom: 1px solid #eee; cursor: pointer; color: #222; }
.history li:hover { background: #f5f5f5; }
.history li.active { background: #1976d2; color: #fff; }
.history li.active .time { color: #cfe3f7; }
.history li.active .gloss-tag { filter: brightness(0.9); }
.history .row1 { display: flex; gap: 10px; align-items: baseline; }
.history .row2 { margin-top: 4px; }
.history .time { color: #888; font-family: monospace; font-size: 12px; min-width: 60px; }
.history .text { flex: 1; }
.empty { color: #888; font-style: italic; font-size: 13px; }
.gloss-tag {
  display: inline-block; margin: 2px 4px; padding: 2px 8px;
  border-radius: 4px; font-size: 13px; background: #eee;
}
.gloss-tag.asl27k     { background: #d6e9f8; }
.gloss-tag.submission { background: #d3f9d8; }
.gloss-tag.generated  { background: #fff3bf; }
.gloss-tag.phase3     { background: #c5f0c0; }
.gloss-tag.letters    { background: #ffe0b2; }
.gloss-tag sup { color: #666; margin-left: 2px; }
.video { width: 100%; max-width: 576px; background: #000; border-radius: 6px; }
.status { color: #666; margin-top: 8px; font-family: monospace; font-size: 13px; }
</style>
