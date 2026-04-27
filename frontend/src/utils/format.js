/**
 * Parse a backend datetime string as UTC.
 * Backend stores UTC via datetime.utcnow() but omits the Z suffix.
 */
function parseUTC(d) {
  if (!d) return null
  const s = String(d)
  return new Date(s.endsWith('Z') ? s : s + 'Z')
}

export function formatDate(d) {
  if (!d) return '-'
  return parseUTC(d).toLocaleString()
}

export function relativeTime(d) {
  if (!d) return ''
  const t = parseUTC(d)?.getTime()
  if (!t || isNaN(t)) return ''
  const sec = Math.round((Date.now() - t) / 1000)
  if (sec < 60) return `${sec}s ago`
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`
  return `${Math.floor(sec / 86400)}d ago`
}

export { parseUTC }
