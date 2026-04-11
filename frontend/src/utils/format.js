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

export { parseUTC }
