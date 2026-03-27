export function formatDate(d) {
  if (!d) return '-'
  return new Date(d).toLocaleString()
}
