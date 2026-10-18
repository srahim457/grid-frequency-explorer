/** UTC-only formatting helpers. Everything the UI shows is UTC, and says so. */
const PAD = (n, width = 2) => String(n).padStart(width, '0');

export function utc(ms, { seconds = true } = {}) {
  const d = new Date(ms);
  const date = `${d.getUTCFullYear()}-${PAD(d.getUTCMonth() + 1)}-${PAD(d.getUTCDate())}`;
  const time = `${PAD(d.getUTCHours())}:${PAD(d.getUTCMinutes())}`;
  return seconds ? `${date} ${time}:${PAD(d.getUTCSeconds())}` : `${date} ${time}`;
}

export function hz(value, digits = 3) {
  return value == null ? '--' : value.toFixed(digits);
}

export function count(value) {
  return value == null ? '--' : value.toLocaleString('en-GB');
}

/** 142 -> "2m 22s", 86400 -> "1d 0h". Durations here span seconds to weeks. */
export function duration(seconds) {
  if (seconds == null) return '--';
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  return `${Math.floor(seconds / 86400)}d ${Math.floor((seconds % 86400) / 3600)}h`;
}
