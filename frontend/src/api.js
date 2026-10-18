/**
 * Thin client for backend/api.py.
 *
 * Every call takes an AbortSignal: the chart fires a new /query and /stats on
 * each zoom or pan, and a stale response arriving late would otherwise repaint
 * the chart with the wrong range.
 */
const BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

async function get(path, params, signal) {
  const url = new URL(path, BASE);
  for (const [key, value] of Object.entries(params ?? {})) {
    url.searchParams.set(key, String(value));
  }
  const response = await fetch(url, { signal });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* non-JSON error body; keep the status line */
    }
    throw new Error(`${path}: ${detail}`);
  }
  return response.json();
}

export const getMeta = (signal) => get('/meta', null, signal);
export const getEvents = (signal) => get('/events', null, signal);
export const getQuery = (start, end, width, signal) =>
  get('/query', { start, end, width }, signal);
export const getStats = (start, end, signal) =>
  get('/stats', { start, end }, signal);
