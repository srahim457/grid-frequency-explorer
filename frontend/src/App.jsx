import { useCallback, useEffect, useState } from 'react';
import Chart from './Chart';
import EventsList from './EventsList';
import StatsPanel from './StatsPanel';
import { getEvents, getMeta, getQuery, getStats } from './api';
import { utc } from './format';

const DEBOUNCE_MS = 100;
/** Chart width in pixels, capped to the API's documented maximum. */
const clampWidth = (px) => Math.min(Math.max(Math.round(px), 1), 5000);

export default function App() {
  const [meta, setMeta] = useState(null);
  const [events, setEvents] = useState([]);
  const [view, setView] = useState(null);
  const [query, setQuery] = useState(null);
  const [stats, setStats] = useState(null);
  const [elapsedMs, setElapsedMs] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [focus, setFocus] = useState(null);
  const [chartWidth, setChartWidth] = useState(1200);

  // Initial load: the dataset's extent, and the events to shade.
  useEffect(() => {
    const controller = new AbortController();
    Promise.all([getMeta(controller.signal), getEvents(controller.signal)])
      .then(([metaResponse, eventList]) => {
        setMeta(metaResponse);
        setEvents(eventList);
        setView([metaResponse.start, metaResponse.end]);
      })
      .catch((err) => {
        if (err.name !== 'AbortError') setError(err.message);
      });
    return () => controller.abort();
  }, []);

  // Every zoom or pan refetches both endpoints, debounced, with the previous
  // pair cancelled so a slow response cannot overwrite a newer one.
  useEffect(() => {
    if (!view) return undefined;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setBusy(true);
      const [start, end] = view;
      try {
        const width = clampWidth(chartWidth);
        const [points, summary] = await Promise.all([
          getQuery(Math.round(start), Math.round(end), width, controller.signal),
          getStats(Math.round(start), Math.round(end), controller.signal),
        ]);
        setQuery(points);
        setStats(summary);
        setElapsedMs(summary.elapsed_ms);
        setError(null);
      } catch (err) {
        if (err.name !== 'AbortError') setError(err.message);
      } finally {
        if (!controller.signal.aborted) setBusy(false);
      }
    }, DEBOUNCE_MS);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [view, chartWidth]);

  const handleViewChange = useCallback((domain) => setView(domain), []);
  const handleFocus = useCallback((domain) => {
    setFocus({ domain, nonce: Date.now() });
  }, []);
  const resetZoom = () => meta && handleFocus([meta.start, meta.end]);

  if (error && !meta) {
    return (
      <div className="app">
        <h1 className="title">Grid Frequency Explorer</h1>
        <p className="error">
          {error}
          <br />
          Is the API running? <code>uvicorn backend.api:app --port 8000</code>
        </p>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1 className="title">Grid Frequency Explorer</h1>
          <p className="subtitle">
            {meta
              ? `GB system frequency · ${utc(meta.start, { seconds: false })} – ${utc(meta.end, { seconds: false })} UTC · ${meta.n_points.toLocaleString()} readings · ${meta.n_missing.toLocaleString()} missing`
              : 'connecting…'}
            {view && ` · viewing ${utc(view[0])} – ${utc(view[1])} UTC`}
          </p>
        </div>
        <StatsPanel stats={stats} query={query} elapsedMs={elapsedMs} />
      </header>

      <Chart
        meta={meta}
        data={query}
        events={events}
        view={view ?? [0, 1]}
        onViewChange={handleViewChange}
        focus={focus}
        busy={busy}
        onWidth={setChartWidth}
      />

      <div className="header" style={{ borderBottom: 'none', paddingBottom: 0 }}>
        <span className="subtitle">
          Band 49.5–50.5 Hz (NESO statutory limits, ±0.5 Hz of 50 Hz)
        </span>
        <button type="button" className="button" onClick={resetZoom}>
          reset to full month
        </button>
      </div>

      <EventsList events={events} onFocus={handleFocus} />
      {error && meta && <p className="error">{error}</p>}
    </div>
  );
}
