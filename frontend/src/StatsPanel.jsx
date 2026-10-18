import { count, hz } from './format';

/**
 * Exact numbers for the visible range, from /stats -- not from the
 * downsampled points the chart draws, which is the whole point of having a
 * separate endpoint.
 */
export default function StatsPanel({ stats, query, elapsedMs }) {
  const level = query?.level;
  const windowSeconds = query?.window_seconds;

  return (
    <div className="stats">
      <Stat label="min" value={hz(stats?.min)} unit="Hz" extreme />
      <Stat label="max" value={hz(stats?.max)} unit="Hz" />
      <Stat label="mean" value={hz(stats?.mean, 4)} unit="Hz" />
      <Stat label="readings" value={count(stats?.count)} />
      <div className="stat-meta">
        <span>
          tree level <b>{level ?? '--'}</b>
          {windowSeconds != null && <> · each slice = <b>2^{level} = {windowSeconds.toLocaleString()} s</b></>}
        </span>
        <span>
          <b>{stats?.nodes_touched ?? '--'}</b> nodes touched ·
          {' '}/stats in <b>{elapsedMs == null ? '--' : `${elapsedMs.toFixed(2)} ms`}</b>
        </span>
      </div>
    </div>
  );
}

function Stat({ label, value, unit, extreme = false }) {
  return (
    <div className={`stat${extreme ? ' is-extreme' : ''}`}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">
        {value}{unit && <span className="stat-unit">{unit}</span>}
      </span>
    </div>
  );
}
