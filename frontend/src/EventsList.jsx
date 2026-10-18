import { duration, hz, utc } from './format';

/** Ten minutes of context on each side when zooming to an event. */
const PADDING_MS = 10 * 60 * 1000;

export default function EventsList({ events, onFocus }) {
  return (
    <section className="events">
      <div className="events-head">
        <span>excursions outside 49.5-50.5 Hz</span>
        <span>{events.length} found · click a row to zoom</span>
      </div>
      {events.length === 0 && <p className="empty">No excursions in this dataset.</p>}
      {events.map((event) => (
        <button
          key={event.id}
          type="button"
          className="event-row"
          onClick={() => onFocus([event.start - PADDING_MS, event.end + PADDING_MS])}
        >
          <span className="event-marker">▶</span>
          <span>{utc(event.start)} <span className="event-dim">UTC</span></span>
          <span>{duration(event.duration_s)}</span>
          <span className="event-dim">{event.direction.toUpperCase()}</span>
          <span className="event-extreme">{hz(event.extreme_hz)} Hz</span>
          <span className="event-dim">
            recovery {event.recovery_s == null ? '--' : `${event.recovery_s}s`}
            {' · '}RoCoF {hz(event.max_rocof_hz_per_s)} Hz/s
          </span>
          <span className="event-go">ZOOM →</span>
        </button>
      ))}
    </section>
  );
}
