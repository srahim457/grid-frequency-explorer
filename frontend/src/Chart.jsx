import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

const MARGIN = { top: 16, right: 18, bottom: 56, left: 70 };
const HEIGHT = 420;
const REF_LINES = [
  { hz: 50.5, label: '50.5 statutory', kind: 'limit' },
  { hz: 50.0, label: '50.0 nominal', kind: 'nominal' },
  { hz: 49.5, label: '49.5 statutory', kind: 'limit' },
];

/** Zooming deeper than one second per pixel shows nothing new. */
function maxZoomScale(meta, innerWidth) {
  if (!meta || innerWidth <= 0) return 1;
  const seconds = (meta.end - meta.start) / 1000;
  return Math.max(1, seconds / innerWidth);
}

/**
 * The chart owns the zoom behaviour and reports the visible range upward;
 * App turns that into /query and /stats calls. Drawing is done imperatively
 * with D3 inside an effect, which keeps React out of the SVG's way.
 */
export default function Chart({
  meta, data, events, view, onViewChange, focus, busy, onWidth,
}) {
  const wrapRef = useRef(null);
  const svgRef = useRef(null);
  const zoomRef = useRef(null);
  const [width, setWidth] = useState(0);

  // The chart asks for one point per pixel, so it has to know its own width.
  useLayoutEffect(() => {
    const element = wrapRef.current;
    if (!element) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      setWidth(Math.floor(entry.contentRect.width));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const innerWidth = Math.max(width - MARGIN.left - MARGIN.right, 10);
  const innerHeight = HEIGHT - MARGIN.top - MARGIN.bottom;

  // Report the plot width upward so /query can ask for one point per pixel.
  useEffect(() => {
    if (innerWidth > 10) onWidth?.(innerWidth);
  }, [innerWidth, onWidth]);

  // --- zoom: attach once, report the rescaled domain upward ---------------
  useEffect(() => {
    if (!meta || innerWidth <= 10) return undefined;
    const baseX = d3.scaleUtc()
      .domain([meta.start, meta.end])
      .range([0, innerWidth]);

    const zoom = d3.zoom()
      .scaleExtent([1, maxZoomScale(meta, innerWidth)])
      .translateExtent([[0, 0], [innerWidth, innerHeight]])
      .extent([[0, 0], [innerWidth, innerHeight]])
      .on('zoom', (zoomEvent) => {
        const [start, end] = zoomEvent.transform.rescaleX(baseX).domain();
        onViewChange([start.getTime(), end.getTime()]);
      });

    zoomRef.current = { zoom, baseX };
    const svg = d3.select(svgRef.current);
    svg.call(zoom);
    return () => svg.on('.zoom', null);
  }, [meta, innerWidth, innerHeight, onViewChange]);

  // --- programmatic zoom when an event row is clicked ---------------------
  useEffect(() => {
    if (!focus || !zoomRef.current || innerWidth <= 10) return;
    const { zoom, baseX } = zoomRef.current;
    const [start, end] = focus.domain;
    const k = Math.min(
      innerWidth / Math.max(baseX(end) - baseX(start), 1),
      zoom.scaleExtent()[1],
    );
    const transform = d3.zoomIdentity.scale(k).translate(-baseX(start), 0);
    d3.select(svgRef.current)
      .transition()
      .duration(650)
      .call(zoom.transform, transform);
  }, [focus, innerWidth]);

  // --- draw ---------------------------------------------------------------
  useEffect(() => {
    if (!meta || innerWidth <= 10) return;
    const svg = d3.select(svgRef.current);
    const points = data?.points ?? [];
    const valid = points.filter((p) => p.count > 0);

    const x = d3.scaleUtc().domain(view).range([0, innerWidth]);

    // Y fits the visible data with a little padding, so zooming into quiet
    // stretches still shows their structure rather than a flat line.
    const lo = d3.min(valid, (p) => p.min);
    const hi = d3.max(valid, (p) => p.max);
    const pad = Math.max((hi - lo) * 0.12, 0.01);
    const y = d3.scaleLinear()
      .domain(valid.length ? [lo - pad, hi + pad] : [49.4, 50.6])
      .range([innerHeight, 0])
      .nice();

    const plot = svg.select('.plot');

    plot.select('.grid')
      .call(d3.axisLeft(y).tickSize(-innerWidth).tickFormat('').ticks(6))
      .call((g) => g.select('.domain').remove());

    plot.select('.x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).ticks(Math.max(Math.floor(innerWidth / 110), 2)));

    plot.select('.y-axis').call(d3.axisLeft(y).ticks(6));

    // Event spans, clipped to the visible range.
    const spans = (events ?? []).filter((e) => e.end > view[0] && e.start < view[1]);
    plot.select('.events').selectAll('rect')
      .data(spans, (d) => d.id)
      .join('rect')
      .attr('x', (d) => x(Math.max(d.start, view[0])))
      .attr('width', (d) => Math.max(
        x(Math.min(d.end, view[1])) - x(Math.max(d.start, view[0])), 1.5))
      .attr('y', 0)
      .attr('height', innerHeight)
      .attr('class', 'event-span');

    // Reference lines, drawn only when they fall inside the visible y range.
    const [yLo, yHi] = y.domain();
    plot.select('.refs').selectAll('g')
      .data(REF_LINES.filter((r) => r.hz >= yLo && r.hz <= yHi), (d) => d.hz)
      .join((enter) => {
        const g = enter.append('g');
        g.append('line');
        g.append('text');
        return g;
      })
      .call((g) => {
        g.attr('class', (d) => `ref ref-${d.kind}`)
          .attr('transform', (d) => `translate(0,${y(d.hz)})`);
        g.select('line').attr('x2', innerWidth);
        g.select('text')
          .attr('x', innerWidth - 4)
          .attr('y', -5)
          .text((d) => d.label);
      });

    // The data itself: a min-max band with the mean drawn through it.
    const defined = (p) => p.count > 0;
    const area = d3.area()
      .defined(defined)
      .x((p) => x(p.t))
      .y0((p) => y(p.min))
      .y1((p) => y(p.max));
    const line = d3.line()
      .defined(defined)
      .x((p) => x(p.t))
      .y((p) => y(p.mean));

    plot.select('.band').datum(points).attr('d', area);
    plot.select('.mean-glow').datum(points).attr('d', line);
    plot.select('.mean').datum(points).attr('d', line);
  }, [meta, data, events, view, innerWidth, innerHeight]);

  const windowSeconds = data?.window_seconds ?? 1;

  return (
    <div className="chart-wrap" ref={wrapRef}>
      <svg ref={svgRef} height={HEIGHT} role="img"
           aria-label="GB grid frequency over the visible time range, in hertz">
        <defs>
          {/* Bloom: the neon look without pushing the hues out of the
              validated lightness band. */}
          <filter id="bloom" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3.2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <clipPath id="plot-clip">
            <rect x="0" y="0" width={innerWidth} height={innerHeight} />
          </clipPath>
        </defs>
        <g className="plot" transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          <g className="grid" />
          <g clipPath="url(#plot-clip)">
            <g className="events" />
            <g className="refs" />
            <path className="band" />
            <path className="mean-glow" filter="url(#bloom)" />
            <path className="mean" />
          </g>
          <g className="axis x-axis" />
          <g className="axis y-axis" />
          <text
            className="axis-title"
            x={innerWidth / 2}
            y={innerHeight + 44}
            textAnchor="middle"
          >
            Time (UTC)
          </text>
          <text
            className="axis-title"
            transform={`translate(${-52},${innerHeight / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            Frequency (Hz)
          </text>
        </g>
      </svg>
      <span className="chart-hint">
        SCROLL TO ZOOM · DRAG TO PAN · EACH SLICE = {windowSeconds.toLocaleString()} s · TIMES UTC
      </span>
      {busy && <span className="chart-busy">querying</span>}
    </div>
  );
}
