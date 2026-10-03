"use strict";
// Shared by the specimen canvas and synchronized experiment comparison views.
// Feature materialization is O(nonzeros) once per gene, never per painted cell.
class TissueIndex {
  constructor(points, measurement = null) {
    this.points = points;
    this.byID = new Map(points.map((p, i) => [p.id, i]));
    this.measurement = measurement;
    this.featureIndex = new Map((measurement?.featureIDs || []).map((g, i) => [g, i]));
    this.entityIndex = new Map((measurement?.entityIDs || []).map((g, i) => [g, i]));
    this.buffers = new Map();
    let xmin = Infinity, ymin = Infinity, xmax = -Infinity, ymax = -Infinity;
    for (const p of points) { xmin = Math.min(xmin, p.xy[0]); ymin = Math.min(ymin, p.xy[1]); xmax = Math.max(xmax, p.xy[0]); ymax = Math.max(ymax, p.xy[1]); }
    this.bounds = {xmin, ymin, dx: Math.max(1e-12, xmax - xmin), dy: Math.max(1e-12, ymax - ymin)};
    this.binSize = Math.max(this.bounds.dx, this.bounds.dy) / 128 || 1;
    this.bins = new Map();
    for (const p of points) { const key = this.key(p.xy[0], p.xy[1]); if (!this.bins.has(key)) this.bins.set(key, []); this.bins.get(key).push(p); }
  }
  key(x, y) { return `${Math.floor(x / this.binSize)},${Math.floor(y / this.binSize)}`; }
  feature(gene) {
    if (this.buffers.has(gene)) return this.buffers.get(gene);
    const values = new Float32Array(this.points.length); values.fill(NaN);
    const m = this.measurement, j = this.featureIndex.get(gene);
    if (m && j !== undefined) {
      for (let i = 0; i < this.points.length; i++) {
        const r = this.entityIndex.get(this.points[i].id); if (r === undefined) continue;
        if (m.absentValue === "zero") values[i] = 0;
        for (let k = m.rowOffsets[r]; k < m.rowOffsets[r + 1]; k++) if (m.featureIndices[k] === j) { values[i] = m.values[k]; break; }
      }
    }
    // Bounded LRU: keep selected features, not the whole dense transcriptome.
    if (this.buffers.size >= 8) this.buffers.delete(this.buffers.keys().next().value);
    this.buffers.set(gene, values); return values;
  }
  pick(x, y, radius) {
    let best = null, distance = radius;
    const loX = Math.floor((x - radius) / this.binSize), hiX = Math.floor((x + radius) / this.binSize);
    const loY = Math.floor((y - radius) / this.binSize), hiY = Math.floor((y + radius) / this.binSize);
    for (let bx = loX; bx <= hiX; bx++) for (let by = loY; by <= hiY; by++) {
      for (const p of this.bins.get(`${bx},${by}`) || []) { const d = Math.hypot(p.xy[0] - x, p.xy[1] - y); if (d <= distance) { best = p; distance = d; } }
    }
    return best;
  }
  visible(transform, width, height, pixelSize = 2) {
    const occupied = new Set(), result = [];
    for (const p of this.points) { const [x,y] = transform(p); if (x < 0 || y < 0 || x > width || y > height) continue;
      const bin = Math.floor(x / pixelSize) + Math.floor(y / pixelSize) * Math.ceil(width / pixelSize);
      if (occupied.has(bin)) continue; occupied.add(bin); result.push({p,x,y}); }
    return result;
  }
  static sharedScale(layers, residuals = []) {
    let min = 0, max = 1e-9, residual = 1e-9;
    for (const layer of layers) for (const v of layer) if (Number.isFinite(v)) { min = Math.min(min,v); max = Math.max(max,v); }
    for (const layer of residuals) for (const v of layer) if (Number.isFinite(v)) residual = Math.max(residual,Math.abs(v));
    return {min,max,residual};
  }
}
if (typeof module !== "undefined") module.exports = TissueIndex;
else window.TissueIndex = TissueIndex;
