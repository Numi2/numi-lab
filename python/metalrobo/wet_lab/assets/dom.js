export function node(tag, text, cls) { const el = document.createElement(tag); if (text !== undefined)
    el.textContent = text; if (cls)
    el.className = cls; return el; }
export function button(text, action, cls) { const b = node('button', text, cls); b.type = 'button'; b.onclick = action; return b; }
export function fmt(value) { return typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString(undefined, { maximumSignificantDigits: 5 }) : 'Unavailable'; }
export function labeled(label, input) { const l = node('label'); l.append(node('span', label), input); return l; }
export function badge(text) { return node('span', text, 'badge'); }
export function notice(text) { const e = document.getElementById('notice'); e.textContent = text; e.hidden = false; }
export function errorMessage(error) { return error instanceof Error ? error.message : String(error); }
export function select(items, value, change) { const el = node('select'); for (const item of items) {
    const o = node('option', item.title || item.cellType || item.id);
    o.value = item.id;
    el.append(o);
} el.value = value; el.onchange = () => change(el.value); return el; }
export function detail(label, data) { const d = node('details'); d.append(node('summary', label), node('pre', JSON.stringify(data, null, 2))); return d; }
