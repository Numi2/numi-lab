export class ServiceError extends Error {
    status;
    details;
    constructor(message, status, details) {
        super(message);
        this.status = status;
        this.details = details;
    }
}
export class ServiceClient {
    token;
    pending = new Map();
    cache = new Map();
    timings = [];
    constructor(token) {
        this.token = token;
    }
    cancel(channel) { this.pending.get(channel)?.abort(); this.pending.delete(channel); }
    clear() { this.cache.clear(); for (const controller of this.pending.values())
        controller.abort(); this.pending.clear(); }
    async request(path, body, options = {}) {
        if (options.cacheKey && this.cache.has(options.cacheKey)) {
            const value = this.cache.get(options.cacheKey);
            this.timings.push({ path, ms: 0, bytes: 0, cached: true });
            return structuredClone(value);
        }
        const controller = new AbortController();
        if (options.channel) {
            this.cancel(options.channel);
            this.pending.set(options.channel, controller);
        }
        const start = performance.now();
        try {
            const response = await fetch('/api/' + path, { method: body === undefined ? 'GET' : 'POST', headers: { 'X-Wet-Lab-Token': this.token, 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal });
            const raw = await response.text();
            const result = JSON.parse(raw);
            this.timings.push({ path, ms: performance.now() - start, bytes: new TextEncoder().encode(raw).length, cached: false });
            if (this.timings.length > 300)
                this.timings.shift();
            if (!response.ok)
                throw new ServiceError(result.error || 'Request failed', response.status, result);
            if (controller.signal.aborted)
                throw new DOMException('Obsolete response', 'AbortError');
            if (options.cacheKey) {
                if (this.cache.size >= 32)
                    this.cache.delete(this.cache.keys().next().value);
                this.cache.set(options.cacheKey, result);
            }
            return result;
        }
        finally {
            if (options.channel && this.pending.get(options.channel) === controller)
                this.pending.delete(options.channel);
        }
    }
}
// Identity includes the full owner binding and access state. No preview survives a scientific-input edit.
export function identity(...parts) { return JSON.stringify(parts); }
