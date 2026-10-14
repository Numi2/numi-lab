import { ServiceClient, identity } from './client.js';
import { AcknowledgedState, Layout } from './state.js';
import { node, button, select, labeled, notice, badge, fmt, detail, errorMessage } from './dom.js';
import { populationView, supportView } from './presentations/population.js';
import { SpatialView } from './presentations/spatial.js';
import { evidenceSummary, failureView } from './evidence.js';
const $ = (id) => document.getElementById(id);
const client = new ServiceClient(window.WET_LAB_TOKEN), state = new AcknowledgedState(client), layout = new Layout();
layout.update({});
let catalogs = [], online = false, lastRevision = -1, lastPending = false, lastProgress = '', renderedSelection = '', savedCamera = null, readEpoch = 0, metadataEpoch = 0, spatial = null, activeReadout = null;
let search = '', offset = 0, candidates = [], candidateTotal = 0, searchTimer, poll;
let localIntent = null;
const feedback = [];
let readinessAt = 0, metadataMs = 0;
const readoutTimings = [];
const cap = (c) => c.capabilities || c;
const catalog = () => catalogs.find(c => c.id === state.value?.selection?.assayID);
const activeDraft = () => state.value?.drafts.find(d => d.id === layout.value.activeDraft);
function run(task) { void task().catch(error => { if (error instanceof DOMException && error.name === 'AbortError')
    return; notice(errorMessage(error)); }); }
function drawer(title, content) { $('drawer-title').textContent = title; $('drawer-content').replaceChildren(content); if (!$('drawer').open)
    $('drawer').showModal(); layout.update({ drawer: title }); }
function showEvidence(data, title = 'Evidence and support') { drawer(title, evidenceSummary(data)); }
function closeDrawer() { $('drawer').close(); layout.update({ drawer: null }); }
function acknowledgeFeedback(event) { if (!event.isTrusted)
    return; const start = performance.now(); requestAnimationFrame(() => { feedback.push(performance.now() - start); if (feedback.length > 300)
    feedback.shift(); }); }
document.addEventListener('pointerdown', acknowledgeFeedback, true);
document.addEventListener('keydown', acknowledgeFeedback, true);
async function mutate(intent) { try {
    return await state.mutate(intent);
}
catch (error) {
    localIntent = intent;
    throw error;
} }
function setSelection(selection) { const d = activeDraft(); if (d?.status === 'draft' && d.selection.assayID === selection.assayID && (d.selection.populationID !== selection.populationID || d.selection.specimenID !== selection.specimenID))
    return mutate({ action: 'edit', id: d.id, selection, conditionIDs: [selection.conditionID] }); return mutate({ action: 'selection', selection }); }
async function openDataset(c) { const previous = state.value?.selection; const p = c.populations.find(p => p.id === previous?.populationID) || c.populations[0]; if (!p)
    throw new Error('No admitted populations. Complete the dataset preparation before opening it.'); layout.update({ activeDraft: null }); offset = 0; search = ''; await setSelection({ assayID: c.id, specimenID: p.specimenID || c.specimens[0].id, populationID: p.id, conditionID: p.conditionID || c.conditions[0].id, gene: c.features.includes(previous?.gene || '') ? previous.gene : (c.features.includes('FOS') ? 'FOS' : c.features[0]), experimentID: null }); }
async function openDraft(d) { layout.update({ activeDraft: d.id }); await setSelection({ ...d.selection, gene: state.value?.selection?.gene && catalogs.find(c => c.id === d.selection.assayID)?.features.includes(state.value.selection.gene) ? state.value.selection.gene : d.genes[0], target: d.targets[0], experimentID: d.record || null }); renderInspector(); renderComparison(); }
function renderNavigation() { const nav = $('navigation'); nav.replaceChildren(button('◈  Laboratory home', () => { $('start').hidden = false; $('investigation').hidden = true; renderStart(); })); nav.append(node('p', 'Datasets', 'nav-group')); for (const c of catalogs)
    nav.append(button(c.title, () => run(() => openDataset(c)))); nav.append(node('p', 'Experiments', 'nav-group')); for (const d of (state.value?.drafts || []).slice().reverse().slice(0, 8)) {
    const b = button(d.title, () => run(() => openDraft(d)));
    b.append(node('small', ' · ' + d.status));
    nav.append(b);
} nav.append(node('p', 'Workspace', 'nav-group'), button('Comparisons', () => { layout.update({ comparison: true }); $('investigation').hidden = false; $('start').hidden = true; renderComparison(); }), button('History', showHistory), button('Snapshot / restore', showStorage)); $('navigation').hidden = !layout.value.navigation; document.querySelector('.workbench').classList.toggle('collapsed', !layout.value.navigation); $('nav-toggle').setAttribute('aria-expanded', String(layout.value.navigation)); }
function renderStart() {
    const start = $('start');
    start.replaceChildren(node('div', 'YOUR RESEARCH WORKBENCH', 'eyebrow'), node('h1', 'Open a specimen. Ask a precise question.'), node('p', 'Inspect measured evidence or compose a supported experiment with Codex. Sealed predictions and observation access stay explicit.', 'muted'));
    const cards = node('div', undefined, 'cards');
    for (const c of catalogs) {
        const card = node('article', undefined, 'dataset-card');
        card.append(badge(cap(c).prediction ? '◇ EXPERIMENTAL MODEL' : '● MEASURED EXPLORATION'), node('h2', c.title), node('p', `${c.populations.length} populations · ${c.specimens.length} source contexts · ${(c.featuresTotal ?? c.features.length).toLocaleString()} feature identifiers`, 'muted'), node('p', cap(c).presentation === 'population' ? 'Population composition and molecular readouts' : 'Measured tissue geometry and population readouts'), button('Open specimen', () => run(() => openDataset(c)), 'primary'));
        cards.append(card);
    }
    start.append(cards);
    const interrupted = (state.value?.operations || []).filter(o => ['interrupted', 'failed', 'cancelled'].includes(o.status));
    if (interrupted.length) {
        start.append(node('h2', 'Interrupted work'));
        for (const op of interrupted) {
            const d = state.value.drafts.find(d => d.id === op.draftID);
            start.append(button((d?.title || op.draftID) + ' · ' + op.status, () => d && run(() => openDraft(d))));
        }
    }
    start.append(node('h2', 'Recent experiments'));
    for (const d of (state.value?.drafts || []).slice().reverse().slice(0, 6))
        start.append(button(d.title + ' · ' + d.status, () => run(() => openDraft(d))));
    start.append(node('p', 'Internal experimental software. Safari is unsupported until qualified. Physical touch and uncoached researcher usability remain pending.', 'muted'));
}
function renderToolbar(c, s) {
    const host = $('selection-toolbar');
    host.replaceChildren();
    const pops = c.populations.filter(p => !p.specimenID || p.specimenID === s.specimenID);
    host.append(labeled('Specimen / context', select(c.specimens, s.specimenID, value => run(async () => { const p = c.populations.find(p => p.specimenID === value) || c.populations[0]; await setSelection({ ...s, specimenID: value, populationID: p.id, conditionID: p.conditionID || c.conditions[0].id, experimentID: null }); }))), labeled('Population', select(pops, s.populationID, value => run(async () => { const p = c.populations.find(p => p.id === value); await setSelection({ ...s, populationID: value, conditionID: p.conditionID || s.conditionID, experimentID: null }); }))));
    const gene = node('input');
    gene.id = 'selected-gene';
    gene.value = s.gene || '';
    gene.setAttribute('list', 'features');
    gene.onchange = () => run(async () => { await setSelection({ ...s, gene: gene.value }); });
    const genes = node('datalist');
    genes.id = 'features';
    for (const g of c.features.slice(0, 12000)) {
        const o = node('option');
        o.value = g;
        genes.append(o);
    }
    const l = labeled('Molecular feature', gene);
    l.classList.add('gene');
    host.append(l, genes);
    if (cap(c).presentation === 'tissue')
        host.append(labeled('Response population', select([{ id: 'direct', title: 'Directly perturbed' }, { id: 'neighbor', title: 'Neighboring population' }], layout.value.role, value => { layout.update({ role: value }); run(loadReadout); })));
    host.append(labeled('Readout units', select(cap(c).prediction ? [{ id: 'normalized', title: 'log1p(CPM) · model units' }] : [{ id: 'raw_counts', title: 'Raw counts' }, { id: 'log1p_10000', title: 'log1p(10,000 · retained panel)' }], cap(c).prediction ? 'normalized' : layout.value.normalization, value => { layout.update({ normalization: value }); run(loadReadout); })), button(layout.value.comparison ? 'Close comparison' : 'Compare views', () => { layout.update({ comparison: !layout.value.comparison }); renderComparison(); }));
}
function renderHeading(c, s) { const p = c.populations.find(p => p.id === s.populationID); const h = $('view-heading'); h.replaceChildren(node('span', cap(c).presentation === 'tissue' ? 'SPATIAL SPECIMEN' : 'CELLULAR SPECIMEN', 'eyebrow'), node('h1', p?.title || p?.cellType || p?.id || s.populationID), badge(activeDraft()?.status.toUpperCase() || (cap(c).prediction ? 'PREVIEW · MODEL INFERENCE' : 'EXPLORE · MEASURED EVIDENCE')), node('p', (s.gene || 'Select a gene') + ' · ' + s.conditionID + ' · ' + c.title, 'muted')); $('dataset-name').textContent = c.title; }
function renderInspector() {
    const c = catalog(), s = state.value?.selection;
    if (!c || !s)
        return;
    const host = $('inspector');
    host.replaceChildren(node('div', 'SELECTION', 'eyebrow'), node('h2', c.populations.find(p => p.id === s.populationID)?.title || s.populationID), supportView(state.value?.selectionSupport));
    const support = state.value?.selectionSupport;
    if (support?.reason)
        host.append(node('p', support.reason, 'muted'));
    host.append(button('Inspect source and support', () => showEvidence({ ...support, ...s, limits: c.limits })), node('hr'));
    const d = activeDraft();
    if (!d) {
        host.append(node('h2', 'Define an experiment'), node('p', 'Reduce a specified RNA program in this population. The original intent stays visible if support is missing.', 'muted'));
        const gene = node('input');
        gene.id = 'objective-genes';
        gene.value = s.gene || '';
        host.append(labeled('Exact objective genes', gene), button('Create proposal', () => run(async () => { await mutate({ action: 'propose', genes: gene.value.split(/[\s,]+/).filter(Boolean), targets: s.target ? [s.target] : [], title: 'Reduce ' + gene.value + ' in ' + s.populationID }); const newest = state.value.drafts.at(-1); layout.update({ activeDraft: newest.id }); renderInspector(); }), 'primary'));
        renderCandidateSearch(host, c, s);
        return;
    }
    host.append(badge(d.status.toUpperCase()), node('h2', d.title), node('p', 'Exact objective: reduce mean expression of ' + d.genes.join(', '), 'muted'));
    const form = node('div', undefined, 'proposal-form');
    const genes = node('textarea');
    genes.id = 'proposal-genes';
    genes.value = (localIntent?.id === d.id && localIntent.genes ? localIntent.genes : d.genes).join(', ');
    genes.disabled = d.status !== 'draft' || state.pending;
    const title = node('input');
    title.value = d.title;
    title.disabled = genes.disabled;
    form.append(labeled('Proposal title', title), labeled('Exact objective genes', genes));
    const models = node('input');
    models.id = 'proposal-models';
    models.value = d.modelIDs.join(', ');
    models.disabled = genes.disabled;
    const conditions = node('input');
    conditions.id = 'proposal-conditions';
    conditions.value = d.conditionIDs.join(', ');
    conditions.disabled = genes.disabled;
    // Searchable identifiers remain editable to retain unsupported intent. Supported options are explicit below.
    const modelOptions = node('fieldset');
    modelOptions.append(node('legend', 'Models to compare'));
    for (const m of state.value.models.filter(m => m.presentation === cap(c).presentation)) {
        const check = node('input');
        check.type = 'checkbox';
        check.checked = d.modelIDs.includes(m.id);
        check.disabled = genes.disabled;
        check.onchange = () => { const chosen = new Set(models.value.split(',').map(v => v.trim()).filter(Boolean)); check.checked ? chosen.add(m.id) : chosen.delete(m.id); models.value = [...chosen].join(', '); };
        const label = node('label', undefined, 'check-option');
        label.append(check, node('span', m.title));
        modelOptions.append(label);
    }
    const conditionOptions = node('fieldset');
    conditionOptions.append(node('legend', 'Conditions to compare'));
    for (const condition of c.conditions) {
        const check = node('input');
        check.type = 'checkbox';
        check.checked = d.conditionIDs.includes(condition.id);
        check.disabled = genes.disabled;
        check.onchange = () => { const chosen = new Set(conditions.value.split(',').map(v => v.trim()).filter(Boolean)); check.checked ? chosen.add(condition.id) : chosen.delete(condition.id); conditions.value = [...chosen].join(', '); };
        const label = node('label', undefined, 'check-option');
        label.append(check, node('span', condition.title || condition.id));
        conditionOptions.append(label);
    }
    const advanced = node('details');
    advanced.append(node('summary', 'Advanced: exact model and condition identifiers'), labeled('Model identifiers', models), labeled('Condition identifiers', conditions));
    form.append(modelOptions, conditionOptions, advanced);
    if (d.status === 'draft') {
        form.append(button(state.pending ? 'Awaiting acknowledgement…' : 'Apply edits', () => run(async () => { const split = (v) => v.split(/[\s,]+/).filter(Boolean); localIntent = { action: 'edit', id: d.id, title: title.value, genes: split(genes.value), modelIDs: split(models.value), conditionIDs: split(conditions.value) }; await mutate(localIntent); localIntent = null; }), 'primary'), button('Undo acknowledged draft edit', () => run(() => mutate({ action: 'undo', id: d.id }))));
    }
    host.append(form, node('p', state.pending ? '◌ Pending — not yet accepted' : '✓ Shared revision ' + state.value.revision, 'pending'));
    const coverage = d.coverage || {};
    host.append(node('h3', 'Coverage before execution'));
    for (const g of coverage.genes || []) {
        host.append(node('p', `${g.gene} · ${g.status || g.referenceQuantity || g.evidence || (g.eligible ? 'supported' : 'UNAVAILABLE')} · ${g.reason || ''}`, 'muted'));
    }
    if (!coverage.canExecute) {
        host.append(node('p', coverage.executionReason || 'Unsupported proposal retained. Review the smallest correction below.', 'blocked'));
        for (const axis of d.axes || [])
            if (!axis.canExecute) {
                host.append(node('p', axis.modelID + ' / ' + axis.conditionID + ': ' + axis.reason, 'blocked'));
                for (const correction of axis.corrections || []) {
                    const b = button('Review supported correction', () => showCorrection(d, correction));
                    host.append(b);
                }
            }
    }
    host.append(node('h3', 'Selected arms'), node('p', d.targets.length ? d.targets.join(' · ') : 'No intervention added. No intervention remains the explicit baseline.', 'muted'));
    if (d.status === 'draft')
        renderCandidateSearch(host, c, s);
    const actions = node('div', undefined, 'actions');
    if (d.status === 'draft') {
        const seal = button('Seal and execute', () => run(() => mutate({ action: 'seal', id: d.id })), 'primary');
        seal.disabled = !coverage.canExecute || !d.targets.length || state.pending;
        actions.append(seal);
    }
    else if (d.status === 'sealed') {
        actions.append(button('Authorize reveal…', () => authorizeReveal(d), 'primary'), button('Exact replay', () => run(() => mutate({ action: 'replay', id: d.id }))));
    }
    else if (d.status === 'revealed') {
        actions.append(button('Investigate objective', () => showFailure(d)), button('Exact replay', () => run(() => mutate({ action: 'replay', id: d.id }))));
    }
    if (['interrupted', 'failed', 'cancelled'].includes(d.status) || state.value.operations.some(o => o.id === d.operation && ['interrupted', 'failed', 'cancelled'].includes(o.status)))
        actions.append(button('Recover interrupted operation', () => run(() => mutate({ action: 'recover', id: d.id }))));
    if (d.status !== 'draft')
        actions.append(button('Create editable revision', () => run(async () => { await mutate({ action: 'revise', id: d.id }); layout.update({ activeDraft: state.value.drafts.at(-1).id }); renderInspector(); })));
    const op = state.value.operations.find(o => o.draftID === d.id && o.status === 'running');
    if (op) {
        actions.replaceChildren(node('p', op.progress || 'Native operation running', 'pending'), button('Cancel operation', () => run(() => mutate({ action: 'cancel', id: op.draftID }))));
    }
    host.append(actions, button('Proposal evidence', () => showEvidence(d)), button('Compare selected candidates', () => { layout.update({ comparison: true }); renderComparison(); }));
}
function showCorrection(d, correction) {
    const host = node('div');
    host.append(node('p', 'Original intent remains: reduce ' + d.genes.join(', ') + ' using ' + d.targets.join(', ')), node('p', correction.conditionID ? 'Use the supported condition ' + correction.conditionID + ' while retaining genes and interventions.' : correction.removeUnsupportedGenes ? 'Remove only the explicitly unsupported genes: ' + correction.removeUnsupportedGenes.join(', ') : correction.removeUnsupportedTargets ? 'Remove only unsupported candidates: ' + correction.removeUnsupportedTargets.join(', ') : 'A compatible capability must be registered.'), detail('Advanced: exact correction', correction));
    let edit = null;
    if (correction.removeUnsupportedGenes)
        edit = { genes: d.genes.filter(g => !correction.removeUnsupportedGenes.includes(g)) };
    if (correction.removeUnsupportedTargets)
        edit = { targets: d.targets.filter(g => !correction.removeUnsupportedTargets.includes(g)) };
    if (correction.conditionID)
        edit = { conditionIDs: [correction.conditionID] };
    if (correction.conditionIDs)
        edit = { conditionIDs: correction.conditionIDs };
    if (correction.modelID)
        edit = { modelIDs: [correction.modelID] };
    if (correction.modelIDs)
        edit = { modelIDs: correction.modelIDs };
    if (edit && (!edit.genes || edit.genes.length))
        host.append(button('Apply this correction explicitly', () => run(async () => { await mutate({ action: 'edit', id: d.id, ...edit }); closeDrawer(); }), 'primary'));
    else
        host.append(node('p', 'This correction requires compatible data or a model. Register the exact missing capability before execution.'));
    drawer('Review correction', host);
}
function renderCandidateSearch(host, c, s) { host.append(node('h3', 'Supported interventions')); const query = node('input'); query.type = 'search'; query.value = search; query.placeholder = 'Search interventions'; query.setAttribute('aria-label', 'Search interventions'); query.oninput = () => { search = query.value; offset = 0; clearTimeout(searchTimer); searchTimer = window.setTimeout(() => run(loadCandidates), 180); }; host.append(query, node('div', undefined, 'candidate-list')); host.querySelector('.candidate-list').id = 'candidate-list'; const pages = node('div', undefined, 'pagination'); pages.append(button('Previous', () => { offset = Math.max(0, offset - 24); run(loadCandidates); }), node('span', 'Loading…'), button('Next', () => { if (offset + 24 < candidateTotal) {
    offset += 24;
    run(loadCandidates);
} })); pages.id = 'candidate-pages'; host.append(pages); renderCandidates(); }
function renderCandidates() { const host = $('candidate-list'); if (!host)
    return; host.replaceChildren(); const d = activeDraft(); for (const item of candidates) {
    const target = String(item.target), b = button('', () => run(() => chooseTarget(target)), 'candidate');
    b.draggable = true;
    b.ondragstart = e => e.dataTransfer.setData('text/numilab-intervention', target);
    b.setAttribute('aria-pressed', String(d?.targets.includes(target) || state.value?.selection?.target === target));
    const content = node('span');
    content.append(node('strong', target), node('small', `${item.controlCells ?? item.reference?.referenceCells ?? 'Unknown'} controls · ${item.outcomeCells ?? 'Unknown'} outcomes · ${item.reservedOutcomeCells ?? 'Owner controlled'} reserved`));
    b.append(content, node('span', d?.targets.includes(target) ? '✓' : '+'));
    host.append(b);
} const pages = $('candidate-pages'); if (pages) {
    pages.querySelector('span').textContent = `${candidateTotal ? offset + 1 : 0}–${Math.min(offset + 24, candidateTotal)} / ${candidateTotal}`;
    pages.querySelector('button').disabled = offset === 0;
    pages.querySelectorAll('button')[1].disabled = offset + 24 >= candidateTotal;
} }
async function chooseTarget(target) { const s = state.value.selection, d = activeDraft(); if (d?.status === 'draft') {
    const targets = d.targets.includes(target) ? d.targets.filter(t => t !== target) : [...d.targets, target];
    await mutate({ action: 'edit', id: d.id, targets });
} await setSelection({ ...s, target }); renderComparison(); }
async function loadCandidates() {
    const c = catalog(), s = state.value?.selection;
    if (!c || !s)
        return;
    const epoch = ++metadataEpoch;
    let items;
    if (cap(c).measuredExploration && !cap(c).prediction) {
        const data = await client.request('readout', { assayID: c.id, selection: s, query: { kind: 'interventions', mode: 'measured', search, offset, limit: 24 } }, { channel: 'candidates', cacheKey: identity(c.id, c.datasetFingerprint, c.observationAccessFingerprint, c.specimens, s.populationID, s.conditionID, search, offset, 'metadata-access') });
        items = data.items || [];
        candidateTotal = data.total || 0;
    }
    else {
        const all = (state.value.selectionSupport?.candidates || c.targets || []).filter((t) => !t.role || t.role === layout.value.role).filter((t) => String(t.target).toLowerCase().includes(search.toLowerCase()));
        candidateTotal = all.length;
        items = all.slice(offset, offset + 24);
    }
    if (epoch !== metadataEpoch)
        return;
    candidates = items;
    renderCandidates();
}
async function loadReadout() {
    const c = catalog(), s = state.value?.selection;
    if (!c || !s || !s.gene)
        return;
    const normalization = cap(c).prediction ? 'normalized' : layout.value.normalization;
    const epoch = ++readEpoch, started = performance.now(), beforeRequests = client.timings.length, host = $('view-content');
    client.cancel('readout');
    client.cancel('field');
    if (spatial)
        savedCamera = { key: identity(c.id, s.specimenID), camera: { ...spatial.camera } };
    spatial?.dispose();
    spatial = null;
    if (!['population', 'tissue'].includes(cap(c).presentation)) {
        host.replaceChildren(node('div', 'Unsupported assay presentation: ' + cap(c).presentation + '. A compatible owner capability is required.', 'empty'));
        return;
    }
    host.replaceChildren(node('p', 'Loading requested molecular readout…', 'pending'));
    const d = activeDraft();
    const record = d?.axisResults?.find(a => a.modelID === s.assayID && a.conditionID === s.conditionID)?.record || (d?.selection.assayID === s.assayID ? d.record : undefined);
    const access = d?.status || 'preview';
    if (d?.status === 'draft' && (!d.modelIDs.includes(c.id) || d.axes.some(a => !a.canExecute))) {
        host.replaceChildren(node('p', 'Prediction preview blocked for this proposal. ' + (d.coverage.executionReason || 'Select the declared compatible model.'), 'blocked'), node('p', 'Your original objective is retained. Correct the explicit support limitation on the proposal; another model is not substituted.', 'muted'));
        activeReadout = null;
        renderComparison();
        return;
    }
    const key = identity(c.id, c.datasetFingerprint, c.observationAccessFingerprint, c.modelVersion, state.value.models.find(m => m.id === c.id)?.version, c.specimens, s, record, d?.modelIDs, d?.conditionIDs, normalization, layout.value.role, access, d?.axes?.map(a => a.binding));
    try {
        if (cap(c).presentation === 'tissue') {
            const geometry = await client.request('specimen', { assayID: c.id, specimen: s.specimenID }, { channel: 'readout', cacheKey: identity(c.id, c.specimens, s.specimenID, 'geometry') });
            if (epoch !== readEpoch)
                return;
            spatial = new SpatialView(host, geometry, s.populationID, p => { showEvidence({ ...p, annotation: 'Owner-projected identity; uncertainty retained', observationAccess: 'unchanged' }, 'Selected source cell'); if (p.population && c.populations.some(x => x.id === p.population))
                run(() => setSelection({ ...state.value.selection, populationID: p.population })); });
            if (savedCamera?.key === identity(c.id, s.specimenID)) {
                spatial.camera = savedCamera.camera;
                spatial.draw();
            }
            const field = await client.request('spatial-feature', { assayID: c.id, specimen: s.specimenID, gene: s.gene, ...(record ? { id: record } : {}) }, { channel: 'field', cacheKey: key });
            if (epoch !== readEpoch)
                return;
            activeReadout = field;
            spatial.updateField(field, s.target, layout.value.role);
            const molecular = node('div');
            populationView(molecular, field, s.target);
            host.append(molecular);
        }
        else {
            const measured = cap(c).prediction === false;
            const field = await client.request('readout', { assayID: c.id, selection: s, ...(record ? { id: record } : {}), query: { kind: 'gene', mode: measured ? 'measured' : 'prediction', gene: s.gene, ...(measured ? { target: s.target } : {}), normalization: normalization, limit: 128, offset: 0 } }, { channel: 'readout', cacheKey: key });
            if (epoch !== readEpoch)
                return;
            activeReadout = field;
            populationView(host, field, s.target);
        }
        performance.mark('numilab-gene-ready');
        readoutTimings.push({ gene: s.gene, dataset: c.id, ms: performance.now() - started, cached: client.timings.slice(beforeRequests).some(t => t.cached), normalization: normalization, access });
        if (readoutTimings.length > 120)
            readoutTimings.shift();
        renderComparison();
    }
    catch (error) {
        if (epoch !== readEpoch || (error instanceof DOMException && error.name === 'AbortError'))
            return;
        host.replaceChildren(node('h2', 'Readout unavailable'), node('p', errorMessage(error), 'blocked'), node('p', 'Restore the named verified dependency or select a compatible assay. Your objective and sealed records are retained.', 'muted'), button('Inspect exact dependency', () => showEvidence({ reason: errorMessage(error), ...s, evidence: 'UNAVAILABLE' })));
    }
}
function renderComparison() {
    const host = $('comparison');
    host.hidden = !layout.value.comparison;
    if (host.hidden)
        return;
    const d = activeDraft();
    host.replaceChildren(node('div', 'PERSISTENT COMPARISON', 'eyebrow'), node('h2', d?.title || 'Compare supported candidates'), node('p', 'One objective, shared readout scales. No intervention is the explicit baseline. Model agreement is not calibrated confidence.', 'muted'));
    host.ondragover = e => e.preventDefault();
    host.ondrop = e => { e.preventDefault(); const t = e.dataTransfer?.getData('text/numilab-intervention'); if (t)
        run(() => chooseTarget(t)); };
    if (!d) {
        host.append(node('p', 'Create a proposal and add candidates with click, keyboard or drag.', 'empty'));
        return;
    }
    const table = node('table');
    const header = node('tr');
    for (const label of ['Intervention', 'Model / condition', 'Predicted utility', 'Observed utility', 'Support'])
        header.append(node('th', label));
    table.append(header);
    const rows = [];
    for (const axis of d.axisResults || []) {
        const ev = axis.result?.objectiveEvaluation;
        for (const a of ev?.arms || [])
            if (a.role === 'direct')
                rows.push({ ...a, modelID: axis.modelID, conditionID: axis.conditionID, record: axis.record });
    }
    if (!rows.length)
        for (const effect of (d.coverage?.candidateEffects || []).filter((e) => d.targets.includes(e.target)))
            rows.push({ ...effect, target: effect.target, predictedUtility: effect.predictedUtility ?? effect.utility, modelID: d.modelIDs[0], conditionID: d.conditionIDs[0] });
    rows.unshift({ target: 'no-intervention', predictedUtility: 0, modelID: 'baseline', conditionID: d.conditionIDs.join(', '), reason: 'Reference baseline' });
    for (const a of rows) {
        const row = node('tr');
        row.dataset.selected = String(a.target === state.value?.selection?.target);
        const cell = node('td');
        cell.append(button(a.target, () => { if (a.target === 'no-intervention') {
            showEvidence({ reason: 'No intervention is the reference baseline', predictedUtility: 0 });
            return;
        } run(async () => { const s = state.value.selection; await setSelection({ ...s, assayID: a.modelID === 'baseline' ? s.assayID : a.modelID, populationID: d.selection.populationID, specimenID: d.selection.specimenID, conditionID: a.conditionID || s.conditionID, target: a.target }); }); }));
        row.append(cell);
        for (const text of [a.modelID + ' / ' + a.conditionID, fmt(a.predictedUtility), fmt(a.observedUtility), a.reason || 'Experimental · recommendation unqualified'])
            row.append(node('td', text));
        table.append(row);
    }
    host.append(table);
    const panels = node('div', undefined, 'comparison-panes');
    if (activeReadout) {
        const selected = (activeReadout.arms || []).filter(a => d.targets.includes(a.target) && (!a.role || a.role === layout.value.role)).slice(0, 4);
        const max = Math.max(1e-9, ...selected.flatMap(a => [a.control, a.predicted, a.observed]).filter((v) => typeof v === 'number').map(Math.abs));
        for (const a of selected) {
            const panel = node('section', undefined, 'readout-card');
            panel.append(button(a.target + ' · ' + (a.role || 'direct'), () => run(() => setSelection({ ...state.value.selection, target: a.target }))));
            const plot = node('div');
            populationView(plot, { ...activeReadout, arms: [a] }, a.target, max);
            panel.append(plot);
            panels.append(panel);
        }
    }
    host.append(panels);
    if (d.modelDifference?.available) {
        host.append(node('h3', 'What changed between models'));
        for (const comparison of d.modelDifference.comparisons) {
            host.append(node('p', comparison.leftModel + ' → ' + comparison.rightModel + ' · ' + (comparison.rankingsChanged ? 'Numerical rankings changed' : 'Rankings do not differ')));
            for (const item of comparison.candidates) {
                host.append(button(item.target + ': utility difference ' + fmt(item.utilityDifference), () => showEvidence(item, 'Gene contributions to model difference')));
            }
        }
    }
    else
        host.append(node('p', d.modelDifference?.reason || 'Seal compatible models to compare exact owner-computed utilities.', 'muted'));
    host.append(button('Inspect objective discrepancy', () => showFailure(d)));
}
function authorizeReveal(d) { $('reveal-description').textContent = `Experiment ${d.title} (${d.id}). Open ${d.genes.join(', ')} observations for ${d.targets.join(', ')} in ${d.selection.populationID}; records ${(d.axisResults || []).map(a => a.record).join(', ') || d.record}.`; $('reveal-dialog').showModal(); $('reveal-confirm').onclick = () => run(async () => { await mutate({ action: 'reveal', id: d.id, authorizeReveal: true }); $('reveal-dialog').close(); }); }
function showFailure(d) { const content = failureView(d.result || {}, (gene, target) => run(async () => { await setSelection({ ...state.value.selection, gene, target }); closeDrawer(); layout.update({ comparison: true }); renderComparison(); })); drawer('Investigate the objective', content); }
function showHistory() { const host = node('div'); for (const d of (state.value?.drafts || []).slice().reverse()) {
    const item = node('div', undefined, 'history-item');
    item.append(badge(d.status.toUpperCase()), button(d.title, () => run(async () => { await openDraft(d); closeDrawer(); })), node('p', d.record ? 'Immutable record ' + d.record : 'Editable draft', 'muted'));
    host.append(item);
} for (const event of (state.value?.history || []).slice().reverse().slice(0, 60))
    host.append(node('p', event.action + ' · revision ' + event.revision, 'muted')); drawer('Experiment history', host); }
function showActivity() { const host = node('div'); for (const op of (state.value?.operations || []).slice().reverse()) {
    const item = node('section', undefined, 'history-item');
    item.append(badge(op.status.toUpperCase()), node('h3', op.action), node('p', op.progress || op.error || op.id));
    if (op.status === 'running')
        item.append(button('Cancel', () => run(() => mutate({ action: 'cancel', id: op.draftID }))));
    if (['interrupted', 'failed', 'cancelled'].includes(op.status))
        item.append(button('Review recovery', () => { const d = state.value.drafts.find(d => d.id === op.draftID); if (d)
            run(async () => { await openDraft(d); closeDrawer(); }); }));
    host.append(item);
} if (!host.children.length)
    host.append(node('p', 'No operations. Inspection never starts execution.')); drawer('Activity and recovery', host); }
function showStorage() {
    const host = node('div');
    host.append(node('p', 'Snapshots capture acknowledged edits, immutable records, artifact bindings and stopped operations. Stop live workers first. Restore issues fresh credentials and does not resume workers automatically.'));
    host.append(button('Create consistent snapshot', () => run(async () => {
        const saved = await client.request('snapshot', { expectedRevision: state.value.revision });
        showEvidence(saved, 'Verified snapshot');
        $('drawer-content').append(button('Export verified snapshot', () => run(async () => { const exported = await client.request('export', { id: saved.id }); showEvidence(exported, 'Portable laboratory export'); })));
    }), 'primary'), node('p', 'Restore through the existing plugin: numi wet-lab restore --id SNAPSHOT --name LAB, then numi wet-lab open --name LAB. The plugin resolves locations; sealed records are never rewritten.', 'muted'));
    drawer('Snapshot, export and restoration', host);
}
function showViewers() {
    run(async () => {
        const data = await client.request('viewers');
        const host = node('div');
        host.append(node('p', 'Native viewer handoffs retain source and selection identity. Preparation and opening do not establish confirmed rendering.', 'muted'));
        for (const viewer of data.viewers || [])
            host.append(node('h3', viewer.name), node('p', viewer.scope, 'muted'), badge('NOT OPENED'));
        if (data.selectedCellExport?.available)
            host.append(button('Prepare selected measured cells', () => run(async () => { const result = await client.request('viewers/selected', { expectedRevision: state.value.revision, limit: 128 }); showEvidence({ ...result, status: 'PREPARED · RENDERING NOT CONFIRMED' }, 'Source-bound viewer handoff'); const prompt = 'Use the existing NumiLab biological viewer workflow to open and verify the latest selected-cell handoff. Run numi view context, retain the exact source and selection, then confirm live viewer rendering. Do not reveal observations.'; $('drawer-content').append(button('Copy request for Codex', () => run(() => navigator.clipboard.writeText(prompt)))); })));
        else
            host.append(node('p', 'No source-bound selected-cell export for this assay. Ask Codex to discover compatible structure, sequence or microscopy artifacts using numi view context. No unverified cross-domain mapping is inferred.', 'blocked'));
        drawer('Native biological viewers', host);
    });
}
function renderConflict() { const host = $('conflict'); host.hidden = !state.rejected; if (!state.rejected)
    return; host.replaceChildren(node('strong', 'Edit not accepted. The shared laboratory changed.'), node('p', 'Review the current acknowledged card and your preserved intent. No automatic retry was made.'), detail('Rejected intent', state.rejected), button('Review current state', () => { localIntent = state.rejected; renderInspector(); }), button('Discard rejected edit', () => { state.rejected = null; localIntent = null; renderConflict(); renderInspector(); })); }
function render() {
    const shared = state.value;
    if (!shared)
        return;
    $('connection').textContent = state.pending ? 'Awaiting acknowledgement' : 'Connected · revision ' + shared.revision;
    renderConflict();
    renderNavigation();
    renderStart();
    const c = catalog(), s = shared.selection;
    if (!c || !s)
        return;
    renderHeading(c, s);
    const progress = identity(shared.operations.filter(o => o.status === 'running').map(o => o.progress));
    if (lastRevision !== shared.revision || lastPending !== state.pending || lastProgress !== progress) {
        renderToolbar(c, s);
        renderHeading(c, s);
        renderInspector();
        renderComparison();
        lastRevision = shared.revision;
        lastPending = state.pending;
        lastProgress = progress;
    }
    if (state.pending) {
        $('connection').textContent = 'Awaiting acknowledgement';
        $('inspector').querySelectorAll('button,input,textarea,select').forEach(e => e.disabled = true);
    }
    else
        $('connection').textContent = 'Connected · revision ' + shared.revision;
    const key = identity(s, activeDraft()?.id, activeDraft()?.status, activeDraft()?.axisResults?.map(a => a.record), activeDraft()?.modelIDs, activeDraft()?.conditionIDs, activeDraft()?.genes);
    if (key !== renderedSelection && !state.pending) {
        renderedSelection = key;
        candidates = [];
        activeReadout = null;
        $('start').hidden = true;
        $('investigation').hidden = false;
        run(loadCandidates);
        run(loadReadout);
    }
}
function showPerformance() { const data = { status: 'CLIENT PERFORMANCE INSTRUMENTATION', format: 'numilab-browser-timings/v1', surface: 'actual Codex browser', metadataAfterReadinessMs: metadataMs, inputFeedbackMs: feedback, inputFeedbackMeasurement: 'Trusted pointer/keyboard event to next animation frame; excludes physical display latency', readoutTimings, requests: client.timings, cache: '32-entry bounded session cache; process and OS caches reported separately', wholeCohortExpressionTransfer: false }; drawer('Performance and support', evidenceSummary(data)); }
function searchCommands() { const text = $('command-query').value.toLowerCase(); const host = $('command-results'); host.replaceChildren(); const actions = [['Open laboratory home', () => { $('start').hidden = false; $('investigation').hidden = true; }], ['Show activity and recover interrupted work', showActivity], ['Show history', showHistory], ['Open native biological viewers', showViewers], ['Snapshot and restore laboratory', showStorage], ['Performance and support', showPerformance], ['Reconnect to service', () => run(connect)], ...catalogs.map(c => ['Open ' + c.title, () => run(() => openDataset(c))]), ...(state.value?.drafts || []).map(d => ['Inspect ' + d.title, () => run(() => openDraft(d))])]; for (const [title, action] of actions.filter(([title]) => title.toLowerCase().includes(text)))
    host.append(button(title, () => { $('command').close(); action(); })); }
function openCommand() { $('command').showModal(); $('command-query').focus(); searchCommands(); }
async function connect() { try {
    await client.request('health');
    readinessAt = performance.now();
    const [all] = await Promise.all([client.request('assays'), state.refresh()]);
    metadataMs = performance.now() - readinessAt;
    catalogs = all;
    online = true;
    render();
    performance.mark('numilab-workspace-ready');
    if (poll)
        clearInterval(poll);
    poll = window.setInterval(() => { if (!state.pending && document.visibilityState === 'visible')
        void state.refresh().catch(() => { online = false; $('connection').textContent = 'Offline · reconnect'; notice('Service interrupted. Drafts and sealed records are retained. Run numi wet-lab open, then reload the returned local URL. Operations require explicit recovery.'); }); }, 2000);
}
catch (error) {
    online = false;
    $('connection').textContent = 'Unavailable';
    $('start').replaceChildren(node('h1', 'Reconnect to your laboratory'), node('p', errorMessage(error), 'blocked'), node('p', 'Run numi wet-lab open in Codex. It discovers the retained laboratory, replaces stale credentials and returns its ready URL. No stopped experiment restarts automatically.'), button('Retry connection', () => run(connect)));
} }
state.onChange = render;
$('nav-toggle').onclick = () => { layout.update({ navigation: !layout.value.navigation }); renderNavigation(); };
$('appearance').onclick = () => { const modes = ['system', 'light', 'dark']; layout.update({ appearance: modes[(modes.indexOf(layout.value.appearance) + 1) % 3] }); notice('Appearance: ' + layout.value.appearance); };
$('command-open').onclick = openCommand;
$('command-query').oninput = searchCommands;
$('command-query').onkeydown = e => { if (e.key === 'Enter' || e.key === 'ArrowDown') {
    const first = $('command-results').querySelector('button');
    if (first) {
        e.preventDefault();
        if (e.key === 'Enter')
            first.click();
        else
            first.focus();
    }
} };
$('activity-open').onclick = showActivity;
$('evidence-open').onclick = () => showEvidence({ ...state.value?.selectionSupport, selection: state.value?.selection, model: catalog()?.id, biologicalPromotion: false });
$('drawer-close').onclick = closeDrawer;
$('reveal-cancel').onclick = () => $('reveal-dialog').close();
document.addEventListener('keydown', e => { if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
    e.preventDefault();
    openCommand();
} });
window.numilabQualification = () => ({ timings: client.timings, inputFeedback: feedback, revision: state.value?.revision, presentation: 'workbench-v1', wholeExpressionTransfer: false, online });
run(connect);
