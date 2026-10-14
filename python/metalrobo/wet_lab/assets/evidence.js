import { node, detail, fmt, button, badge } from './dom.js';
export function evidenceSummary(data) {
    const host = node('section');
    host.append(badge(data.status || data.evidence || 'OWNER EVIDENCE'));
    const binding = data.axes?.[0]?.binding || data.binding || {};
    const selection = data.selection || {};
    const dl = node('dl', undefined, 'evidence-summary');
    for (const [label, value] of Object.entries({ Source: data.sourceCitation || data.source || data.sourceSHA256 || binding.specimenSHA256, Model: data.modelVersion || data.modelID || binding.modelVersion, Population: data.population || selection.populationID, Units: data.units, Access: data.access || data.observationAccess || 'unchanged', Support: data.reason || data.distributionStatus, Condition: selection.conditionID || data.conditionID, Runtime: binding.runtimeSHA256, Scoring: binding.scoringOwnerSHA256, Replay: data.record, Promotion: data.biologicalPromotion === true ? 'Promoted' : 'Not biologically promoted' })) {
        if (value == null)
            continue;
        dl.append(node('dt', label), node('dd', typeof value === 'object' ? JSON.stringify(value) : String(value)));
    }
    host.append(dl, detail('Advanced: exact owner response', data));
    return host;
}
export function failureView(result, onGene) {
    const host = node('section');
    const ev = result.objectiveEvaluation;
    if (!ev) {
        host.append(node('p', 'No revealed objective evaluation is available. Observation access remains unchanged.'));
        return host;
    }
    host.append(node('p', 'Gene contributions and baseline differences below are computed by the scientific owner. Numerical ordering does not establish a recommendation.', 'muted'));
    for (const arm of ev.arms || []) {
        const box = node('section', undefined, 'support-card');
        box.append(node('h3', `${arm.target} · ${arm.role} · ${arm.population || ''}`), node('p', `Predicted utility ${fmt(arm.predictedUtility)} · observed utility ${fmt(arm.observedUtility)}`));
        const table = node('table');
        const head = node('tr');
        for (const h of ['Gene', 'Control', 'Predicted', 'Observed', 'Residual', 'Training mean Δ', 'Inspect'])
            head.append(node('th', h));
        table.append(head);
        for (const g of arm.genes || []) {
            const row = node('tr');
            for (const v of [g.gene, fmt(g.control), fmt(g.predicted), fmt(g.observed), fmt(g.residual), fmt(g.trainingMeanDelta)])
                row.append(node('td', v));
            const cell = node('td');
            cell.append(button('Distributions', () => onGene(g.gene, arm.target)));
            row.append(cell);
            table.append(row);
        }
        box.append(table);
        host.append(box);
    }
    host.append(detail('Exact evaluation and selection baselines', ev));
    return host;
}
