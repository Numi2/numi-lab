"use strict";
// Native viewers live in Codex. This surface carries intent and the existing
// biological selection, with bounded owner-admitted exports and no raw-assay access.
(() => {
  const element = (tag, text) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const entry = element("button", "Inspect biology");
  entry.className = "quiet";
  entry.type = "button";
  document.querySelector(".header-right").prepend(entry);
  const dialog = element("dialog");
  dialog.setAttribute("aria-label", "Inspect biology with Numi and Codex");
  document.body.append(dialog);
  let epoch = 0;
  entry.onclick = async () => {
    const request = ++epoch;
    dialog.replaceChildren();
    const close = element("button", "Close");
    close.onclick = () => { epoch++; dialog.close(); };
    dialog.append(close, element("h2", "Inspect biology"), element("p", "Reading your current selection…"));
    dialog.showModal();
    try {
      const response = await fetch("/api/viewers", {headers: {"X-Wet-Lab-Token": window.WET_LAB_TOKEN}});
      const context = await response.json();
      if (!response.ok) throw Error(context.error || "Viewer context unavailable");
      if (request !== epoch || !dialog.open) return;
      dialog.lastChild.remove();
      const selected = context.selection;
      dialog.append(element("p", selected
        ? [selected.gene, selected.populationID, selected.specimenID, selected.conditionID].filter(Boolean).join(" · ")
        : "Select a specimen or population to carry its context into an inspection."));
      dialog.append(element("p", "OpenAI’s viewers open beside your Numi conversation in Codex. Copy a request below to inspect a source with this selection. Measurements keep their existing access state."));
      for (const viewer of context.viewers) {
        const section = element("section");
        section.className = "panel";
        section.append(element("h3", viewer.name), element("p", viewer.scope));
        const inspect = element("button", "Copy " + viewer.kind + " inspection request");
        const feedback = element("p");
        feedback.setAttribute("role", "status");
        let prepared = null;
        if (viewer.kind === "slide" && context.selectedCellExport?.available) {
          const prepare = element("button", "Prepare selected cells for Slide Viewer");
          prepare.onclick = async () => {
            prepare.disabled = true;
            feedback.textContent = "Preparing the accessible cell page from NumiVivo…";
            try {
              const response = await fetch("/api/viewers/selected", {method: "POST",
                headers: {"X-Wet-Lab-Token": window.WET_LAB_TOKEN, "Content-Type": "application/json"},
                body: JSON.stringify({expectedRevision: context.revision, limit: 128, offset: 0})});
              const result = await response.json();
              if (!response.ok) throw Error(result.error || "Cell export unavailable");
              if (request !== epoch || !dialog.open) return;
              prepared = result;
              inspect.textContent = "Copy prepared cell inspection request";
              feedback.textContent = `Prepared ${result.coverage.exportedCells} accessible cells for ${selected.gene}. ` +
                "This is one measured gene and a bounded page, with no invented tissue positions. Copy the request to open it in Codex.";
            } catch (error) {
              if (request === epoch) feedback.textContent = "Could not prepare cells: " + error.message;
            } finally { prepare.disabled = false; }
          };
          section.append(prepare, element("p", context.selectedCellExport.scope));
        }
        inspect.onclick = async () => {
          // The request is context, not a source association, executable code or access grant.
          const prompt = prepared
            ? `Use Numi to verify the viewer handoff at ${JSON.stringify(prepared.handoffPath)} and open its authorized cell artifact with Slide Viewer. ` +
              `Keep its exact source and selection: ${JSON.stringify(selected)} at revision ${prepared.revision}. ` +
              "This is an owner-authorized bounded page of one measured gene. Preserve original cell IDs, raw UMI counts and access provenance; do not infer spatial coordinates or full-assay coverage. Check native readiness and retain the same viewer session."
            : `Use Numi and the ${viewer.name} to inspect the current biological selection. ` +
            `Read numi view context first; the requested Wet Lab revision is ${context.revision}. ` +
            `Selection: ${JSON.stringify(selected)}. Resolve an authorized ${viewer.kind} artifact from its Numi owner, ` +
            "preserve the source identity and verify its association with this selection. " +
            "Open the native viewer, retain the same session for follow-up, and carry exact identities back to Numi. " +
            "Do not reveal reserved observations or substitute another specimen, gene, sequence or structure.";
          try {
            await navigator.clipboard.writeText(prompt);
            feedback.textContent = "Copied. Paste into your Numi conversation in Codex to open the native viewer.";
          } catch (_) {
            const text = element("textarea");
            text.value = prompt;
            text.readOnly = true;
            text.setAttribute("aria-label", viewer.name + " inspection request");
            feedback.replaceChildren(element("span", "Copy this request into Codex:"), text);
            text.focus(); text.select();
          }
        };
        section.append(inspect, feedback); dialog.append(section);
      }
      dialog.append(element("p", "A gene name alone does not establish a protein isoform, residue mapping or image registration. Numi resolves those links from source evidence."));
    } catch (error) {
      if (request === epoch) dialog.append(element("p", "Inspection unavailable: " + error.message));
    }
  };
  dialog.addEventListener("cancel", () => { epoch++; });
})();
