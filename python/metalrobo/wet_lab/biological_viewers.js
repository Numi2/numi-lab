"use strict";
// Native viewers live in Codex. This surface carries intent and the existing
// biological selection, without copying assay matrices or revealing observations.
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
        inspect.onclick = async () => {
          // The request is context, not a source association, executable code or access grant.
          const prompt = `Use Numi and the ${viewer.name} to inspect the current biological selection. ` +
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
