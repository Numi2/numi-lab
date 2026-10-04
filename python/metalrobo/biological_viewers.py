"""Source-bound handoffs from Numi owners to OpenAI's native biological viewers.

This module prepares tool arguments, never runs a second renderer or scientific
engine. A prepared handoff is not a mounted viewer or an observation reveal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid

SCHEMA = "numi.biological-view.v1"
LIMIT = 512 * 1024 * 1024
VIEWERS = {
    "structure": {
        "name": "Molecular Structure Viewer", "plugin": "structure-viewer",
        "skill": "structure-viewer", "tool": "structure.open_from_chat",
        "extensions": ["pdb", "cif", "mmcif", "mol", "sdf", "mol2", "pqr", "pdbqt", "gro", "xyz",
                       "pdb.gz", "cif.gz", "mmcif.gz", "pdb.bgz", "cif.bgz", "mmcif.bgz"],
        "owners": ["NumiVivo", "NumiTissue", "Numi Human", "Numi Automata"],
        "scope": "Structures, ligands, density and topology-bound trajectories",
        "identity": "Object, model, author chain/residue, insertion code, atom and coordinate frame",
    },
    "sequence": {
        "name": "Biological Sequence & Alignment Viewer", "plugin": "sequence-viewer",
        "skill": "biological-sequence-viewer", "tool": "sequence.open_from_chat",
        "extensions": ["fasta", "fa", "fas", "fna", "faa", "ffn", "frn", "mfa", "aln-fasta", "afa",
                       "afasta", "gb", "gbk", "genbank", "gbff", "embl", "emb", "fastq", "fq",
                       "ab1", "abi", "scf", "dna", "aln", "clustal", "clw", "sto", "stk",
                       "stockholm", "a2m", "a3m", "msf", "phy", "phylip", "nex", "nexus"],
        "owners": ["NumiVivo", "NumiTissue", "NumiBrain", "Numi Automata"],
        "scope": "Sequences, alignments, annotations, traces and regional read evidence",
        "identity": "Record/accession version, strand and 1-based inclusive source coordinates; alignment columns are separate",
    },
    "slide": {
        "name": "Slide Viewer", "plugin": "slide-viewer", "skill": "slide-viewer",
        "tool": "slide.open_from_chat", "extensions": ["svs", "tif", "tiff", "h5ad", "dcm", "dicom"],
        "owners": ["NumiVivo", "NumiTissue", "NumiBrain", "Numi Human"],
        "scope": "Microscopy, tissue slides and AnnData; spatial views require actual coordinates",
        "identity": "Source, matrix, physical gene column, observation IDs, specimen, scene/plane and calibrated frame",
    },
}
COMPANIONS = {"structure": {"xtc", "dcd", "trr", "nc", "netcdf", "lammpstrj", "psf", "prmtop", "top",
                             "ccp4", "mrc", "map", "dsn6", "cube", "dx"},
              "sequence": {"bed", "gff", "gff3", "vcf", "sam", "bam", "bai"},
              "slide": {"geojson", "json"}}


def catalog():
    return [{"kind": k, **v, "pluginID": v["plugin"] + "@openai-curated-remote"} for k, v in VIEWERS.items()]


def route(path):
    name = Path(path).name.lower()
    for kind, viewer in VIEWERS.items():
        if any(name.endswith("." + extension) for extension in viewer["extensions"]):
            return kind
    raise ValueError("Unsupported primary artifact. Trajectories/maps need a structure; overlays need a slide. "
                     "OME-Zarr and DICOM series use the viewer's dedicated source-admission tools.")


def fingerprint(path, max_bytes=LIMIT):
    path = Path(path).expanduser().absolute()
    if path.is_symlink():
        raise ValueError("Use an explicit regular source file, not a symbolic link")
    # Reject symlink parents too: publication/verification must retain the same source identity.
    if path.resolve() != path:
        raise ValueError("Source path contains a symbolic link or unresolved parent")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= max_bytes:
            raise ValueError("Source must be a nonempty regular file within --max-source-bytes")
        digest = hashlib.sha256()
        total = 0
        while block := stream.read(1024 * 1024):
            total += len(block)
            if total > max_bytes:
                raise ValueError("Source grew beyond the admitted byte limit")
            digest.update(block)
        after = os.fstat(stream.fileno())
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if signature(before) != signature(after) or signature(after) != signature(path.stat()):
        raise ValueError("Source changed while preparing the handoff")
    return {"path": str(path), "bytes": total, "sha256": digest.hexdigest()}


def public_context(state, workspace):
    selection = state.get("selection")
    keys = ("assayID", "specimenID", "populationID", "conditionID", "gene", "target", "experimentID")
    return {"workspace": str(Path(workspace).resolve()), "revision": state["revision"],
            "selection": {k: selection[k] for k in keys if k in selection} if selection else None,
            "observationAccess": "unchanged", "viewers": catalog(),
            "association": "Selection context only; source-to-specimen and gene-to-protein mappings must be verified"}


def active_workspace():
    config = json.loads(Path(os.environ.get("NUMI_WET_LAB_CONNECTION", str(Path.home() / ".numi/wet-lab-active.json"))).read_text())
    url = urllib.parse.urlsplit(config["url"])
    if url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port or url.path not in ("", "/") or url.query or url.fragment or url.username:
        raise ValueError("Wet Lab must be the active local loopback service")
    return config


def workspace_request(config, endpoint="/api/shared", body=None):
    req = urllib.request.Request(config["url"].rstrip("/") + endpoint,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"X-Wet-Lab-Token": config["token"], "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def prepare(path, owner, *, companions=(), context=None, placement="chat", max_bytes=LIMIT):
    kind = route(path)
    source = fingerprint(path, max_bytes)
    related = []
    for companion in companions:
        if Path(companion).suffix.lower().lstrip(".") not in COMPANIONS[kind]:
            raise ValueError("Unsupported companion for " + kind)
        related.append(fingerprint(companion, max_bytes))
    tool = VIEWERS[kind]["tool"]
    arguments = {"path": source["path"]}
    if kind == "structure":
        arguments["openIntentId"] = str(uuid.uuid4())
        if placement == "side-pane":
            tool = "structure.open_in_side_pane"
    return {"schema": SCHEMA, "id": str(uuid.uuid4()), "status": "prepared",
            "viewerReady": False, "owner": owner, "kind": kind, "source": source,
            "companions": related, "context": context, "placement": placement,
            "pluginID": VIEWERS[kind]["plugin"] + "@openai-curated-remote",
            "open": {"tool": tool, "arguments": arguments},
            "afterOpen": {"retain": "viewerSessionId", "placement": placement,
                          "readiness": "Read live viewer state; session creation is not rendering",
                          "companions": "Load through the same viewer after checking format, topology and registration",
                          "selection": "Resolve exact current viewer IDs; never guess a cross-domain mapping"},
            "evidence": "inspection-only", "observationAccess": "unchanged"}


def verify(handoff, max_bytes=LIMIT):
    if handoff.get("schema") != SCHEMA:
        raise ValueError("Unsupported Numi viewer handoff")
    if handoff.get("kind") != route(handoff["source"]["path"]):
        raise ValueError("Viewer kind does not match the source")
    for source in [handoff["source"], *handoff.get("companions", [])]:
        if fingerprint(source["path"], max_bytes) != source:
            raise ValueError("Source changed; prepare a new handoff and retain the old evidence")
    # Never execute a tool/payload supplied by a handoff file. Reconstruct known arguments.
    kind = handoff["kind"]
    expected = VIEWERS[kind]["tool"]
    if kind == "structure" and handoff.get("placement") == "side-pane":
        expected = "structure.open_in_side_pane"
    arguments = {"path": handoff["source"]["path"]}
    if kind == "structure":
        arguments["openIntentId"] = str(uuid.UUID(handoff["open"]["arguments"]["openIntentId"]))
    if handoff.get("open") != {"tool": expected, "arguments": arguments}:
        raise ValueError("Unexpected viewer opening arguments")
    return {"status": "verified", "viewerReady": False, "open": {"tool": expected, "arguments": arguments},
            "context": handoff.get("context"), "observationAccess": "unchanged"}


def return_selection(handoff, config, current, gene, revision):
    """Return an explicitly resolved source gene through the existing owner CAS."""
    context = handoff.get("context")
    if not context or context["workspace"] != str(Path(config["workspace"]).resolve()):
        raise ValueError("This handoff is not bound to the active Wet Lab")
    if revision != current["revision"] or context["revision"] != revision or context["selection"] != current.get("selection"):
        raise ValueError("Wet Lab selection changed. Review current human edits and prepare a fresh handoff.")
    if not isinstance(gene, str) or not gene.strip() or len(gene) > 256:
        raise ValueError("An exact owner gene identifier is required")
    if not current.get("selection"):
        raise ValueError("Select a Wet Lab population first")
    return {"action": "selection", "expectedRevision": revision, "actor": "codex-viewer",
            "selection": {**current["selection"], "gene": gene}}


def installed_status():
    result = subprocess.run(["codex", "plugin", "list", "--json"], capture_output=True, text=True, timeout=30, check=True)
    installed = {x["pluginId"]: x for x in json.loads(result.stdout)["installed"]}
    return [{**v, "installed": installed.get(v["pluginID"], {}).get("installed", False),
             "enabled": installed.get(v["pluginID"], {}).get("enabled", False),
             "version": installed.get(v["pluginID"], {}).get("version"),
             "toolAvailability": "Check current conversation tools; a fresh session may be needed after installation"} for v in catalog()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("catalog", help="Discover viewers and source/identity contracts")
    sub.add_parser("status", help="Check actual companion plugin installations")
    sub.add_parser("context", help="Inspect the current Wet Lab selection without reading observations")
    prepare_parser = sub.add_parser("prepare", help="Prepare an exact native-viewer opening; does not open or reveal")
    prepare_parser.add_argument("source", type=Path)
    prepare_parser.add_argument("--owner", default="NumiVivo")
    prepare_parser.add_argument("--companion", action="append", default=[], type=Path)
    prepare_parser.add_argument("--placement", choices=["chat", "side-pane"], default="chat")
    prepare_parser.add_argument("--wet-lab", action="store_true", help="Bind current shared selection; not proof the artifact belongs to it")
    prepare_parser.add_argument("--revision", type=int)
    prepare_parser.add_argument("--output", type=Path, help="Save a new handoff without overwriting")
    for name in ("verify", "return"):
        parser = sub.add_parser(name)
        parser.add_argument("handoff", type=Path)
        parser.add_argument("--max-source-bytes", type=int, default=LIMIT)
        if name == "return":
            parser.add_argument("--gene", required=True, help="Exact gene resolved from live viewer and owner feature identities")
            parser.add_argument("--revision", required=True, type=int)
    prepare_parser.add_argument("--max-source-bytes", type=int, default=LIMIT)
    a = p.parse_args(argv)
    try:
        if a.action == "catalog": result = catalog()
        elif a.action == "status": result = installed_status()
        elif a.action == "context":
            c = active_workspace(); result = workspace_request(c, "/api/viewers")
        elif a.action == "prepare":
            context = None
            if a.wet_lab:
                c = active_workspace(); context = workspace_request(c, "/api/viewers")
                if a.revision != context["revision"]: raise ValueError("Use --revision from fresh numi view context")
            elif a.revision is not None: raise ValueError("--revision requires --wet-lab")
            result = prepare(a.source, a.owner, companions=a.companion, context=context,
                             placement=a.placement, max_bytes=a.max_source_bytes)
            if a.output:
                with a.output.open("x") as stream: json.dump(result, stream, indent=2, allow_nan=False)
        else:
            handoff = json.loads(a.handoff.read_text()); result = verify(handoff, a.max_source_bytes)
            if a.action == "return":
                c = active_workspace(); current_context = workspace_request(c, "/api/viewers")
                c["workspace"] = current_context["workspace"]
                current = workspace_request(c)
                body = return_selection(handoff, c, current, a.gene, a.revision)
                updated = workspace_request(c, body=body)
                result = public_context(updated, c["workspace"])
        print(json.dumps(result, indent=2, allow_nan=False)); return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print("numi view: " + str(error), file=sys.stderr); return 2


if __name__ == "__main__":
    sys.exit(main())
