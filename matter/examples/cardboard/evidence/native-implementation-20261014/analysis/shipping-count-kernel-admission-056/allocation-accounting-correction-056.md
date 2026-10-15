# Allocation-accounting correction for source56

The previous source55 CPU report is retained unmodified at `/Users/home/cardboard-implementation-20261014/shipping-allocation-preflight-source055.json` (SHA-256 `cf8d36279f2f062d45a6759a06327089698725ddd7bfbd4f6482e2ff1700ee71`). It used `max(modeled Metal subtotal, modeled host subtotal)` as though allocation estimates were overlapping observations. That understates simultaneously live modeled allocations and is not a native or shipping qualification.

The corrected source-bound CPU report is `shipping-allocation-preflight-corrected-056.json` (SHA-256 `4a328687615caa53ecf0bd8d21798a46df609d4a9ea3e2ec9a97cd6ac420dbfd`). Regenerate it with:

```sh
python3 matter/tools/cardboard_shipping_allocation_preflight.py \
  --superseded-source54 /Users/home/cardboard-implementation-20261014/shipping-allocation-preflight-source054.json \
  --superseded-source55 /Users/home/cardboard-implementation-20261014/shipping-allocation-preflight-source055.json \
  --output /Users/home/cardboard-implementation-20261014/analysis/shipping-count-kernel-admission-056/shipping-allocation-preflight-corrected-056.json
```

The corrected combined model is `estimatedMetalPeakBytes + hostMeshAndCompileBytes + hostCheckpointBytes`. The Metal subtotal includes the upload-staging reserve and one snapshot-readback reserve. The host subtotal also reports those two reserves, but the combined estimate does not add that duplicated subtotal; this counts each shared reserve once. Host mesh/compile buffers and retained complete checkpoint copies are distinct host allocations and are added to the Metal envelope.

For the prebuild 12-proxy illustration, modeled Metal is 15,264,608,176 bytes and the host subtotal is 9,179,465,152 bytes. Each is below 16 GiB, while the additive estimate is 21,552,064,816 bytes. This is a planner-level prebuild scenario, not an initialized exact shipping mesh.

For the exact authored CPU topology scenarios, the corrected 131,072-capacity, compact-surface model is 11,946,009,962 bytes before the optional tangent-cache add-on; it is under the 16-GiB allocation-model threshold, but required private arenas remain unadmitted. Default contact capacity is over the allocation limit. The report separately records pure allocation-model status, required-arena status, and overall planner admission. It makes no claim that a full-world runtime initialized, that the modeled estimate equals Metal peak, or that shipping physics is qualified.

The runtime-observation rule remains separate and unchanged: compare initialized runtime resident bytes, Metal `currentAllocatedSize`, and available host footprint by maximum because those measured counters may overlap on unified memory. The additive rule applies to the modeled allocation inventory, not those observed counters.

CPU validation: `cmake --build build-cardboard --target numi-matter-cardboard-shipping-memory-check numi-matter-cardboard-probe -j2`; `./build-cardboard/numi-matter-cardboard-shipping-memory-check` passed; `python3 -m py_compile matter/tools/cardboard_shipping_allocation_preflight.py` passed; regenerated report passed JSON parsing and contains source54/source55 SHA bindings. No Metal runtime was initialized and no GPU test was run.

Current-source SHA-256 (after the probe RAP recomputation fix):

- `matter/tools/cardboard_shipping_memory.hpp`: `5ee051c26bfa0d31544999454b07b12f75fa9d95977030f7aba4bbb5fb94903e`
- `matter/tools/cardboard_probe.mm`: `d94fe0aa982de7aa950963faabfef4a596345d19cc68fe9dbf75c7ff37ba6984`
- `matter/tools/cardboard_shipping_memory_check.cpp`: `fa7d3e6e8fbc2120214cdf23e132074f1c51474bf5b052833ac4b07b4abbcb0a`
- `matter/tools/cardboard_shipping_allocation_preflight.cpp`: `2da03189878ceb8ea85a46c21b524949dc614cb6c6da0fee5a0fb8e16d346dff`
- `matter/tools/cardboard_shipping_allocation_preflight.py`: `7202f142aace21e0556189dfdad2747129f95cbfa97fd73848cec62193f0de3f`
