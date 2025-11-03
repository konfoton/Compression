## CUDA Run-Length Encoding (RLE) — GPU-only

This project implements run-length encoding (RLE) compression and decompression entirely on the GPU using CUDA. It showcases an efficient, thrust-free, cross-block scan to compute global prefix sums, which is used to derive run boundaries and reconstruct the original data.

Highlights:
- No Thrust: all scans and reductions are implemented from scratch with CUDA kernels.
- Cross-block scans: block-level scans + a second-level scan over per-block totals + offset add-back.
- End-to-end pipeline: compression to (symbols, counts) and decompression back to the original sequence.
- Comes with a simple host-side self-check that validates decompression equals the input.

---

## Project layout

```
Makefile
README.md
include/
	kernels/
		runarray.cuh        # Kernel declarations (interfaces)
src/
	main.cu               # Orchestrates compression & decompression passes
	kernels/
		runarray.cu         # Kernel implementations (scans, mark flags, mapping, etc.)
build/
	app                   # Built binary (created by make)
```

---
