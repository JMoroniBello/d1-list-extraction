# Upstream sources

The source under `src/d1_list_extraction/_vendor/liquidonnx/` is copied without
modification from [Liquid4All/onnx-export](https://github.com/Liquid4All/onnx-export),
commit `1712505e7e0c312e0fef5908c187a4e7c6f68121`, under the LFM Open License v1.0.
It is bundled to preserve the exact tested host implementation without requiring
the unrelated generation/export dependencies of the full upstream distribution.

Model files are downloaded from [LiquidAI/d1-2B-ONNX](https://huggingface.co/LiquidAI/d1-2B-ONNX),
revision `7f5c6801cba6164dda9daae925178f89717b960f`, under its LFM Open License.
They are not included in this Git repository. `model.lock.json` records the model
revision, host-file hashes, graph hash, weight size and official LFS weight hash.

The model checkpoint includes Python host code. The loader verifies its pinned
file hashes before importing it. Downloaded upstream files retain their own
license and notices. Our adapter subclasses the upstream engine and leaves the
vendored runtime files unchanged.
