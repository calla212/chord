# Chord

Cooperative scheduling for multi-instance LLM serving with NVIDIA Dynamo and vLLM.

Requires Python 3.11, CUDA 13, compatible PyTorch, C/C++ build tools, Rust,
protobuf, etcd, and the Barex package listed in [sources.lock.json](sources.lock.json).

Check out the pinned dependency revisions into `.deps/dynamo`, `.deps/vllm`, and
`.deps/llumnix-kv`. From this repository's root, apply and build:

```bash
CHORD_ROOT="$PWD"
git -C .deps/dynamo apply "$CHORD_ROOT/patches/dynamo-integration.patch"
cp -R overlay/. .deps/dynamo/
git -C .deps/vllm apply "$CHORD_ROOT/third_party/vllm-chord/patches/0001-feat-scheduler-support-Chord-waiting-request-resume.patch"
git -C .deps/llumnix-kv apply "$CHORD_ROOT/third_party/llumnix-kv/patches/vllm-0.23-cuda13.patch"
python -m pip install -r .deps/vllm/requirements/build.txt
VLLM_VERSION_OVERRIDE=0.23.0+chord python -m pip install --no-build-isolation -e .deps/vllm
python -m pip install 'maturin[patchelf]'
(cd .deps/dynamo/lib/bindings/python && maturin develop --release --locked)
python -m pip install -e '.deps/dynamo[vllm]'
python -m pip wheel --no-deps --no-build-isolation --wheel-dir .deps/wheels .deps/llumnix-kv
python -m pip install --no-deps .deps/wheels/blade_kvt-*.whl
```

Export [configs/chord/chord-reference.env](configs/chord/chord-reference.env) in
both worker and frontend environments. Configure worker/control endpoints,
separate GPU and transfer ports, and a shared etcd endpoint and namespace.
Start aggregate workers with `python -m dynamo.vllm` and the frontend with
`python -m dynamo.frontend --router-mode kv`. Use `--help` for model and topology options.

[Apache-2.0](LICENSE).
