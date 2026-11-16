The native S4 backend needs NVIDIA CUTLASS headers. From the repository root:

```bash
git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git third_party/cutlass
```

Alternatively set `CUTLASS_PATH` to an existing CUTLASS v3.5.1 checkout.
The `int8-fused` backend does not require CUTLASS. The downloaded checkout is
ignored by Git and retains its upstream license.
