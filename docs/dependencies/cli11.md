# Qualification report: CLI11

- Manifest entry: `cli11` — approved, feature `core`, owner `cli`
- Pin: `37bb6edc5317e99af72ef48405e65d9ca5218861` (release tag `v2.6.2`)
- License: BSD-3-Clause, `third_party/CLI11/LICENSE`

## 1. Which InferX contract uses it?

CLI11 registers and parses every user-facing command, option, validator, enum, help page, and version
flag in the single `inferx` executable. It is linked privately by the non-installed `inferx_cli`
library and does not appear in reusable runtime, model, server, benchmark, or installed SDK targets.

## 2. Required now, deferred, experimental, or rejected?

Required when `INFERX_BUILD_CLI=ON`, which is the top-level default. It is not required when InferX is
embedded as a library with the CLI disabled.

## 3. Source and transitive dependencies

CLI11 is header-only in InferX's configuration and depends only on the C++ standard library. Its
tests, examples, documentation, single-header generator, install rules, and precompiled-library mode
are disabled.

## 4. Toolchain/C++23 compatibility

The pinned v2.6.2 release configures as a subproject and compiles the InferX parser with the qualified
GCC 13 and Clang 18 C++23 lanes. InferX consumes the upstream `CLI11::CLI11` CMake target.

## 5. Runtime behavior caveats

Parsing may allocate and CLI11 reports failures with exceptions internally. The `inferx` process
boundary catches parse and unexpected exceptions; no exception crosses into reusable InferX APIs.
Help and parse errors perform no model, accelerator, filesystem, or network initialization.

## 6. API stability and namespaces

The public namespace is `CLI`. InferX confines it to `src/cli` and exposes only InferX-owned typed
configuration and dispatcher interfaces. The release tag and exact gitlink are the compatibility
boundary.

## 7. License/notice obligations and security

BSD-3-Clause notice retention applies. The license is represented in the dependency manifest and
generated source/install SBOM evidence. No known issue changes the accepted local CLI threat model.

## 8. Binary/build/startup cost

CLI11 is header-only and contributes parser code only to `inferx_cli`/`inferx`; it creates no
additional runtime artifact or installed executable.

## 9. Upgrade/rollback procedure

Advance the gitlink and manifest in one change, run manifest validation, build GCC/Clang lanes, and
run all CLI help, invalid-input, configuration-mapping, dispatch, and exit-code tests. Rollback is a
revert of that change.

## 10. Disposition and approvals

**Approved** as the sole command-line parser for `inferx`, private to the CLI layer.
