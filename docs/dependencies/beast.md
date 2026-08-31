# Qualification report: beast (Boost.Beast)

- Manifest entry: `beast` — qualified-deferred, feature `http-server` (M9), owner `server`
- Pin: `cbbabdec878641ddf83a9be24f75666742cd2b8d` (uninitialized in the core profile)
- License: BSL-1.0, `third_party/beast/LICENSE_1_0.txt`

## 1. Which InferX contract would use it?

HTTP/SSE transport for the OpenAI-compatible server layer (M9, plan section 13), over
Boost.Asio.

## 2. Required now, deferred, experimental, or rejected?

Deferred to M9. Not initialized by any M0 profile; no M0 target reads this directory.

## 3. Source and transitive dependencies

Beast is a header-only Boost module and **not a complete dependency story**: it requires
the Boost.Asio/System (and transitive Boost.Utility/Assert/...) header closure. The
qualified introduction must pin a Boost release (superproject or enumerated modules) with
its own manifest entries and reports. This gap is the reason for deferral.

## 4. Toolchain/C++23 compatibility

Unverified at the pin (audit pending). Beast tracking Boost releases generally supports
GCC 13/Clang 18; must be proven in the M9 qualification with the chosen Boost closure
and C++23 host mode.

## 5. Runtime behavior caveats

Header-only templates; the runtime behavior (threads, exceptions via Asio handlers,
global state) comes from the Boost closure and the server's I/O design, to be audited at
introduction.

## 6. API stability and namespaces

`boost::beast::`; follows Boost release cadence.

## 7. License/notice obligations and security

BSL-1.0 (Boost Software License) — permissive, notice retention on redistribution. No
known security issues relevant to the pinned revision; re-check at qualification.

## 8. Binary/build/startup cost

Header-only template instantiation costs measurable compile time in server TUs; measure
during the M9 spike before accepting.

## 9. Upgrade/rollback procedure

On qualification: pin Boost closure modules, add manifest entries/reports, initialize a
`http-server` profile, build + server tests; upgrades advance gitlinks together with the
Boost set.

## 10. Disposition and approvals

**Qualified-deferred** to M9 per `docs/milestones/m0.md` section 8.3. Owner: `server`.
