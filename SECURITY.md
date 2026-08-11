# Security Policy & Threat Model

## Supported versions

`photorand` is actively developed on `main`. Security fixes target the latest
release and the `main` branch; no separate LTS lines are maintained.

| Version | Supported |
|---------|-----------|
| `main` / latest release | ✅ |
| Older releases | ❌ |

---

## Project status: research & educational

`photorand` is a **research and educational** True Random Number Generator
(TRNG) that harvests physical entropy from digital camera sensors. It is
designed to demonstrate and study entropy-extraction techniques — NIST
SP 800-90B estimation, SP 800-90A DRBG construction, and continuous health
testing — in real code.

**It is not certified, audited, or endorsed for any security-sensitive use.**

---

## What this tool IS

- **A TRNG that measures before it conditions.** The raw sensor-noise pool is
  assessed with NIST SP 800-90B min-entropy estimators *before* SHA3-512
  conditioning. The conditioner refuses to emit more bytes than the source was
  measured to contain (`max_seed_bytes`), and rejects sources that fail health
  checks or fall below the entropy floor.
- **Conservative (non-IID) entropy estimation.** Min-entropy is the minimum of
  the Most Common Value, Collision, and Markov (non-IID) estimators — the most
  conservative bound, suitable for correlated sensor data.
- **Sampler-stage degeneracy rejection.** A pool that collapses to a single
  symbol (all-same) is rejected at sampling time, before it can become a
  uniform-looking-but-empty digest.
- **A reseedable, backtrack-resistant DRBG.** The CSPRNG layer
  (`PhotoRandEngine`) uses an HMAC-DRBG (SP 800-90A §10.1.2) with
  post-generation state mixing, periodic reseeding from OS entropy, and
  optional prediction resistance.
- **Continuous runtime health testing.** Repetition Count and Adaptive
  Proportion tests (SP 800-90B §4.4) run on the DRBG output stream, so a
  source that goes bad mid-run is detected.

## What this tool IS NOT

- **Not FIPS 140-2 / 140-3 certified.** It has not been through any
  cryptographic-module validation programme.
- **Not Common Criteria certified.** No evaluation assurance level has been
  assigned.
- **Not a certified entropy source.** No entropy source validation (e.g.
  SP 800-90B Entropy Source Validator) has been performed.
- **Not independently audited.** There has been no third-party security audit.
- **Not proof of entropy.** The min-entropy estimators are **statistical
  estimates**, not proofs. They reduce the risk of over-claiming entropy but
  cannot *guarantee* a lower bound on the physical entropy of the sensor.
- **Not side-channel hardened.** No defences are provided against timing,
  power, electromagnetic, or memory-disclosure side channels.

> **Bottom line:** do **not** use `photorand` to generate keys, seeds, tokens,
> or any value that protects real assets without an independent, qualified
> security review appropriate to your threat model. For production
> cryptographic randomness, use a vetted, certified entropy source and DRBG.

---

## Threat model

| Assumption / scope | Detail |
|---|---|
| **Trusted host** | The code assumes it runs on a host that is not compromised at runtime. There is no protection against an attacker who can read process memory, hijack execution, or tamper with the Python interpreter or NumPy/OpenCV libraries. |
| **Genuine sensor noise** | Entropy quality depends entirely on the physical noise floor of the camera sensor. A sensor under attacker control (e.g. a virtual camera streaming a crafted, low-entropy video), a saturated/flat capture, or a heavily compressed source can yield little or no entropy. The estimators and health checks mitigate — but cannot fully prevent — this. |
| **RAW / webcam integrity** | The pipeline assumes the ingested data is genuinely raw sensor output. If the OS, driver, or capture library silently denoises, compresses, or substitutes data, the extracted "entropy" may be far lower than estimated. |
| **DRBG backtracking resistance** | HMAC-DRBG updates state after each generate (SP 800-90A §10.1.2.2), so a single-point state compromise cannot reveal *past* output. Prediction resistance (reseed-before-generate) is optional. |
| **Forward security** | If the full DRBG state and the OS entropy source are both compromised *at the same time*, future output may be predictable. This is inherent to any DRBG without a hardware entropy source on the generate path. |
| **No network exposure** | `photorand` performs no network operations. The only I/O is reading a local RAW image or the local webcam. |

### Non-security uniqueness tweak

`PhotoRandEngine(salt=True)` mixes a timestamp and PID into the seed so that
two engines sharing a seed produce divergent streams. **This is explicitly
not a security measure** — both values are predictable. All cryptographic
strength comes from the seed entropy and the HMAC-DRBG.

---

## Known limitations

1. **Estimators can over- or under-estimate.** Statistical estimators are
   fallible, especially on short or pathological inputs. A "GOOD" status is a
   best-effort assessment, not a guarantee.
2. **No continuous entropy estimation.** Min-entropy is estimated once at
   capture/conditioning time. Continuous health tests monitor for catastrophic
   failures (stuck-at, drift) but do not continuously re-estimate entropy.
3. **Software-only environment.** Unlike a hardware TRNG, there is no
   protected entropy source or tamper-resistant boundary.
4. **Certification is out of scope for code.** FIPS 140-2/3 or Common
   Criteria certification requires hardware, firmware, physical security, and
   accredited-lab evaluation that cannot be achieved in a software-only
   project. These are documented here as known limitations, not as planned PRs.

---

## Reporting a vulnerability

If you believe you have found a security-relevant bug (e.g. a way to make the
conditioner emit more entropy than was measured, a DRBG state-recovery
weakness, or a bypass of a health check), please **do not open a public
issue**.

Instead, report it privately:

1. Open a **private security advisory** via GitHub's
   ["Report a vulnerability"](https://github.com/SynthLuvr/photorand/security/advisories/new)
   tab, **or**
2. Email the maintainer directly.

Include a description, reproduction steps, and your assessment of impact.
You will receive an acknowledgement within a reasonable timeframe. Public
disclosure is coordinated after a fix is available.

---

## Summary

`photorand` is a well-instrumented study of entropy extraction and DRBG
construction that follows NIST SP 800-90A/B/C methodology as closely as a
software-only, uncertified project reasonably can. It is suitable for
**learning, experimentation, and non-adversarial research**. It is **not** a
certified cryptographic random source and must not be relied upon to protect
real-world assets without independent, qualified review.
