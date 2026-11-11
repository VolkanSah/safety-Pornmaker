---
title: Safe Adult Image Studio
emoji: 🎨
colorFrom: gray
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
short_description: Private 18+ image generator with child-safety layers
hf_oauth: true
tags:
- not-for-all-audiences
sdk_version: 6.31.0
---

# Adult Image Studio (18+)

A small text-to-image tool for **adults**. It can produce adult (NSFW) imagery of fictional adult
persons and runs on ZeroGPU. Because that is a sensitive use case, the safety design is part of the
product, not an afterthought.

**Who may use it:** adults only (18+), signed in with a Hugging Face account.
**Not allowed, and actively blocked:** anything that depicts or suggests minors, violence/gore,
non-consensual or degrading content. Please do not use real people's names or likenesses
(this rule is **not** technically enforced; see "Limits").

## Models

| Model | Notes |
|---|---|
| SDXL checkpoint `votepurchase/pornmasterPro_realismILV4` (default) | Uses long-prompt support (Compel). Check the model card for its license. |
| FLUX.1-dev (`black-forest-labs/FLUX.1-dev`) | Gated model. License: FLUX.1 [dev] Non-Commercial. Natural-language prompts, **no negative prompt**. |

Stable Diffusion 3.5 Medium was deliberately left out: its acceptable use policy appears to forbid
sexual-act content.

## Safety measures (several layers, none of them perfect on its own)

1. **Sign-in required.** Every request belongs to a Hugging Face account.
2. **Text filter** (`safety.py`) on every prompt: blocks words and spellings that suggest minors
   (English and German; also numeric ages under 18, spaced-out letters, leetspeak), childlike *body*
   descriptions ("age-washing" such as an adult age combined with a child's body), violence/gore,
   and religious places and garments (sexual imagery in that context is harassment).
3. **Negative-prompt guard:** age-related words (e.g. "adult", "mature") are not allowed in the
   negative prompt, so nobody can cancel the next layer.
4. **Forced hardening on the server:** an "adult, 25+" instruction is added to every prompt and a
   "no children / minors / teen / young" list to every negative prompt. Users cannot remove it.
5. **Age check on the finished image** (`nateraw/vit-age-classifier`). If it looks young, or the
   checker fails, no image is shown (fail-closed). False alarms are possible by design.
6. **Logging and automatic bans.** Violations are logged. Repeated hits of the serious categories
   lead to an account ban (default: 3 strikes).

## Limits (please read)

- A word filter only knows words it has seen. New slang, other languages and character tricks
  can slip through; layers 4-6 exist to catch what layer 2 misses.
- The image check is weak on faces that are small or hidden, and it can wrongly flag some adults.
- Real people are not detected. Do not create images of real persons.
- This is a hobby project, not a guarantee. Reports of gaps are welcome and get fixed.

## Logging and privacy

Stored in a **private** dataset: timestamp, account name, violation category, and a short hash of
the prompt. Prompt text is **not** stored unless the owner switches that on for blocked prompts.
**Generated images are not stored.** Logs are used only for abuse handling.

## Report a problem

Use the **Community** tab of this Space. Anything involving minors can also be reported to your
national hotline, for example [internet-beschwerdestelle.de](https://www.internet-beschwerdestelle.de)
in Germany.

## For the owner: Space settings

| Name | Type | Meaning |
|---|---|---|
| `HF_TOKEN` | secret | Read token of an account that accepted the FLUX.1-dev license |
| `LOG_REPO` | variable | e.g. `yourname/abuse-log`; private dataset for logs and the ban list |
| `BAN_AFTER` | variable | Strikes until a ban (default 3) |
| `LOG_PROMPT_TEXT` | variable | `1` = also store the first 300 characters of blocked prompts (default off) |
| `REQUIRE_LOGIN` | variable | `0` = test mode without sign-in. Keep at default (login required) for anything public. |

---

## Kurzfassung (Deutsch)

Ein Bild-Generator nur für **Erwachsene** (18+), Anmeldung mit Hugging-Face-Account Pflicht.
Schutzschichten: Textfilter (Minderjährige, kindliche Körperbeschreibungen, Gewalt, religiöse
Orte), Schutz des Negativprompts, erzwungener "Erwachsene"-Zusatz, Alterscheck am fertigen Bild
(fail-closed) sowie Logging mit automatischer Sperre. Gespeichert werden nur Zeit, Account-Name,
Kategorie und ein Hash des Prompts, keine Bilder. Kein Filter ist perfekt; Meldungen über Lücken
bitte im Community-Tab.

**Cheers**
