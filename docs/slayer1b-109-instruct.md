# Slayer1B 1.09 Instruct w tym repo

Instrukcja i pomiar dla `SlayerLab/Slayer1B-1.09-Instruct-26.11` (SFT E3, 1B, kontekst 4096,
ChatML, apache-2.0). Pakiet leży w `models/Slayer1B-1.09-Instruct/` i jest ignorowany przez gita
(`models/`, `*.safetensors`).

## Paczka lokalna

| plik | źródło |
|---|---|
| `config.json`, `tokenizer.json`, `model.safetensors` (4,3 GB fp32) | `SlayerLab/Slayer1B-1.09-Instruct-26.11` @ `c4170ff925ac2deb7de433173909b4b4c4b09306` |
| `train_gpt_ref_r24_2.py`, `load_slayer.py` | `SlayerLab/Slayer1B-1.09` @ `b5fb64ffbd2a6c8852ecb5af2be91d5dfb39e6f1` |

Wagi są wiązane (`head = tok.weight`) i nie leżą w safetensors osobno. Skrypty trenera 1.09
pochodzą z repo bazowego, bo checkpoint zapisano klasą `r24_2`, a nie `r23`; `PROVENANCE.json`
z SHA-256 każdego pliku generuje skrypt budujący paczkę (nie jest wersjonowany).

`scripts/slayer_runtime.py` szuka teraz **najnowszego** `train_gpt_ref_*.py` w katalogu modelu
(`_trainer_source`), więc 1.05/1.06 (r23) i 1.09+ (r24_2) ładują się tą samą ścieżką.

## Format rozmowy

Karta modelu: `<|im_start|>` = 1, `<|im_end|>` = 2, BOS `<|endoftext|>` = 0, stop na `<|im_end|>`,
dekodowanie T=0,3 / top-p=0,9 / kara powtórzeń 1,1 (czysty greedy zapętla się w ~8% odpowiedzi).
`scripts/slayer_chat.py` dokłada to, czego nie ma w runtime: szablon ChatML, BOS, stop na 1/2 i
karę powtórzeń na wygenerowanych tokenach. `--no-cache` wymusza pełny forward, gdyby KV cache
nie pasował do architektury.

```bash
python scripts/slayer_chat.py --model-dir models/Slayer1B-1.09-Instruct \
    --quantization int8 --prompts-file eval/chat_prompts.txt \
    --max-new-tokens 64 --out reports/slayer1b-1.09-instruct-chat.jsonl
```

## Zmierzone na tym komputerze (CPU, bez GPU)

| pomiar | fp32 | int8 (dynamiczna) |
|---|---|---|
| pamięć po załadowaniu | 4,4 GB RSS | 4,2 GB RSS |
| dekodowanie z KV cache | 3,5 tok/s (283 ms/token) | 4,5 tok/s (221 ms/token) |
| prefill, prompt 8 tokenów | 1 375 ms | 846 ms |
| pełny forward, 8 tokenów | 2 145 ms | 8 946 ms |
| parity KV cache vs pełny forward | **7·10⁻⁶** | ~1,0 (patrz niżej) |

Perplexity na `eval/corpus/wiki-pl-20231101-proxy.txt`, identyczne ustawienia
(`--dtype float32 --max-tokens 512 --block 256`):

| model | PPL | czas |
|---|---|---|
| Slayer1B-1.05 (baza) | **16,84** | 15,9 s |
| Slayer1B-1.09-Instruct (po SFT) | **17,26** | 13,2 s |

Wcześniejszy wpis w `reports/slayer1b-1.05-perplexity.json` (PPL 14,67) mierzył bf16 na 2048
tokenach — inna konfiguracja, nieporównywalny. SFT podnosi PPL o 2,5% względnie, co jest
oczekiwanym skutkiem przestawienia rozkładu na rozmowę.

Odpowiedzi na 5 pytań technicznych (`eval/chat_prompts.txt`): 2–14 s każde, tura domknięta,
**treść błędna**.

Parity int8 ≈ 1,0 to nie błąd cache: `quantize_dynamic` kwantyzuje aktywacje per tensor, a
`prefill` liczy głowę na jednej pozycji (`lnf(x[:, -1:])`), więc skala kwantyzacji wejścia do
`_head` różni się od pełnego forwardu. Dla porównania: w fp32 oba tory zgadzają się do 7·10⁻⁶.
Dlatego int8 kupuje tylko ~22% prędkości kosztem wierności — domyślnie używaj **fp32**, a int8
zostaw do pomiarów pamięci.

Przykłady z `eval/chat_prompts.txt` (int8, T=0,3):

- „Co to jest G90 w sterowaniu CNC?” → „G90 to sterowanie CNC, które jest używane do sterowania
  maszynami CNC. G90 to skrót od «G-code»…” — tautologia i błąd rzeczowy;
- „Różnica między G53 a G54” → „G53 to G54 z tym samym kodem.”;
- „Kim jesteś?” → odpowiedź ogólna, bez tożsamości.

## Czego ten model tu nie zastąpi

- Nie jest wystarczający do odpowiedzi domenowych (dokumentacja CNC): myli kody, zmyśla
  definicje. Do tej roli potrzebny model ≥7B albo deterministyczne wyszukiwanie + kontrola źródeł,
  które repo już ma.
- Nie ma GGUF/vLLM — serwowanie wyłącznie przez `train_gpt_ref_r24_2.py` albo
  `research/common/slayer_model.py` z `slayerlabs/benchmarks-fabryka-ai`.
- Na CPU 3–5 tok/s: odpowiedź 250 tokenów to ~1 min. Do czatu interaktywnego potrzebne GPU.
