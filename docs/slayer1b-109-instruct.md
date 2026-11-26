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

## Benchmark instrukcji (`scripts/bench_instruct.py`)

Sprawdzenia są deterministyczne (parsowanie JSON, regex, liczniki), bez sędziego-LLM. Zestawy
odtwarzają twierdzenia z karty modelu, żeby zweryfikować je na własnym runtime.

```bash
python scripts/bench_instruct.py --model-dir models/Slayer1B-1.09-Instruct --only format tool
python scripts/bench_instruct.py --model-dir models/Slayer1B-1.09-Instruct --only loops --loop-tokens 192
```

| zestaw | wynik |
|---|---|
| format (6 sprawdzeń: sam JSON, jedno zdanie, 3 numerowane pozycje, limit 20 słów, po angielsku, bez powtarzania pytania) | **3/6 = 50%** |
| tool-call (6 sprawdzeń: 4 wywołania + 2 przypadki „bez narzędzia”) | **3/6 = 50%** |
| pętle, greedy, 192 tokeny | powtórzenia 0,26, **domknięte 0/3**, 3,1 tok/s |
| pętle, T=0,3 / top-p 0,9 / kara 1,1 | powtórzenia 0,02, **domknięte 0/3**, 3,2 tok/s |

Dwa przebiegi format+tool dały identyczny wynik (3/6, 3/6) — zestaw jest powtarzalny.

Co się powtarza z karty: **format wywołania** jest poprawny (2/2 razy, gdy model zdecyduje się
wywołać: `<tool_call>{"name": "calculator", "arguments": {"expression": "17 * 23"}}</tool_call>`),
a greedy powtarza znacznie więcej niż próbkowanie z karą (0,26 vs 0,02) — stąd zalecenie karty.

Co się nie powtarza / co jest gorsze:

- „sam JSON” → model pisze meta-instrukcję („Niech «kod» będzie tekstem w języku polskim…”), nie JSON;
- „3 pozycje z numeracją” → jedna linia z `1. **Numeracja**: 1, 2, 3…`;
- limit 20 słów → przekroczony;
- G53 = „skrót od «Gotowe do użycia»” — halucynacja;
- pytanie o narzędzie: „Która godzina w Krakowie?” → odpowiedź z pamięci („**12:00**”) bez wywołania;
- pytanie o dokumentację G54 → wymyślone „komenda w języku C#… system kontroli wersji Git”;
- „Napisz krótkie powitanie” → wywołanie `get_time` (karta sama to opisuje jako utratę powściągliwości).

`domknięte 0/3` przy 192 tokenach oznacza, że model **nie emituje** `<|im_end|>` i pisze do
wyczerpania budżetu; klient czatu musi sam ucinać długość (tak robi `slayer_chat.py`).

## Panel lm-eval i serwer OpenAI-compatible

Paczka ma teraz `load_slayer.py` z sygnaturą `load(path, device)` (`scripts/load_slayer_109.py`),
bo bazowy loader 1.09 ma `load(device, update)` i pobiera `wagi_<update>.pt` — na naszej paczce
kończy się to `ValueError: Published updates: 5000 through 45000 in steps of 5000`.
Po podmianie loader-a `slayer_kv_lm.model_from_package(pkg, "cpu")` ładuje paczkę w 19 s
(`block=4096`, `vocab=32768`, `head` związany z `tok`), więc otwarte są:

```bash
# panel 50 zadań (wymaga: pip install lm-eval)
python slayer_kv_lm.py --pkg models/Slayer1B-1.09-Instruct --tasks arc_challenge

# serwer OpenAI-compatible (127.0.0.1:8001) i pomiar przez API
python slayer_server.py --pkg models/Slayer1B-1.09-Instruct --port 8001 --log reports/api-log
```

Uwagi metodologiczne: panel lm-eval jest w większości loglikelihood-owy, więc mierzy wiedzę, a
nie posłuszeństwo instrukcjom; `generate_until` w adapterze jest greedy i **bez szablonu ChatML**,
więc wynik instructa będzie zaniżony — do uczciwego porównania trzeba dodać szablon do adaptera.

## GPU (Colab T4) i panel lm-eval

Notebook: `slayer1b_gpu_bench.ipynb` w repo (18 komórek) — klon gałęzi, budowa paczki, nasz zestaw,
panel. Wyniki z przebiegu na T4:

| pomiar | CPU fp32 | T4 fp32 (Colab) |
|---|---|---|
| pętle, greedy, 192 tokeny | 3,1 tok/s | **27,8 tok/s** |
| pętle, T=0,3 / kara 1,1 | 3,2 tok/s | **31,0 tok/s** |
| powtórzenia (greedy / sampled) | 0,255 / 0,022 | 0,255 / 0,022 |
| domknięte tury (długie odpowiedzi) | 0/3 | **0/3** |

Czyli GPU daje ~9× prędkości i **nie zmienia zachowania**: te same wskaźniki powtórzeń i nadal brak
`<|im_end|>` przy długich odpowiedziach. Domknięcie zależy od długości odpowiedzi, nie od sprzętu:
krótkie pytania czatu zamykały się 4/5 przy budżecie 64 tokenów.

Panel (adapter Slayera, raw continuation, 0-shot, `--limit 100`, batch 4, `lm_eval 0.4.13`,
KV cache, 102 s):

| zadanie | acc | acc_norm |
|---|---|---|
| `arc_challenge` | 30,0% | 36,0% |
| `piqa` | 73,0% | 74,0% |

Losowy poziom to 25% (arc) i 50% (piqa), więc arc ledwo ponad przypadek, piqa wyraźnie lepiej.
Przy 100 dokumentach niepewność to ~±5 pkt, a adapter podaje prompt **bez szablonu ChatML**, więc
instruct jest tu mierzony jak baza — liczby służą do porównania z bazą 1.09, nie do oceny czatu.
Uwaga: `gpu-format-tool.json` nie dotarł w archiwum, więc format/tool na GPU nie są zmierzone;
na CPU jest 3/6 i 3/6.

## Dlaczego 3/6 nie przeczy 92% z karty

Własny zestaw autorów (`training/results/sft_step1605.json`) raportuje: `chat_format_ok` 69,33%
(300 przypadków), `tool_replay_ok` 92,47% (385), `loop_end_to_end` 62,96% (60),
`loop_no_tool_correct` 66,67%.

`tool_replay` to **odtworzenie wywołania z wymuszonym prefiksem** `<tool_call>` — mierzy mechanikę
formatu, nie decyzję. Nasz zestaw mierzy decyzję i ugruntowanie w dowodach na 6 przypadkach:
2/2 poprawne, gdy model zdecyduje się wywołać (JSON i argumenty w porządku), ale 2/4 decyzji błędne
(godzina z pamięci, wymyślona dokumentacja) plus 1/2 fałszywe wywołanie przy small talku. Obie
liczby mogą być prawdziwe: format jest dobry, autonomia zawodzi.
