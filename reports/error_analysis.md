# Where judges disagree with human experts

Agreement with expert votes, ties included (S1), as % with a 95 % bootstrap interval over questions (2000 resamples) and the number of judge/expert vote pairs in brackets. Turn 1 unless stated. Categories have only 10 questions each, so their intervals are wide.

## By MT-Bench category

| Judge | writing | roleplay | reasoning | math | coding | extraction | stem | humanities |
|---|---|---|---|---|---|---|---|---|
| llama-3.2-3b (swap, flip -> tie) | 52 [36, 68] (85) | 56 [41, 69] (124) | 54 [33, 74] (72) | 58 [39, 76] (102) | 62 [50, 72] (87) | 42 [26, 59] (86) | 55 [47, 67] (106) | 56 [46, 66] (93) |
| llama-3.2-3b (swap, abstain) | 63 [50, 76] (62) | 83 [77, 89] (71) | 56 [26, 82] (39) | 63 [44, 80] (63) | 76 [67, 85] (50) | 83 [62, 100] (29) | 73 [65, 86] (63) | 67 [54, 88] (58) |
| qwen2.5-7b (swap, flip -> tie) | 52 [33, 71] (48) | 48 [25, 70] (67) | 50 [31, 72] (44) | 47 [29, 73] (45) | 62 [33, 80] (40) | 55 [40, 70] (44) | 58 [46, 74] (48) | 52 [44, 71] (46) |
| qwen2.5-7b (swap, abstain) | 65 [44, 82] (37) | 74 [47, 96] (42) | 54 [30, 76] (35) | 54 [37, 74] (26) | 67 [42, 82] (30) | 79 [63, 100] (29) | 81 [72, 92] (27) | 58 [43, 93] (24) |
| GPT-4 (same keys as the local judges) | 58 [39, 77] (48) | 76 [61, 89] (67) | 75 [56, 95] (44) | 78 [63, 89] (45) | 62 [44, 76] (40) | 68 [54, 94] (44) | 65 [51, 78] (48) | 59 [46, 81] (46) |
| GPT-4 (all turn-1 keys) | 63 [51, 75] (190) | 66 [54, 76] (234) | 70 [57, 82] (218) | 71 [56, 82] (207) | 59 [53, 65] (200) | 69 [61, 79] (203) | 68 [62, 73] (210) | 67 [58, 75] (227) |
| GPT-4 (all keys, turns 1-2) | 61 [54, 68] (378) | 65 [56, 72] (464) | 65 [56, 74] (434) | 66 [54, 76] (412) | 60 [55, 65] (397) | 71 [64, 78] (402) | 72 [66, 76] (416) | 70 [65, 75] (452) |

## By the expert's vote

| Judge | expert decisive | expert tie |
|---|---|---|
| llama-3.2-3b (swap, flip -> tie) | 53 [46, 59] (571) | 60 [50, 69] (184) |
| llama-3.2-3b (swap, abstain) | 85 [80, 89] (355) | 8 [1, 16] (80) |
| qwen2.5-7b (swap, flip -> tie) | 56 [47, 65] (288) | 43 [29, 56] (94) |
| qwen2.5-7b (swap, abstain) | 85 [77, 91] (190) | 10 [2, 22] (60) |
| GPT-4 (same keys as the local judges) | 73 [65, 80] (288) | 53 [38, 68] (94) |
| GPT-4 (all turn-1 keys) | 71 [66, 75] (1284) | 53 [44, 61] (405) |
| GPT-4 (all keys, turns 1-2) | 71 [67, 74] (2575) | 51 [43, 58] (780) |

## By expert consensus (keys with 2+ expert votes)

| Judge | experts unanimous | experts split | single vote |
|---|---|---|---|
| llama-3.2-3b (swap, flip -> tie) | 62 [52, 73] (315) | 44 [39, 49] (261) | 56 [48, 64] (179) |
| llama-3.2-3b (swap, abstain) | 85 [76, 92] (204) | 52 [47, 56] (124) | 65 [56, 75] (107) |
| qwen2.5-7b (swap, flip -> tie) | 60 [47, 74] (159) | 37 [29, 45] (130) | 61 [51, 72] (93) |
| qwen2.5-7b (swap, abstain) | 78 [65, 89] (123) | 45 [33, 55] (66) | 67 [56, 79] (61) |
| GPT-4 (same keys as the local judges) | 86 [76, 94] (159) | 50 [44, 56] (130) | 63 [52, 75] (93) |
| GPT-4 (all turn-1 keys) | 84 [77, 90] (707) | 45 [42, 48] (557) | 66 [61, 72] (425) |
| GPT-4 (all keys, turns 1-2) | 82 [77, 86] (1406) | 46 [43, 49] (1096) | 67 [63, 71] (853) |

## By strength gap of the model pair

| Judge | 0.00-0.15 | 0.15-0.30 | 0.30-1.00 |
|---|---|---|---|
| llama-3.2-3b (swap, flip -> tie) | 47 [38, 57] (199) | 50 [42, 59] (215) | 62 [53, 70] (341) |
| llama-3.2-3b (swap, abstain) | 59 [45, 73] (88) | 59 [50, 69] (103) | 80 [73, 86] (244) |
| qwen2.5-7b (swap, flip -> tie) | 34 [25, 46] (116) | 52 [40, 65] (97) | 66 [56, 76] (169) |
| qwen2.5-7b (swap, abstain) | 43 [26, 60] (56) | 66 [49, 85] (53) | 77 [68, 85] (141) |
| GPT-4 (same keys as the local judges) | 59 [49, 69] (116) | 62 [50, 73] (97) | 78 [70, 85] (169) |
| GPT-4 (all turn-1 keys) | 60 [54, 67] (443) | 55 [51, 61] (449) | 76 [71, 82] (797) |
| GPT-4 (all keys, turns 1-2) | 56 [52, 61] (877) | 57 [53, 61] (889) | 77 [72, 81] (1589) |

Strength = a model's expert win rate over all its votes (tie = half): gpt-4 73 %, gpt-3.5-turbo 65 %, claude-v1 65 %, vicuna-13b-v1.2 48 %, alpaca-13b 29 %, llama-13b 15 %.

## Confusion matrices (rows: judge, columns: expert; first / second / tie)

- llama-3.2-3b (swap, flip -> tie): 168 / 38 / 43; 15 / 133 / 31; 96 / 121 / 110
- llama-3.2-3b (swap, abstain): 168 / 38 / 43; 15 / 133 / 31; 0 / 1 / 6
- qwen2.5-7b (swap, flip -> tie): 87 / 14 / 29; 14 / 74 / 25; 48 / 51 / 40
- qwen2.5-7b (swap, abstain): 87 / 14 / 29; 14 / 74 / 25; 0 / 1 / 6
- GPT-4 (same keys as the local judges): 110 / 17 / 26; 24 / 100 / 18; 15 / 22 / 50
- GPT-4 (all turn-1 keys): 458 / 72 / 87; 77 / 453 / 104; 104 / 120 / 214
- GPT-4 (all keys, turns 1-2): 956 / 149 / 177; 157 / 867 / 206; 229 / 217 / 397

## Verdict flips under A/B swap by category (local judges)

| Judge | writing | roleplay | reasoning | math | coding | extraction | stem | humanities |
|---|---|---|---|---|---|---|---|---|
| llama-3.2-3b | 20 (50) | 42 (57) | 44 (36) | 41 (54) | 42 (50) | 60 (47) | 38 (55) | 42 (45) |
| qwen2.5-7b | 21 (29) | 34 (29) | 19 (21) | 37 (27) | 27 (22) | 25 (24) | 48 (25) | 43 (23) |
