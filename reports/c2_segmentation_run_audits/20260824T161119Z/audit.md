# C2 deterministic segmentation promotion audit

| Dataset | Model | Adaptation | Seed | Status | Best epoch |
|---|---|---|---:|---|---:|
| cloudsen12 | dofa | frozen | 42 | PROMOTABLE | 16 |
| cloudsen12 | dofa | frozen | 43 | PROMOTABLE | 24 |
| cloudsen12 | dofa | frozen | 44 | PROMOTABLE | 24 |
| cloudsen12 | panopticon | frozen | 42 | PROMOTABLE | 29 |
| cloudsen12 | panopticon | frozen | 43 | PROMOTABLE | 21 |
| cloudsen12 | panopticon | frozen | 44 | PROMOTABLE | 16 |
| cloudsen12 | dofa | full_finetune | 42 | PROMOTABLE | 50 |
| cloudsen12 | dofa | full_finetune | 43 | PROMOTABLE | 49 |
| cloudsen12 | dofa | full_finetune | 44 | BLOCKED | 42 |
| cloudsen12 | panopticon | full_finetune | 42 | PROMOTABLE | 44 |
| cloudsen12 | panopticon | full_finetune | 43 | PROMOTABLE | 39 |
| cloudsen12 | panopticon | full_finetune | 44 | PROMOTABLE | 44 |
| spacenet7 | dofa | frozen | 42 | PROMOTABLE | 23 |
| spacenet7 | dofa | frozen | 43 | PROMOTABLE | 31 |
| spacenet7 | dofa | frozen | 44 | PROMOTABLE | 44 |
| spacenet7 | panopticon | frozen | 42 | PROMOTABLE | 43 |
| spacenet7 | panopticon | frozen | 43 | PROMOTABLE | 50 |
| spacenet7 | panopticon | frozen | 44 | PROMOTABLE | 49 |
| spacenet7 | dofa | full_finetune | 42 | PROMOTABLE | 23 |
| spacenet7 | dofa | full_finetune | 43 | PROMOTABLE | 16 |
| spacenet7 | dofa | full_finetune | 44 | PROMOTABLE | 24 |
| spacenet7 | panopticon | full_finetune | 42 | PROMOTABLE | 40 |
| spacenet7 | panopticon | full_finetune | 43 | PROMOTABLE | 23 |
| spacenet7 | panopticon | full_finetune | 44 | PROMOTABLE | 42 |

Promotable: **23/24**.

- `cloudsen12/dofa/full_finetune/seed44`: deep export confusion matrix mismatch

## Preserved non-candidate attempts

- `cloudsen12/dofa/full_finetune/seed42` — `20260823T211934923213Z_cloudsen12_dofa_full_finetune_seed42_0238c9b8`: **NONCANDIDATE_ZERO_EPOCH_INITIALIZATION**; initialization audits exist, but no completed epoch, checkpoint, gradient audit, or run summary exists
- `spacenet7/dofa/full_finetune/seed42` — `20260823T211934803653Z_spacenet7_dofa_full_finetune_seed42_5e4e4aac`: **NONCANDIDATE_ZERO_EPOCH_INITIALIZATION**; initialization audits exist, but no completed epoch, checkpoint, gradient audit, or run summary exists
