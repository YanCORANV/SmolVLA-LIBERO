# Selected post-training failure review

Eight cases were selected by explicit criteria: the three models' original A at
state 0, untuned and augmented B/C at state 0, and augmented A's new success at state 1.
This is a purposive review, not an estimate of failure-category frequency across
all 36 episodes. The review uses both cameras at steps 0, 40, 80, 120, 160, 200, 240
and 280, plus all 280 object-position records per selected episode. It is an
assistant-authored annotation, not independent human agreement or continuous-video
contact verification.

| Model / state / wording | Maximum target-bowl height rise | Observed outcome and interpretation |
| --- | ---: | --- |
| Untuned / 0 / A | 116.1 mm | Bowl transferred to plate; native goal passes. |
| Original-language tuning / 0 / A | 0.1 mm | No meaningful bowl lift; arm subsequently operates near plate without completing the transfer. Grasp/lift failure. |
| Augmented / 0 / A | 1.6 mm | No sustained lift/transfer; plate is also disturbed. Grasp/lift failure. |
| Untuned / 0 / B | 4.9 mm | Target remains near tabletop; transfer incomplete. Grasp/lift failure. |
| Augmented / 0 / B | 4.7 mm | Target remains near tabletop despite movement toward plate. Grasp/lift failure. |
| Untuned / 0 / C | 40.8 mm | Target is lifted/moved toward plate, but native goal never passes. Transfer/placement incomplete. |
| Augmented / 0 / C | 36.7 mm | Transient target lift; target returns to tabletop and transfer is incomplete. Acquisition/retention/transfer failure; contact data would be needed to distinguish them. |
| Augmented / 1 / A | 99.0 mm | Bowl transferred to plate; native goal passes. |

Height rise is measured relative to the logged object position at **step 1**, not the
reset frame. It is descriptive, not a grasp predicate. The alternative bowl and
ramekin have negligible vertical motion in these cases. Selected frames show motion
around the requested target and plate; they do not show a clear alternative-object
pickup. This does **not** establish that language was correctly understood. Position
logs and sparse frames cannot recover model intent or identify a causal mechanism.

The shared pattern matters for the next experiment: B/C failures are not solely
obvious wrong-object selections, and original-language tuning can also impair
grasp execution. We will try a smaller learning-rate peak with all other training
controls fixed, rather than assuming more updates alone will help. This is a
testable optimization hypothesis, not a root-cause diagnosis.

[Machine-readable annotations](qualitative_review.json) retain source hashes and
per-case measurements. To regenerate the eight local sheets without loading a model
or simulator:

```bash
python scripts/review_adaptation_rollouts.py
```

Local output: `outputs/adaptation_review/task0_adaptation_eval/`. On continuation the
script asks before reusing hash-verified evidence. Raw videos remain in the original
evaluation directory; they are not committed to Git.
