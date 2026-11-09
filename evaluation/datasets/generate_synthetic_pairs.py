"""Regenerate the controlled synthetic passage-pair dataset (not real submissions)."""
import csv
from pathlib import Path

OUTPUT = Path(__file__).with_name("synthetic_pairs.csv")

QUERIES = [
    ("exact_copy", "exact-01", "A lunar eclipse occurs when Earth passes directly between the Sun and Moon, casting Earth's shadow across the lunar surface."),
    ("exact_copy", "exact-02", "Glaciers move slowly downhill because gravity pulls compacted ice across the underlying land."),
    ("paraphrasing", "paraphrase-01", "Urban green spaces planted with native flowers give pollinators reliable food and places to shelter."),
    ("paraphrasing", "paraphrase-02", "Allowing garden soil to dry slightly between watering reduces waste and encourages plant roots to grow deeper."),
    ("partial_copy", "partial-01", "A strong account security plan combines unique passwords with an additional verification step. Hardware security keys also resist many phishing attacks."),
    ("partial_copy", "partial-02", "Wetlands slow floodwater and store runoff before it reaches nearby rivers. Their varied habitats also support many species."),
    ("common_topic_independent", "common-01", "Coffee grind size affects how quickly water extracts flavor during brewing."),
    ("common_topic_independent", "common-02", "Public charging access can influence whether drivers choose an electric vehicle."),
    ("unrelated", "unrelated-01", "A saxophone produces sound when a reed vibrates against its mouthpiece."),
    ("unrelated", "unrelated-02", "The painter mixed ultramarine pigment with linseed oil before applying it to the canvas."),
]

CANDIDATES = [
    ("exact-target-01", "A lunar eclipse occurs when Earth passes directly between the Sun and Moon, casting Earth's shadow across the lunar surface."),
    ("exact-target-02", "Glaciers move slowly downhill because gravity pulls compacted ice across the underlying land."),
    ("paraphrase-target-01", "City gardens filled with local blossoms provide pollinating insects with dependable nectar and protected places to live."),
    ("paraphrase-target-02", "Watering less often, after the surface soil has had time to dry, can conserve water and promote deeper roots."),
    ("partial-target-01", "A strong account security plan combines unique passwords with an additional verification step. Regular account reviews can reveal suspicious changes."),
    ("partial-target-02", "Wetlands slow floodwater and store runoff before it reaches nearby rivers. These saturated areas can also filter sediment from water."),
    ("common-topic-target-01", "Coffee roasting time changes bean color and aroma as heat drives chemical reactions inside each bean."),
    ("common-topic-target-02", "Electric vehicle batteries store energy chemically and gradually lose capacity over repeated charging cycles."),
    ("unrelated-target-01", "Coral reefs form complex underwater structures that provide habitat for many marine organisms."),
    ("unrelated-target-02", "A ceramic bowl is shaped on a potter's wheel and hardened in a kiln."),
]

POSITIVE_PAIRS = {
    ("exact-01", "exact-target-01"),
    ("exact-02", "exact-target-02"),
    ("paraphrase-01", "paraphrase-target-01"),
    ("paraphrase-02", "paraphrase-target-02"),
    ("partial-01", "partial-target-01"),
    ("partial-02", "partial-target-02"),
}


def main():
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario", "query_id", "query_passage", "candidate_id", "candidate_passage", "relevant"],
        )
        writer.writeheader()
        for scenario, query_id, query in QUERIES:
            for candidate_id, candidate in CANDIDATES:
                writer.writerow({
                    "scenario": scenario,
                    "query_id": query_id,
                    "query_passage": query,
                    "candidate_id": candidate_id,
                    "candidate_passage": candidate,
                    "relevant": int((query_id, candidate_id) in POSITIVE_PAIRS),
                })
    print(f"Wrote {len(QUERIES) * len(CANDIDATES)} synthetic labeled pairs to {OUTPUT}")


if __name__ == "__main__":
    main()
