"""Generate controlled synthetic validation/test examples with disjoint topics/templates."""
import csv
from pathlib import Path

ROOT = Path(__file__).parent
CASES = {
 "validation": [
  ("astro-1", "planetary", "definition-A", "Exact copying", "A lunar eclipse occurs when Earth passes between the Sun and Moon, casting a shadow across the lunar surface.", "A lunar eclipse occurs when Earth passes between the Sun and Moon, casting a shadow across the lunar surface.", 1),
  ("garden-1", "horticulture", "cause-A", "Paraphrase", "Native flowers provide pollinators food and safe shelter throughout the growing season.", "Local blossoms give pollinating insects dependable food and protected places during the season.", 1),
  ("secure-1", "cybersecurity", "sequence-A", "Partial copy", "Use a unique password for every account and enable a second verification step. Review account activity monthly.", "Use a unique password for every account and enable a second verification step. Update recovery details too.", 1),
  ("coffee-1", "beverages", "cause-B", "Common topic independent", "Coffee grind size changes extraction speed during brewing.", "Coffee beans develop aroma as roasting heat changes their chemistry.", 0),
  ("music-1", "music", "definition-B", "Unrelated", "A reed vibrates against a saxophone mouthpiece to produce sound.", "Clay hardens in a kiln after a potter shapes a bowl.", 0),
  ("astro-2", "planetary", "definition-A", "Common prompt", "Explain how a lunar eclipse occurs and identify the objects involved.", "Describe the stages of a lunar eclipse and name the bodies involved.", 0),
 ],
 "test": [
  ("tide-1", "oceanography", "process-X", "Exact copying", "Ocean tides change as the Moon's gravity pulls water into bulges on opposite sides of Earth.", "Ocean tides change as the Moon's gravity pulls water into bulges on opposite sides of Earth.", 1),
  ("forest-1", "forestry", "benefit-X", "Paraphrase", "Tree canopies cool cities by shading streets and releasing water vapor.", "Urban areas become cooler when trees shade roads and add moisture to the air.", 1),
  ("health-1", "public-health", "recommendation-X", "Partial copy", "Regular handwashing with soap reduces the spread of many infections. Dry hands with a clean towel.", "Regular handwashing with soap reduces the spread of many infections. Keep shared surfaces clean.", 1),
  ("bread-1", "food-science", "cause-Y", "Common topic independent", "Yeast produces gas that makes bread dough expand during fermentation.", "Longer baking creates a darker crust through browning reactions.", 0),
  ("painting-1", "visual-arts", "definition-Y", "Unrelated", "A watercolor pigment disperses in water and settles into paper fibers.", "A tide gauge records changes in sea level over time.", 0),
  ("tide-2", "oceanography", "process-X", "Common prompt", "Explain what causes ocean tides and describe the Moon's role.", "Describe how lunar gravity contributes to the daily rise and fall of ocean water.", 0),
 ]
}

def main():
 for split, cases in CASES.items():
  dest = ROOT / f"hybrid_{split}.csv"
  with dest.open("w", newline="", encoding="utf-8") as f:
   w = csv.DictWriter(f, fieldnames=["query_id","topic","template","scenario","query_passage","candidate_id","candidate_passage","relevant"])
   w.writeheader()
   for qid, topic, template, scenario, query, candidate, label in cases:
    # Fixed negatives are kept in the same topic/template set for a complete pair evaluation.
    w.writerow(dict(query_id=qid,topic=topic,template=template,scenario=scenario,query_passage=query,candidate_id=qid+"-candidate",candidate_passage=candidate,relevant=label))
 print("Generated synthetic hybrid validation and held-out test CSVs")
if __name__ == "__main__": main()
