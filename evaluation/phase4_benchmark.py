"""Calibrate hybrid signals on validation, freeze, then evaluate held-out synthetic pairs."""
import argparse, csv, hashlib, json, sys, time, resource
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "backend"))
from hybrid_evaluation import validate_disjoint_splits, calibrate, evaluate, freeze_config
from app.services.embedding import encode, get_model
from app.services.reranking import rerank, get_reranker
from app.services.hybrid_similarity import overlap_signals

def read_rows(path):
 with Path(path).open(newline="", encoding="utf-8") as f: rows=list(csv.DictReader(f))
 for r in rows: r["relevant"]=int(r["relevant"])
 return rows

def prepare(rows):
 texts=list(dict.fromkeys([r["query_passage"] for r in rows]+[r["candidate_passage"] for r in rows]))
 vectors=encode(texts); ix={t:i for i,t in enumerate(texts)}
 for row in rows:
  a,b=row["query_passage"],row["candidate_passage"]
  row["semantic"]=float(np.dot(vectors[ix[a]],vectors[ix[b]]))
  row.update({"lexical":overlap_signals(a,b)["lexical_similarity"],"phrase":overlap_signals(a,b)["phrase_overlap"]})
 return rows

def metrics_report(rows, baseline, cross, hybrid):
 from metrics import confusion_counts, precision_recall_f1
 result={}
 for key,preds in (("faiss_baseline",baseline),("faiss_cross_encoder",cross),("hybrid",hybrid)):
  counts=confusion_counts([r["relevant"] for r in rows],preds); p,r,f=precision_recall_f1(counts); tn,fp=counts["tn"],counts["fp"]
  result[key]={**counts,"precision":p,"recall":r,"f1":f,"specificity":tn/(tn+fp) if tn+fp else 0,"false_positive_rate":fp/(fp+tn) if tn+fp else 0}
 return result

def run(validation_path, test_path, results_dir):
 val,test=read_rows(validation_path),read_rows(test_path); validate_disjoint_splits(val,test)
 results_dir=Path(results_dir); results_dir.mkdir(parents=True,exist_ok=True)
 memory_before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 started=time.perf_counter(); get_model(); emb_load=time.perf_counter()-started
 started=time.perf_counter(); get_reranker(); ce_load=time.perf_counter()-started
 val=prepare(val); test=prepare(test)
 val_pairs=[(r["query_passage"],r["candidate_passage"]) for r in val]
 test_pairs=[(r["query_passage"],r["candidate_passage"]) for r in test]
 t=time.perf_counter(); val_ce=rerank(val_pairs); ce_val_time=time.perf_counter()-t
 t=time.perf_counter(); test_ce=rerank(test_pairs); ce_test_time=time.perf_counter()-t
 # Baseline threshold mirrors current API's configurable cosine cutoff.
 for rows, scores in ((val,val_ce),(test,test_ce)):
  for row, score in zip(rows,scores): row["cross_encoder"]=float(score)
 config=calibrate(val)
 frozen=freeze_config(results_dir / "hybrid_calibration.json",config,Path(validation_path))
 vresult=evaluate(val,frozen); tres=evaluate(test,frozen)
 baseline=[int(r["semantic"] >= .45) for r in test]
 # Reranking only changes ordering; with one retrieved candidate per labeled pair,
 # the existing baseline has identical membership decisions.
 cross=list(baseline)
 hybrid=[int(x >= frozen["threshold"]) for x in tres.pop("scores")]
 memory_after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 rss_scale=1024 * 1024 if sys.platform == "darwin" else 1024
 results={"validation":vresult,"test":metrics_report(test,baseline,cross,hybrid),"per_scenario":{},"timing":{"embedding_load_seconds":emb_load,"cross_encoder_load_seconds":ce_load,"validation_cross_encoder_inference_seconds":ce_val_time,"test_cross_encoder_inference_seconds":ce_test_time,"benchmark_peak_rss_delta_mb":max(0,memory_after-memory_before)/rss_scale}}
 for scenario in sorted({r["scenario"] for r in test}):
  ix=[i for i,r in enumerate(test) if r["scenario"]==scenario]
  results["per_scenario"][scenario]=metrics_report([test[i] for i in ix],[baseline[i] for i in ix],[cross[i] for i in ix],[hybrid[i] for i in ix])
 out=Path(results_dir)
 fields=list(dict.fromkeys(k for row in test for k in row))+["faiss_prediction","cross_encoder_prediction","hybrid_prediction"]
 with (out/"hybrid_test_pairs.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row,b,c,h in zip(test,baseline,cross,hybrid):w.writerow({**row,"faiss_prediction":b,"cross_encoder_prediction":c,"hybrid_prediction":h})
 (out/"phase4_report.md").write_text(report_text(val,test,results,frozen,validation_path,test_path),encoding="utf-8")
 (out/"phase4_metrics.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")
 return results

def report_text(val,test,results,config,vpath,tpath):
 testm=results["test"]
 lines=["# SemanticGuard Phase 4 — Synthetic False-Positive Evaluation","","**Controlled synthetic examples only; these results are not real-world plagiarism accuracy. Scores are not plagiarism probabilities.**","",f"Validation SHA-256: `{hashlib.sha256(Path(vpath).read_bytes()).hexdigest()}`",f"Held-out test SHA-256: `{hashlib.sha256(Path(tpath).read_bytes()).hexdigest()}`","","Validation and test contain disjoint topic and template identifiers. Weights and threshold were calibrated on validation, written to `hybrid_calibration.json`, then frozen before test scoring.","",f"Frozen weights: `{config['weights']}`; threshold: `{config['threshold']:.2f}`.","","## Held-out test metrics","","| Pipeline | Precision | Recall | F1 | Specificity | False-positive rate | TN | FP | FN | TP |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
 for name,m in testm.items():lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['specificity']:.3f} | {m['false_positive_rate']:.3f} | {m['tn']} | {m['fp']} | {m['fn']} | {m['tp']} |")
 lines += ["","## Scenario breakdown","","| Scenario | Pipeline | TN | FP | FN | TP | FPR | F1 |","|---|---|---:|---:|---:|---:|---:|---:|"]
 for scenario, pipelines in results["per_scenario"].items():
  for name,m in pipelines.items(): lines.append(f"| {scenario} | {name} | {m['tn']} | {m['fp']} | {m['fn']} | {m['tp']} | {m['false_positive_rate']:.3f} | {m['f1']:.3f} |")
 lines += ["","## Timing and memory","",f"Embedding model load: {results['timing']['embedding_load_seconds']:.4f}s",f"Cross-Encoder model load: {results['timing']['cross_encoder_load_seconds']:.4f}s",f"Cross-Encoder validation inference: {results['timing']['validation_cross_encoder_inference_seconds']:.4f}s",f"Cross-Encoder test inference: {results['timing']['test_cross_encoder_inference_seconds']:.4f}s",f"Process peak RSS increase over the full benchmark run: {results['timing']['benchmark_peak_rss_delta_mb']:.2f} MiB (includes model loading and inference).","","The pair dataset tests threshold behavior, not FAISS retrieval recall: each row is a labeled candidate pair. Phase 3 separately reports retrieval/ranking metrics. The baseline is the current cosine cutoff (0.45); Cross-Encoder reranking has identical candidate membership and thus identical binary decisions in this one-candidate-per-pair design. Hybrid predictions use validation-frozen weights/threshold.","","Results are authored examples with one observation per scenario. Per-scenario false-positive rates are therefore highly unstable and descriptive only. A negative result does not support general claims about academic integrity or student misconduct.",""]
 return "\n".join(lines)

def main():
 p=argparse.ArgumentParser();p.add_argument("--validation",type=Path,default=HERE/"datasets/hybrid_validation.csv");p.add_argument("--test",type=Path,default=HERE/"datasets/hybrid_test.csv");p.add_argument("--results-dir",type=Path,default=HERE/"results");a=p.parse_args(); print(json.dumps(run(a.validation,a.test,a.results_dir),indent=2))
if __name__=="__main__":main()
