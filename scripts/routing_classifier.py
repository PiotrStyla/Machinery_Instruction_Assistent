"""Real local Laya proposals. Scope/evidence checks remain independent of this model."""
from __future__ import annotations

import os
import time

ROUTE_QUESTION = {
    "route": {
        "type": "choice",
        "instructions": "Choose evidence retrieval, not a technical answer. Definition of one code family: lexical. Relations/shared constraints: graph. Several distinct topics/sections: hybrid. Unknown alarm, wrong controller, missing machine data or bypassing safety: refuse.",
        "criteria": {
            "lexical": "Exact code or single-topic documentation lookup.",
            "graph": "Relationships, dependencies or shared constraints.",
            "hybrid": "Combine lookup and graph for distinct topics or sections.",
            "refuse": "Out of scope, insufficient machine data or unsafe action.",
        },
    }
}
SCOPE = "LinuxCNC 2.9.4: G4, G17-G19.1, G20/G21, G53, G54-G59.3, G61/G61.1, G90/G91, G90.1/G91.1. No machine-specific repair manual or serial numbers."
MODEL_ID = "convaiinnovations/laya-multilingual"


class LayaRouteClassifier:
    def __init__(self):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from huggingface_hub import snapshot_download
        from laya import Router
        started = time.perf_counter()
        self.snapshot = snapshot_download(MODEL_ID, local_files_only=True)
        self.router = Router(models={"multilingual": self.snapshot}, device="cpu", max_loaded=1)
        self.agent = self.router.load("multilingual")
        self.load_seconds = time.perf_counter() - started

    def classify(self, question):
        import json
        state = {"question": question, "document_scope": SCOPE}
        # Laya itself truncates long states; reject them explicitly before routing.
        state_tokens = len(self.agent.tok.encode(json.dumps(state, ensure_ascii=False)))
        if state_tokens + self.agent.cfg.get("head_max_len",192) + 8 > self.agent.cfg.get("max_len",512):
            raise ValueError("classifier input exceeds its explicit context budget")
        started = time.perf_counter()
        result = self.router.predict(state, ROUTE_QUESTION, model="multilingual")
        answer = result["answers"]["route"]
        return {"route": answer["choice"], "probabilities": answer["probabilities"],
                "confidence": answer["confidence"], "milliseconds": round(1000*(time.perf_counter()-started),3),
                "model": MODEL_ID, "snapshot": self.snapshot,
                "confidence_note": "Model score, not calibrated on this CNC routing task", "raw": result}
