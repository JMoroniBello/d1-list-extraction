"""Public synthetic score fixtures; no model download or private warehouse data."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from d1_list_extraction import Snapshot, decode
from d1_list_extraction.questions import KINDS


def fixture():
    text = "Slack, that Acme Labs"
    offsets = np.array([[-1,-1],[0,5],[5,6],[7,11],[12,16],[17,21]])
    scores = np.full((4,2,6), -8., dtype=np.float32)
    scores[0,:,0] = 8
    scores[1,0,1] = scores[1,1,1] = 9
    scores[2,0,1] = scores[2,1,1] = 8
    scores[2,0,4] = scores[2,1,5] = 7
    scores[3,0,1] = 8
    scores[3,0,4] = 7
    scores[3,1,1] = 7.5
    scores[3,1,5] = 8.2
    native = {"people":{"text":None,"span":None,"found":.01},
              "products":{"text":"Slack","span":[0,5],"found":.99},
              "organizations":{"text":"Slack","span":[0,5],"found":.99},
              "combined":{"text":text,"span":[0,len(text)],"found":.99}}
    return Snapshot(text,scores,offsets,native)


class DecoderTests(unittest.TestCase):
    def test_reject_merge_before_suppression_and_check_type(self):
        s=fixture();out=decode(s)
        self.assertEqual([a["text"] for a in out["answers"]["combined"]["items"]],["Slack","Acme Labs"])
        self.assertEqual([a["text"] for a in out["answers"]["organizations"]["items"]],["Acme Labs"])
        self.assertEqual(out["decoder"]["transformer_calls"],0)

    def test_baseline_is_permitted_to_return_merged_span(self):
        s=fixture();out=decode(s,policy="baseline")
        self.assertEqual(out["answers"]["combined"]["items"][0]["span"],[0,len(s.text)])

    def test_source_offsets_nonoverlap_and_immutable_scores(self):
        s=fixture();before=s.logits.tobytes();out=decode(s)
        for answer in out["answers"].values():
            items=answer["items"]
            for a in items:self.assertEqual(s.text[slice(*a["span"])],a["text"])
            for i,a in enumerate(items):
                for b in items[i+1:]:self.assertFalse(a["span"][0]<b["span"][1] and b["span"][0]<a["span"][1])
        self.assertEqual(before,s.logits.tobytes())
        with self.assertRaises(ValueError):s.logits[0,0,0]=1

    def test_native_absence_and_capacity(self):
        out=decode(fixture(),max_items=1)
        self.assertEqual(out["answers"]["people"]["items"],[])
        self.assertEqual(out["answers"]["people"]["stop"],"native_initial_not_found")
        self.assertTrue(out["answers"]["combined"]["at_capacity"])

    def test_preserves_multiword_name(self):
        result=decode(fixture(),kinds=["organizations"])
        self.assertEqual(result["answers"]["organizations"]["items"][0]["text"],"Acme Labs")

    def test_snapshot_roundtrip_and_cli_without_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"scores.npz";fixture().save(path)
            loaded=Snapshot.load(path)
            self.assertEqual(decode(loaded),decode(fixture()))
            run=subprocess.run([sys.executable,"-m","d1_list_extraction.cli","decode","--scores",str(path),"--kinds","products"],capture_output=True,text=True,check=True)
            self.assertEqual(json.loads(run.stdout)["answers"]["products"]["items"][0]["text"],"Slack")

    def test_repeated_mentions_remain_separate(self):
        text="Slack Slack";offsets=np.array([[-1,-1],[0,5],[6,11]])
        scores=np.full((4,2,3),-9.,np.float32)
        for i in [0,2,3]:scores[i,:,0]=8
        scores[1,:,1:]=8
        native={k:{"text":None,"span":None,"found":.01} for k in KINDS}
        native["products"]={"text":"Slack","span":[0,5],"found":.99}
        result=decode(Snapshot(text,scores,offsets,native),kinds=["products"])
        self.assertEqual([p["span"] for p in result["answers"]["products"]["items"]],[[0,5],[6,11]])

    def test_invalid_inputs_rejected(self):
        for kwargs in [{"cutoff":-1},{"max_items":0},{"max_items":True},{"policy":"unknown"},{"kinds":["unknown"]}]:
            with self.assertRaises(ValueError):decode(fixture(),**kwargs)
        s=fixture();scores=s.logits.copy();scores[0,0,0]=np.nan
        with self.assertRaises(ValueError):Snapshot(s.text,scores,s.offsets,s.native_answers)


if __name__=="__main__":unittest.main()
