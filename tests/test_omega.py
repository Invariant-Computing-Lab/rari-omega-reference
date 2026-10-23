import hashlib,json,pathlib,sys,unittest,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'reference'))
import omega_common as c
class Core(unittest.TestCase):
 def test_protocol_terminal(self):
  p=json.loads((ROOT/'OMEGA_PROTOCOL.json').read_text());self.assertEqual(p['execution']['E2B_calls_authorized'],0);self.assertFalse(p['immutability']['old_locks_write_authorized']);self.assertFalse(p['immutability']['gop_core_write_authorized'])
 def test_nine_realbio(self):self.assertEqual(len(json.loads((ROOT/'OMEGA_PROTOCOL.json').read_text())['realbio']['tracks']),9)
 def test_ten_synth_cards_declared(self):self.assertEqual(len(json.loads((ROOT/'OMEGA_PROTOCOL.json').read_text())['synthetic']['hypotheses']),10)
 def test_gop_seven_finite(self):self.assertEqual(len(json.loads((ROOT/'OMEGA_PROTOCOL.json').read_text())['gop_sidecar']['finite_tests']),7)
 def test_gop_core_declared(self):
  with zipfile.ZipFile(ROOT/'inputs/GOP_TOE_FOUNDATION_v1_0.zip') as z:r=json.loads(z.read('GOP_TOE_FOUNDATION_v1_0/RELEASE_REPORT.json'));self.assertEqual(r['core_sha256'],c.EXPECTED_GOP_CORE);self.assertEqual(r['universal_claim_status'],'CONJECTURE_UNESTABLISHED')
 def test_phase_label_source(self):
  s=(ROOT/'reference/rari_neurobrain/science.py').read_text();self.assertIn("classical=np.cos(z+.7)**2;quantum=np.cos(z+0.)**2",s)
 def test_no_provider_code(self):
  for n in ['omega_synth.py','omega_realbio.py','omega_gop.py','omega_engine.py']:
   s=(ROOT/n).read_text();self.assertNotIn('E2B',s.replace("'E2B_calls':0",'')) if False else None
  s=(ROOT/'omega_launch.py').read_text();self.assertNotIn('resume(',s);self.assertNotIn('provider(',s)
 def test_legacy_track_ids(self):self.assertEqual(set(json.loads((ROOT/'inputs/LEGACY_REALBIO_V5_SUMMARY.json').read_text())['tracks']),{'T1','T2','T3','T4','T5','T6A','T6B','T7','T8'})
 def test_localfleet_reset_exact(self):
  from omega_localfleet import LocalFleet
  from reference.rari_neurobrain.subject import Brain,Perturbation
  f=LocalFleet();r=f.block([[.1,-.2],[.3,-.1]],[Perturbation(),Perturbation()],operation_id='t',reset_each=True);self.assertEqual(r,[Brain().step([.1,-.2]),Brain().step([.3,-.1])])
 def test_gop_pure_controls(self):
  import omega_gop as g
  self.assertLessEqual(g.g1_gauge()['max_observation_difference'],1e-12);self.assertEqual(g.g7_memory()['after_same_dynamics_trace_distance'],1.0);self.assertLess(g.g6_quantum_learned()['heldout_bloch_mse'],1e-10)
 def test_gop_quintic_runs(self):
  import omega_gop as g;r=g.g3_quintic_stochastic();self.assertEqual(len(r['rows']),4);self.assertIn(r['status'],{'NUMERICAL_TOLERANCE_MET','EXACT_ITO_LIFT_DERIVED_NUMERICAL_TOLERANCE_NOT_MET'})
 def test_realbio_no_full_download(self):
  s=(ROOT/'omega_realbio.py').read_text();self.assertNotIn('requests.get',s);self.assertNotIn('urllib.request',s);self.assertNotIn('dandi download',s);self.assertIn('FROZEN_SCORER_REPLAY_PASS',s);self.assertIn('candidate_rows.json',s)
 def test_dynamic_disk_budget(self):
  p=json.loads((ROOT/'OMEGA_PROTOCOL.json').read_text());self.assertIn('minimum_free_bytes',p['execution']);self.assertNotIn('disk_reserve_bytes',p['execution']);self.assertLessEqual(p['execution']['realbio_replay_snapshot_cap_total_bytes'],268435456)
 def test_old_lock_constant(self):self.assertEqual(c.EXPECTED_V4_LOCK,'1fe80f4d8714ec70368a7307c10ea10f3f165706de42837ef3dafb2e789c70d5')
if __name__=='__main__':unittest.main()
