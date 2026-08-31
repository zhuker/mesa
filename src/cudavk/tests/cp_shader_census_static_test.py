#!/usr/bin/env python3
"""No-device source invariants for the diagnostic full-key census."""
import re
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
PIPE=(ROOT/"src/cudavk/cpvk_pipeline.c").read_text()
RENDER=(ROOT/"src/cudavk/cp_renderer.c").read_text()
DEVICE=(ROOT/"src/cudavk/cpvk_device_memory.c").read_text()
HEADER=(ROOT/"src/cudavk/nir_to_ptx/cp_nir_to_llvm.h").read_text()
DEBUG=(ROOT/"src/cudavk/cp_debug.c").read_text()

class StaticTest(unittest.TestCase):
    def test_one_generated_value_flag(self):
        self.assertEqual(DEBUG.count('"CUDAVK_SHADER_CENSUS"'),1)
        self.assertIn('CP_FLAG_BOOL_VALUE, F(shader_census)',DEBUG)

    def test_hash_is_diagnostic_tail(self):
        tail=HEADER[HEADER.index("struct cp_shader_binary {"):HEADER.index("static inline struct cp_shader_exec *")]
        self.assertIsNotNone(re.search(r"out_location\[CP_MAX_IO_SLOTS\];\s*/\* Diagnostic.*stage_hash\[32\];\s*bool stage_hash_valid;\s*};", tail, re.S))

    def test_only_winning_canonical_binary_is_tagged(self):
        self.assertEqual(PIPE.count("memcpy(bin->stage_hash, hash, BLAKE3_OUT_LEN)"),1)
        lock=PIPE.index("simple_mtx_lock(&dev->shader_cache_lock);",PIPE.index("bool loaded"))
        loser=PIPE.index("cp_shader_binary_destroy(bin);",lock)
        tag=PIPE.index("memcpy(bin->stage_hash, hash, BLAKE3_OUT_LEN)",loser)
        publish=PIPE.index("memcpy(dev->shader_cache[dev->num_shaders].hash",tag)
        increment=PIPE.index("dev->num_shaders++;",publish)
        self.assertLess(lock,loser); self.assertLess(loser,tag); self.assertLess(tag,publish); self.assertLess(publish,increment)
        guard=PIPE.rfind("if (cp_debug->shader_census)",loser,tag)
        self.assertGreater(guard,loser)

    def test_both_cache_hits_verify_full_key_under_lock(self):
        self.assertEqual(PIPE.count("cpvk_shader_census_check_key("),3) # definition + two hits
        self.assertIn("memcmp(bin->stage_hash, hash, BLAKE3_OUT_LEN)",PIPE)

    def test_record_is_strictly_after_successful_launch(self):
        hook=RENDER.index("cp_shader_census_record_vs(cp")
        success=RENDER.rfind("if (vs_err == CUDA_SUCCESS && cp_debug->shader_census)",0,hook)
        launch=RENDER.rfind("CUresult vs_err = cp_launch",0,success)
        self.assertLess(launch,success); self.assertLess(success,hook)

    def test_launch_record_path_has_no_forbidden_work(self):
        start=RENDER.index("cp_shader_census_record_vs(")
        end=RENDER.index("static void\ncp_shader_census_report",start)
        body=RENDER[start:end]
        for token in ("calloc(","malloc(","realloc(","fprintf(","snprintf(","cuMem", "cuCtx", "mtx_"):
            self.assertNotIn(token,body)

    def test_key_mismatch_is_per_context(self):
        debug=(ROOT/"src/cudavk/cp_debug.c").read_text()
        header=(ROOT/"src/cudavk/cp_renderer.h").read_text()
        self.assertNotIn("shader_census_key_error_count", debug)
        self.assertIn("uint64_t key_mismatch;", header)
        self.assertIn("dev->renderer.shader_census.key_mismatch++", PIPE)

    def test_cleanup_sync_controls_report_validity(self):
        self.assertIn("cp_shader_census_report(cp, context_sync == CUDA_SUCCESS)", RENDER)
        self.assertRegex(RENDER, r"population_ok = context_sync_ok")

    def test_ids_are_flag_gated(self):
        self.assertRegex(RENDER, r"cp_debug->shader_census\s*\?\s*\+\+cp->shader_census.batch_ordinal\s*:\s*0")
        self.assertIn("if (cp_debug->shader_census && submit->command_buffer_count)",DEVICE)

if __name__=="__main__": unittest.main()
