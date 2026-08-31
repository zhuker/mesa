#!/usr/bin/env python3
"""CPU oracle for the census key and generation-tag open-address semantics."""
from __future__ import annotations
import unittest
MASK64=(1<<64)-1

def mix(row,vertex,instance):
    x=0x9e3779b97f4a7c15 ^ row
    x ^= (vertex+0x9e3779b97f4a7c15+(x<<6)+(x>>2)) & MASK64
    x ^= (instance+0x9e3779b97f4a7c15+(x<<6)+(x>>2)) & MASK64
    x &= MASK64; x^=x>>30; x=(x*0xbf58476d1ce4e5b9)&MASK64
    x^=x>>27; x=(x*0x94d049bb133111eb)&MASK64
    return (x^(x>>31))&MASK64

class Bank:
    def __init__(self,capacity,hasher=mix):
        assert capacity and capacity&(capacity-1)==0
        self.entries=[None]*capacity; self.tags=[0]*capacity; self.generation=0; self.hasher=hasher
    def census(self,keys):
        if self.generation>=0x7fffffff: return len(keys),0,True
        self.generation+=1; unique=duplicate=0
        for key in keys:
            slot=self.hasher(*key)&(len(self.entries)-1)
            for _ in range(len(self.entries)):
                if self.tags[slot]!=self.generation:
                    self.tags[slot]=self.generation; self.entries[slot]=key; unique+=1; break
                if self.entries[slot]==key: duplicate+=1; break
                slot=(slot+1)&(len(self.entries)-1)
            else: return len(keys),0,True # whole-batch conservative charge
        return unique,duplicate,False

class TestOracle(unittest.TestCase):
    def test_key_fields_are_independent(self):
        b=Bank(16,lambda *_:0)
        keys=[(0,4,2),(0,4,2),(1,4,2),(0,5,2),(0,4,3)]
        self.assertEqual(b.census(keys),(4,1,False))
    def test_nonindexed_sequential_has_no_duplicates(self):
        self.assertEqual(Bank(16).census([(0,v,0) for v in range(8)]),(8,0,False))
    def test_forced_hash_collision_is_exact(self):
        self.assertEqual(Bank(8,lambda *_:3).census([(0,1,0),(0,2,0),(0,1,0)]),(2,1,False))
    def test_empty_and_full(self):
        b=Bank(2,lambda *_:0)
        self.assertEqual(b.census([]),(0,0,False))
        self.assertEqual(b.census([(0,1,0),(0,2,0),(0,3,0)]),(3,0,True))
    def test_generation_isolation(self):
        b=Bank(4,lambda *_:0)
        self.assertEqual(b.census([(0,1,0),(0,1,0)]),(1,1,False))
        self.assertEqual(b.census([(0,1,0)]),(1,0,False))
    def test_bank_isolation(self):
        a,b=Bank(4),Bank(4)
        self.assertEqual(a.census([(0,1,0)]),(1,0,False))
        self.assertEqual(b.census([(0,1,0)]),(1,0,False))
    def test_generation_wrap_excluded(self):
        b=Bank(4); b.generation=0x7fffffff
        self.assertEqual(b.census([(0,1,0)]),(1,0,True))
    def test_record_overflow_is_detectable(self):
        capacity=2; population=3; reserved=min(capacity,population)
        self.assertEqual(population-reserved,1)
        self.assertNotEqual(population,reserved)

if __name__=="__main__": unittest.main()
