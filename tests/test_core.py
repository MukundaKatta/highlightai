"""Tests for Highlightai."""
from src.core import Highlightai
def test_init(): assert Highlightai().get_stats()["ops"] == 0
def test_op(): c = Highlightai(); c.generate(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = Highlightai(); [c.generate() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = Highlightai(); c.generate(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = Highlightai(); r = c.generate(); assert r["service"] == "highlightai"
