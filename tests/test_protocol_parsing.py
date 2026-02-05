import pytest
import pandas as pd
import os
from src.data.protocols import parse_protocol, get_file_list

def test_parse_protocol_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_protocol("non_existent_file.txt")

def test_parse_protocol_structure(tmp_path):
    # Create dummy protocol
    p = tmp_path / "dummy_protocol.txt"
    p.write_text("LA_0001 LA_E_10001 - - bonafide\nLA_0001 LA_E_10002 - - spoof")
    
    df = parse_protocol(str(p))
    assert len(df) == 2
    assert "filename" in df.columns
    assert "key" in df.columns
    assert df.iloc[0]['key'] == 'bonafide'
    assert df.iloc[1]['key'] == 'spoof'

def test_get_file_list(tmp_path):
    p = tmp_path / "dummy_protocol.txt"
    p.write_text("SPEAKER FILE1 SYSTEM - bonafide")
    df = parse_protocol(str(p))
    
    paths, labels, attacks = get_file_list(df, str(tmp_path), track="LA", subset="train")
    
    assert len(paths) == 1
    assert labels[0] == 1 # Bonafide
    # Check path construction
    expected_suffix = os.path.join("LA", "ASVspoof2019_LA_train", "flac", "FILE1.flac")
    assert paths[0].endswith(expected_suffix)
