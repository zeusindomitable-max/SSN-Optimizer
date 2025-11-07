# tests/conftest.py
import pytest
import torch
import torch.nn as nn

@pytest.fixture
def linear_model():
    """Model linear dummy untuk pengujian SSN."""
    model = nn.Linear(10, 1)
    torch.manual_seed(42)
    return model

@pytest.fixture
def data_batch():
    """Data dummy (fitur dan target)"""
    torch.manual_seed(42)
    x = torch.randn(64, 10)
    y = torch.randn(64, 1)
    return x, y
