"""
Module for global variable declarations
"""
import torch 

GPU = 1  if torch.cuda.is_available() else 0
