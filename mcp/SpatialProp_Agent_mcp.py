"""
Model Context Protocol (MCP) for SpatialProp_Agent

SpatialProp is a graph neural network method for modeling spatial gene expression propagation in spatial transcriptomics data. This implementation provides tools for training spatial propagation models and predicting perturbation effects.

This MCP Server contains tools extracted from the following tutorial files:
1. api_demo
    - train_spatialprop_model: Train a SpatialProp GNN model from scratch using the tutorial configuration
    - plot_training_performance: Generate performance plots for the trained SpatialProp model
    - create_inflammatory_perturbation: Create perturbation input matrix for inflammatory gene amplification study
    - predict_spatial_propagation: Predict spatial propagation effects using trained SpatialProp model
    - visualize_propagation_results: Generate spatial visualization plots for SpatialProp results
"""

from fastmcp import FastMCP

# Import statements (alphabetical order)
from tools.api_demo import api_demo_mcp

# Server definition and mounting
mcp = FastMCP(name="SpatialProp_Agent")
mcp.mount(api_demo_mcp)

if __name__ == "__main__":
    mcp.run()