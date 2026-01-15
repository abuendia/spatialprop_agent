# [Paper2Agent](https://github.com/jmiao24/Paper2Agent): [SpatialProp](https://github.com/abuendia/spatial-prop) Demo

A demonstration of turning the [SpatialProp paper](https://www.biorxiv.org/content/10.64898/2025.11.30.691355v1.full) (Sun et al. 2025) into an interactive AI agent. This project transforms SpatialProp (Spatial Propagation of Single-cell Perturbations) into a conversational agent which can train graph deep learning models on spatial transcriptomics data to predict the effects of single-cell genetic perturbations.

## Folder Structure

```
SpatialProp_agent/
├── mcp/
│   ├── SpatialProp_Agent_mcp.py             # MCP server entry point
│   ├── requirements.txt                     # Python dependencies
│   └── tools/
│       └── api_demo.py                      # Training and evaluation tools
└── tmp/
    ├── inputs/                              # Input dataset directory
    └── outputs/                             # Generated results and figures directory
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/abuendia/SpatialProp_agent.git
cd SpatialProp_agent
```

### 2. Install Gemini CLI

Install the [Google Gemini CLI](https://github.com/google-gemini/gemini-cli):

```bash
brew install gemini-cli
```

### 3. Install FastMCP

```bash
pip install fastmcp
```

### 4. Install MCP Server

Install the SpatialProp server using fastmcp:

```bash
fastmcp install gemini-cli ./mcp/SpatialProp_Agent_mcp.py --with-requirements ./mcp/requirements.txt
```

### 5. Start the Agent

Start Gemini CLI in the repository folder:

```bash
gemini
```

You will now have access to the SpatialProp agent with all available tools.

## Example Query

```
Using the coronal aging dataset and model from the SpatialProp demo, perturb the 
genes TP53 and IFNGR1 by a factor of 20 in microglia. Visualize the predicted
propagation and output the plot.
```

## Available Agent Tools

The agent provides the following capabilities through natural language:

### Train and evaluate a SpatialProp model
- train_spatialprop_model: Train a SpatialProp GNN model from scratch using the tutorial configuration
- plot_training_performance: Generate performance plots for the trained SpatialProp model
- create_inflammatory_perturbation: Create perturbation input matrix for inflammatory gene amplification study
- predict_spatial_propagation: Predict spatial propagation effects using trained SpatialProp model
- visualize_propagation_results: Generate spatial visualization plots for SpatialProp results

## About SpatialProp

SpatialProp takes as input a spatially resolved single-cell transcriptomics dataset of intact tissue and a user-defined set of single-cell perturbations represented by their perturbed gene expression profiles. Then, using the core graph neural network module, SpatialProp predicts perturbed gene expression in a cell-by-cell manner and calibrates these predictions for model error to update gene expression profiles for every cell in the tissue. Finally, SpatialProp outputs a prediction of the perturbed gene expression profiles for all cells in the spatially resolved single-cell transcriptomics data, including for cells that did not receive a direct user-specified perturbation.

For more details, see the [SpatialProp repository](https://github.com/abuendia/spatial-prop).
