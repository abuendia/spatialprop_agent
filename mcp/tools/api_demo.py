"""
SpatialProp API Demo Tools - Production-ready implementations extracted from tutorial notebook.

This module contains tools extracted from the SpatialProp API demo tutorial, preserving exact
tutorial structure and functionality for real-world spatial transcriptomics analysis.

Source: notebooks/api_demo/api_demo_execution_final.ipynb
Tutorial: Training and deploying a SpatialProp model from scratch
Dataset: aging_coronal.h5ad (MERFISH spatial transcriptomics)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import scanpy as sc
import torch
from fastmcp import FastMCP

# Create MCP server instance for this tutorial's tools
api_demo_mcp = FastMCP("SpatialProp API Demo Tools")

# Tutorial imports preserved exactly as in notebook
try:
    from spatial_gnn.api.perturbation_api import (
        create_perturbation_input_matrix, predict_perturbation_effects,
        train_perturbation_model)
    from spatial_gnn.utils.plot_utils import (
        plot_celltype_performance, plot_celltypes_in_section, plot_loss_curves,
        plot_propagation_results_for_gene_set)

    SPATIAL_GNN_AVAILABLE = True
except ImportError:
    SPATIAL_GNN_AVAILABLE = False
    print("Warning: spatial_gnn not available. Tools will return error messages.")


@api_demo_mcp.tool()
def train_spatialprop_model(
    adata_path: str, train_ids: List[str], test_ids: List[str], exp_name: str
) -> Dict[str, Any]:
    """
    Train a SpatialProp GNN model from scratch using the tutorial configuration.

    Reproduces the exact training workflow from the tutorial with the model configuration
    reported in the SpatialProp paper. Creates graphs using 2-hop neighbors and trains
    on masked center cell expression prediction.

    Args:
        adata_path: Path to the h5ad dataset file (e.g., "./data/aging_coronal.h5ad")
        train_ids: List of mouse IDs for training (e.g., ["14"])
        test_ids: List of mouse IDs for testing (e.g., ["57"])
        exp_name: Experiment name for organizing outputs (e.g., "api_demo")

    Returns:
        Dictionary containing:
        - trained_model_path: Path to the saved trained model
        - gene_names: List of gene names in the model
        - training_args: Complete training configuration used
        - model_available: Boolean indicating if model training completed successfully

    Note:
        - Requires CUDA-compatible GPU for optimal performance
        - Creates training/testing datasets in ./data/gnn_datasets/ directory
        - Uses exact paper configuration: 2-hop graphs, 100 cells per cell type, weighted-l1 loss
        - Training limited to 30 epochs for demo purposes
    """
    if not SPATIAL_GNN_AVAILABLE:
        return {
            "error": "spatial_gnn package not available",
            "trained_model_path": None,
            "gene_names": [],
            "training_args": {},
            "model_available": False,
        }

    # Basic input validation
    if not os.path.exists(adata_path):
        return {
            "error": f"Dataset file not found: {adata_path}",
            "trained_model_path": None,
            "gene_names": [],
            "training_args": {},
            "model_available": False,
        }

    # Exact training configuration from tutorial
    training_args = {
        "dataset": "aging_coronal",
        "file_path": adata_path,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "exp_name": exp_name,
        "k_hop": 2,
        "augment_hop": 2,
        "center_celltypes": "all",
        "node_feature": "expression",
        "inject_feature": "none",
        "learning_rate": 0.0001,
        "loss": "weightedl1",
        "epochs": 30,
        "normalize_total": True,
        "num_cells_per_ct_id": 100,
        "predict_celltype": False,
        "pool": "center",
        "do_eval": True,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    try:
        # Execute exact tutorial training call
        test_loader, gene_names, (model, model_config, trained_model_path) = (
            train_perturbation_model(
                **training_args,
            )
        )

        return {
            "trained_model_path": trained_model_path,
            "gene_names": gene_names,
            "training_args": training_args,
            "model_available": True,
            "device_used": training_args["device"],
        }

    except Exception as e:
        return {
            "error": f"Training failed: {str(e)}",
            "trained_model_path": None,
            "gene_names": [],
            "training_args": training_args,
            "model_available": False,
        }


@api_demo_mcp.tool()
def plot_training_performance(trained_model_path: str) -> Dict[str, Any]:
    """
    Generate performance plots for the trained SpatialProp model.

    Creates loss curves and cell type performance visualizations exactly as shown
    in the tutorial. Plots training convergence and held-out mouse performance.

    Args:
        trained_model_path: Path to the trained model file

    Returns:
        Dictionary containing:
        - model_dir: Directory containing the model and plots
        - loss_plot_generated: Boolean indicating if loss curves were plotted
        - performance_plot_generated: Boolean indicating if cell type performance was plotted
        - plots_available: List of generated plot files
    """
    if not SPATIAL_GNN_AVAILABLE:
        return {
            "error": "spatial_gnn package not available",
            "model_dir": None,
            "loss_plot_generated": False,
            "performance_plot_generated": False,
            "plots_available": [],
        }

    # Basic input validation
    if not os.path.exists(trained_model_path):
        return {
            "error": f"Trained model not found: {trained_model_path}",
            "model_dir": None,
            "loss_plot_generated": False,
            "performance_plot_generated": False,
            "plots_available": [],
        }

    model_dir = os.path.dirname(trained_model_path)
    plots_generated = []

    try:
        # Generate loss curves exactly as in tutorial
        plot_loss_curves(model_dir)
        loss_plot_generated = True
        plots_generated.append("loss_curves")

    except Exception as e:
        loss_plot_generated = False
        print(f"Loss curves plotting failed: {e}")

    try:
        # Generate cell type performance plots exactly as in tutorial
        plot_celltype_performance(model_dir)
        performance_plot_generated = True
        plots_generated.append("celltype_performance")

    except Exception as e:
        performance_plot_generated = False
        print(f"Cell type performance plotting failed: {e}")

    return {
        "model_dir": model_dir,
        "loss_plot_generated": loss_plot_generated,
        "performance_plot_generated": performance_plot_generated,
        "plots_available": plots_generated,
    }


@api_demo_mcp.tool()
def create_inflammatory_perturbation(adata_path: str, save_path: str) -> Dict[str, Any]:
    """
    Create perturbation input matrix for inflammatory gene amplification study.

    Implements the exact perturbation from the tutorial: amplifying pro-inflammatory
    genes by a factor of 10 in T cells and microglia. Saves perturbed expression
    matrix into anndata.obsm['perturbed_input'].

    Args:
        adata_path: Path to the original h5ad dataset file
        save_path: Path where the perturbed dataset will be saved

    Returns:
        Dictionary containing:
        - save_path: Path to saved perturbed dataset
        - perturbation_dict: The exact perturbation dictionary applied
        - inflammatory_genes: List of genes that were amplified
        - cells_affected: Summary of cells affected by perturbation
    """
    if not SPATIAL_GNN_AVAILABLE:
        return {
            "error": "spatial_gnn package not available",
            "save_path": None,
            "perturbation_dict": {},
            "inflammatory_genes": [],
            "cells_affected": {},
        }

    # Exact inflammatory gene set from tutorial (define first for consistent returns)
    inflammatory_genes = [
        "Ifng",
        "Il6",
        "Tnf",
        "Il1a",
        "Il1b",
        "Jun",
        "Apoe",
        "B2m",
        "C1qa",
        "Cd69",
        "Cd9",
        "Lyz2",
    ]

    # Exact perturbation dictionary from tutorial
    perturbation_dict = {
        "T cell": {gene: 10.0 for gene in inflammatory_genes},
        "Microglia": {gene: 10.0 for gene in inflammatory_genes},
    }

    # Basic input validation
    if not os.path.exists(adata_path):
        return {
            "error": f"Dataset file not found: {adata_path}",
            "save_path": None,
            "perturbation_dict": perturbation_dict,
            "inflammatory_genes": inflammatory_genes,
            "cells_affected": {},
        }

    try:
        # Load data exactly as in tutorial
        adata = sc.read_h5ad(adata_path)

        # Apply perturbations exactly as in tutorial
        actual_save_path = create_perturbation_input_matrix(
            adata, perturbation_dict, save_path=save_path
        )

        return {
            "save_path": actual_save_path,
            "perturbation_dict": perturbation_dict,
            "inflammatory_genes": inflammatory_genes,
            "cells_affected": {
                "T_cell_genes_amplified": len(inflammatory_genes),
                "Microglia_genes_amplified": len(inflammatory_genes),
                "amplification_factor": 10.0,
            },
        }

    except Exception as e:
        return {
            "error": f"Perturbation creation failed: {str(e)}",
            "save_path": None,
            "perturbation_dict": perturbation_dict,
            "inflammatory_genes": inflammatory_genes,
            "cells_affected": {},
        }


@api_demo_mcp.tool()
def predict_spatial_propagation(
    perturbed_adata_path: str,
    trained_model_path: str,
    exp_name: str,
    test_ids: List[str],
) -> Dict[str, Any]:
    """
    Predict spatial propagation effects using trained SpatialProp model.

    Executes the main SpatialProp prediction pipeline from the tutorial. Creates
    perturbed graphs from every cell and applies the SparseRenorm procedure to
    temper predictions.

    Args:
        perturbed_adata_path: Path to the h5ad file with perturbation input matrix
        trained_model_path: Path to the trained SpatialProp model
        exp_name: Experiment name for organizing outputs
        test_ids: List of mouse IDs to use for prediction

    Returns:
        Dictionary containing:
        - adata_result_available: Boolean indicating successful prediction
        - prediction_layers: List of available prediction layers in result
        - test_ids_used: List of mouse IDs processed
        - exp_name_used: Experiment name used

    Note:
        - Results are accessible through adata_result.layers['predicted_perturbed'] (raw predictions)
        - Tempered results available through adata_result.layers['predicted_tempered'] (full pipeline)
    """
    if not SPATIAL_GNN_AVAILABLE:
        return {
            "error": "spatial_gnn package not available",
            "adata_result_available": False,
            "prediction_layers": [],
            "test_ids_used": test_ids,
            "exp_name_used": exp_name,
        }

    # Basic input validation
    if not os.path.exists(perturbed_adata_path):
        return {
            "error": f"Perturbed dataset not found: {perturbed_adata_path}",
            "adata_result_available": False,
            "prediction_layers": [],
            "test_ids_used": test_ids,
            "exp_name_used": exp_name,
        }

    if not os.path.exists(trained_model_path):
        return {
            "error": f"Trained model not found: {trained_model_path}",
            "adata_result_available": False,
            "prediction_layers": [],
            "test_ids_used": test_ids,
            "exp_name_used": exp_name,
        }

    try:
        # Execute exact tutorial prediction call
        adata_result = predict_perturbation_effects(
            perturbed_adata_path, trained_model_path, exp_name, use_ids=test_ids
        )

        # Identify available prediction layers
        available_layers = []
        if hasattr(adata_result, "layers"):
            if "predicted_perturbed" in adata_result.layers:
                available_layers.append("predicted_perturbed")
            if "predicted_tempered" in adata_result.layers:
                available_layers.append("predicted_tempered")

        return {
            "adata_result_available": True,
            "prediction_layers": available_layers,
            "test_ids_used": test_ids,
            "exp_name_used": exp_name,
            "result_shape": adata_result.shape if adata_result is not None else None,
        }

    except Exception as e:
        return {
            "error": f"Spatial propagation prediction failed: {str(e)}",
            "adata_result_available": False,
            "prediction_layers": [],
            "test_ids_used": test_ids,
            "exp_name_used": exp_name,
        }


@api_demo_mcp.tool()
def visualize_propagation_results(
    adata_result_path: str, output_dir: str
) -> Dict[str, Any]:
    """
    Generate spatial visualization plots for SpatialProp results.

    Creates cell type spatial distribution and propagation results visualizations
    exactly as shown in the tutorial. Uses response genes related to interferon
    signaling and cell-cycle arrest.

    Args:
        adata_result_path: Path to the h5ad file containing prediction results
        output_dir: Directory where visualization plots will be saved

    Returns:
        Dictionary containing:
        - celltype_plot_generated: Boolean indicating if cell type plot was created
        - propagation_plot_generated: Boolean indicating if propagation plot was created
        - response_genes_used: List of genes visualized in propagation plot
        - output_files: List of generated plot files
    """
    if not SPATIAL_GNN_AVAILABLE:
        return {
            "error": "spatial_gnn package not available",
            "celltype_plot_generated": False,
            "propagation_plot_generated": False,
            "response_genes_used": [],
            "output_files": [],
        }

    # Exact response gene set from tutorial (define first for consistent returns)
    response_genes = [
        "Stat1",
        "Bst2",
        "Jak1",
        "Ifit1",
        "Cdkn1a",
        "Cdkn2a",
        "C4b",
        "H2-D1",
        "H2-K1",
    ]

    # Basic input validation
    if not os.path.exists(adata_result_path):
        return {
            "error": f"Results dataset not found: {adata_result_path}",
            "celltype_plot_generated": False,
            "propagation_plot_generated": False,
            "response_genes_used": response_genes,
            "output_files": [],
        }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    try:
        # Load results exactly as in tutorial
        adata_result = sc.read_h5ad(adata_result_path)

        # Generate cell type spatial distribution plot
        celltype_plot_generated = False
        try:
            plot_celltypes_in_section(
                adata_result, ct_key="celltype", s=0.5, figsize=(6, 6)
            )
            celltype_plot_generated = True
            output_files.append("celltype_spatial_distribution")
        except Exception as e:
            print(f"Cell type plotting failed: {e}")

        # Generate propagation results plot
        propagation_plot_generated = False
        try:
            # Extract expression layers exactly as in tutorial
            original_expn = adata_result.X
            perturbed_expn = adata_result.layers["predicted_perturbed"]
            tempered_expn = adata_result.layers["predicted_tempered"]

            # Generate propagation plot exactly as in tutorial
            save_path = os.path.join(output_dir, "aging_coronal_response_genes.png")
            plot_propagation_results_for_gene_set(
                adata_result,
                response_genes,
                orig_layer=original_expn,
                pert_layer=perturbed_expn,
                temp_layer=tempered_expn,
                save_path=save_path,
                fig_title="SpatialProp results on response gene set",
                point_size=0.1,
            )
            propagation_plot_generated = True
            output_files.append(save_path)

        except Exception as e:
            print(f"Propagation plotting failed: {e}")

        return {
            "celltype_plot_generated": celltype_plot_generated,
            "propagation_plot_generated": propagation_plot_generated,
            "response_genes_used": response_genes,
            "output_files": output_files,
        }

    except Exception as e:
        return {
            "error": f"Visualization failed: {str(e)}",
            "celltype_plot_generated": False,
            "propagation_plot_generated": False,
            "response_genes_used": response_genes,
            "output_files": [],
        }


if __name__ == "__main__":
    # Start the MCP server
    api_demo_mcp.run()
