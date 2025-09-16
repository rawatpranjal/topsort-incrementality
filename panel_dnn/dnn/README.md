# Panel Data Deep Learning with Fixed Effects

Deep learning approaches for panel data with two-way fixed effects, demonstrating that neural networks can replicate traditional econometric fixed effects models.

## Files

### Core Implementations
- `panel_fixed_effects_deep_learning_vs_traditional_econometrics_comparison.py` - Comparison of DL vs traditional econometric methods
- `panel_nonlinear_beta_function_deep_neural_network_high_dimensional.py` - Nonlinear β(X) function recovery with high-dimensional DNNs
- `panel_two_way_fixed_effects_r_fixest_integration_simulation.py` - R's fixest package integration for benchmark comparisons
- `vendor_week_panel_deep_learning_fixed_effects_neural_network_analysis.py` - Production DL analysis on vendor-week panel data
- `vendor_week_panel_r_feols_fixed_effects_econometric_analysis.py` - R feols benchmark on vendor-week panel data

### Data
- `data/vendor_panel_full_history_clicks_only.parquet` - Vendor-week panel data (1M+ observations)

### Results
All outputs consolidated in `results/` folder including:
- Model outputs (`.txt`, `.csv`)
- Visualizations (`.png`)
- Trained models (`.pth`)
- Fixed effects extractions
- Training diagnostics

## Key Findings

### Production Results (Vendor-Week Panel)
- **R feols β**: 0.6422
- **Deep Learning β**: 0.6382
- **Error**: 1.49%

Successfully replicated econometric fixed effects with 152,389 vendor FE + 27 week FE.

### Simulation Results
- **Linear β recovery**: Errors reduced from ~0.4 to ~0.03
- **Nonlinear β(X) recovery**: Correlation 0.995 with complex functions
- **Fixed effects recovery**: Correlation >0.99 with true values

### Technical Insights
1. **No standardization** - Preserves coefficient interpretation
2. **Separate learning rates** - FE need 10x higher LR than coefficients
3. **Proper initialization** - Match FE distribution (std ~1.5-2.0)
4. **High patience** - 100+ epochs for FE convergence
5. **Light regularization** - Heavy weight decay hurts FE recovery