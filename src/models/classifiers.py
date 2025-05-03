def get_model_dict(selection="paper", grid_search=False):
    """
    Return dictionary of classifiers based on selection.

    Parameters:
    -----------
    selection : str
        "fast" - Fastest models for quick testing and development (4 models)
        "paper" - Balanced selection for publication (8 models)
        "full" - All available models (20 models)
        "sktime" - Only models from the sktime library
        "sklearn" - Only models from the scikit-learn library
        "grid" - Models configured for hyperparameter tuning
    grid_search : bool, default=False
        If True, return models with parameter grids for hyperparameter tuning

    Returns:
    --------
    Dictionary of initialized model instances or (model, param_grid) tuples for grid search
    """
    import os
    from importlib import import_module

    import yaml

    # Define path to YAML file - adjust as needed for your project structure
    yaml_path = os.path.join(os.path.dirname(__file__), "classifiers.yml")

    # Load model specifications from YAML file
    with open(yaml_path, "r") as file:
        model_specs = yaml.safe_load(file)

    # Validate selection
    valid_selections = ["fast", "paper", "full", "sktime", "sklearn", "grid"]
    if selection not in valid_selections:
        raise ValueError(
            f"Unknown model selection: {selection}. "
            f"Choose from {', '.join(valid_selections)}."
        )

    # Initialize empty dictionary for models
    models = {}

    # Process each model based on selection
    for model_name, model_info in model_specs.items():
        # Skip models not in the selected set, except for grid search which uses all models
        if selection != "grid" and selection not in model_info.get("in_sets", []):
            continue

        # For fast selection, only include models with fast configuration
        if selection == "fast" and not model_info.get("fast_params"):
            continue

        # Import the model class
        module_path = model_info["import_path"]
        class_name = model_info["class_name"]

        # Use dynamic import to get the model class
        try:
            module = import_module(module_path)
            model_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not import {class_name} from {module_path}: {e}")
            continue

        # Determine which parameters to use based on selection
        if selection == "fast" and "fast_params" in model_info:
            params = model_info["fast_params"]
            # Use fast-specific model name if specified
            display_name = model_info.get("fast_name", f"{model_name}-fast")
        elif selection == "paper" and "paper_params" in model_info:
            params = model_info["paper_params"]
            display_name = model_name
        else:
            params = model_info["params"]
            display_name = model_name

        # If grid search is enabled, return model with param grid
        if grid_search or selection == "grid":
            if "param_grid" in model_info:
                # Initialize model with default parameters
                model = model_class(**params)
                # Return model with parameter grid
                models[display_name] = (model, model_info["param_grid"])
            else:
                # Skip models without parameter grid if grid search is requested
                continue
        else:
            # Initialize and return model instance
            models[display_name] = model_class(**params)

    return models
