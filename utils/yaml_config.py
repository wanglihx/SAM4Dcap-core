import yaml
import re

# Regular expression to match ${...} references
VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

def resolve_ref(cfg, ref):
    """
    Resolve a dot-path reference inside the config dictionary.

    Args:
        cfg (dict): The root configuration dictionary.
        ref (str): A reference string like 'paths.ckpt_root'.

    Returns:
        Any: The resolved value from the config dictionary.

    Raises:
        KeyError: If the reference does not exist in the config.
    """
    parts = ref.split(".")
    val = cfg
    for p in parts:
        if p not in val:
            raise KeyError(f"Config reference '{ref}' not found (stopped at '{p}')")
        val = val[p]
    return val

def expand_config(obj, root_cfg):
    """
    Recursively expand all ${...} references inside the config.

    Args:
        obj (Any): A node in the config, could be dict/list/string/etc.
        root_cfg (dict): The full config for reference lookup.

    Returns:
        Any: The same structure as input but with expanded references.
    """
    if isinstance(obj, dict):
        return {k: expand_config(v, root_cfg) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [expand_config(v, root_cfg) for v in obj]
    elif isinstance(obj, str):
        # Replace all matches of ${...} inside the string
        def _repl(m):
            ref = m.group(1)  # e.g., "paths.ckpt_root"
            val = resolve_ref(root_cfg, ref)
            return str(val)
        return VAR_PATTERN.sub(_repl, obj)
    else:
        return obj

# # ---------- Usage ----------
# with open("config.yaml", "r") as f:
#     raw_cfg = yaml.safe_load(f)

# # Perform variable expansion
# cfg = expand_config(raw_cfg, raw_cfg)

# # Example:
# # cfg["sam3"]["ckpt_path"] =>
# #     "/root/autodl-tmp/checkpoints/sam3/sam3.pt"
# # cfg["completion"]["model_path_mask"] =>
# #     "/root/autodl-tmp/checkpoints/diffusion-vas-amodal-segmentation"
