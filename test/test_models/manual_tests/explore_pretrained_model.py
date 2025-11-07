import os
from collections import Counter
import torch
import contextlib
import sys
from datetime import datetime

"""
In an attemt to understand more about the SOTA Unets, I am trying to learn more about the setup from the pretrained models I have found.
"""

# model_path = "/home/users/bradlesc/projects/ClimSim/models/models/saved_model/v2rh_unet_nonaggressive_cliprh_huber/model.pt" #26, 14
# model_path = "/home/users/bradlesc/projects/ClimSim/models/models/saved_model/v2rh_unet_nonaggressive_cliprh_huber_rop2/model.pt" #26, 14
# model_path = "/home/users/bradlesc/projects/ClimSim/models/models/saved_model/v2rh_unet_nonaggressive_cliprh_mae/model.pt" #26, 14
# model_path = "/home/users/bradlesc/projects/ClimSim/models/models/saved_model/v4plus_unet_nonaggressive_cliprh_huber/model.pt" #57, 14
# model_path = "/home/users/bradlesc/projects/ClimSim/models/models/saved_model/v4plus_unet_nonaggressive_cliprh_huber_rop2_r3/model.pt" #57, 14
# model_path = "/home/users/bradlesc/projects/ClimSim/models/models/saved_model/v4plus_unet_nonaggressive_cliprh_mae/model.pt"
# model_path = "/home/users/bradlesc/projects/ClimSim/models/models/saved_model/v5_unet_nonaggressive_cliprh_huber/model.pt" # 55, 13
# model_path = "/home/users/bradlesc/projects/ClimSim/models/models/saved_model/v5_unet_nonaggressive_cliprh_huber_rop2_r2/model.pt" # 55, 13
# model_path = "/home/users/bradlesc/projects/ClimSim/models/models/saved_model/v5_unet_nonaggressive_cliprh_mae/model.pt" # 55, 13

def _print_tensor_info(name, t):
    print(f"  {name}: shape={tuple(t.size())} dtype={t.dtype} numel={t.numel()}")

def summarize_state_dict(sd):
    total = 0
    tensors = [(k, v) for k, v in sd.items() if isinstance(v, torch.Tensor)]
    non_tensors = [(k, v) for k, v in sd.items() if not isinstance(v, torch.Tensor)]
    print(f"State-dict contains {len(tensors)} tensors and {len(non_tensors)} non-tensor entries")
    # show a few largest tensors first
    for k, v in sorted(tensors, key=lambda kv: kv[1].numel(), reverse=True)[:40]:
        _print_tensor_info(k, v)
        total += v.numel()
    if len(tensors) > 40:
        print(f"  ... ({len(tensors)-40} more tensors omitted)")
    print(f"Total params (numel) in state-dict: {total}")

def summarize_module(mod):
    print("Module architecture (str):")
    print(mod)
    params = list(mod.named_parameters())
    total = sum(p.numel() for _, p in params)
    trainable = sum(p.numel() for _, p in params if p.requires_grad)
    print(f"Parameters: {len(params)} tensors, total numel={total}, trainable numel={trainable}")
    # top parameter tensors
    for name, p in sorted(params, key=lambda kv: kv[1].numel(), reverse=True)[:40]:
        print(f"  {name}: shape={tuple(p.size())} numel={p.numel()} requires_grad={p.requires_grad}")
    if len(params) > 40:
        print(f"  ... ({len(params)-40} more parameter tensors omitted)")
    # module type summary
    types = Counter(type(m).__name__ for m in mod.modules())
    print("Module type counts:")
    for t, c in types.most_common():
        print(f"  {t}: {c}")

def _run_inspection(log_f=None, model_path="model.pt"):
    # This function contains the previous top-level inspection logic. When
    # called with a file-like `log_f` open and stdout redirected, all prints
    # will be written to that file.
    print("Model file:", model_path)
    try:
        print("File size (bytes):", os.path.getsize(model_path))
    except Exception:
        pass

    obj = torch.load(model_path, map_location="cpu", weights_only=False)
    print("Loaded object type:", type(obj))

    if isinstance(obj, dict):
        print("Top-level dict keys:", list(obj.keys()))
        # try to locate a state_dict inside the dict
        state_dict = None
        candidate_keys = ("state_dict", "model_state_dict", "model", "state")
        for k in candidate_keys:
            if k in obj and isinstance(obj[k], dict):
                state_dict = obj[k]
                print(f"Using '{k}' as state-dict candidate")
                break
        # if the dict itself is map of tensors -> treat it as a state_dict
        if state_dict is None and all(isinstance(v, torch.Tensor) for v in obj.values()):
            state_dict = obj
            print("Top-level dict looks like a state-dict; using it directly")
        if state_dict is not None:
            summarize_state_dict(state_dict)
        else:
            # print brief info about other entries
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k}: tensor shape={tuple(v.size())} dtype={v.dtype}")
                else:
                    print(f"{k}: {type(v)}")
    elif isinstance(obj, torch.nn.Module):
        summarize_module(obj)
    else:
        # fallback: maybe a custom container with state_dict method
        if hasattr(obj, "state_dict") and callable(obj.state_dict):
            try:
                sd = obj.state_dict()
                print("Object has state_dict(); summarizing it:")
                summarize_state_dict(sd)
            except Exception as e:
                print("Calling state_dict() failed:", e)
        else:
            print("Unrecognized object type. Try inspecting its attributes:")
            for a in dir(obj):
                if not a.startswith("_"):
                    print(" ", a)

def _print_inferred(label, info):
    print(f"{label}: {info}")

def _infer_from_state_dict(sd):
    tensors = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    if not tensors:
        _print_inferred("State-dict inference", "no tensor entries found")
        return

    # find candidate weight tensors (keep original ordering if possible)
    weight_items = [(k, v) for k, v in tensors.items() if k.endswith("weight") or ".weight" in k]
    if not weight_items:
        weight_items = list(tensors.items())

    first_key, first_tensor = weight_items[0]
    last_key, last_tensor = weight_items[-1]

    def interpret(t):
        if t.dim() == 4:
            # Conv2d: (out, in, kH, kW)
            return {"type": "conv2d", "out_channels": t.size(0), "in_channels": t.size(1), "kernel": tuple(t.size()[2:])}
        if t.dim() == 5:
            # Conv3d: (out, in, kD, kH, kW)
            return {"type": "conv3d", "out_channels": t.size(0), "in_channels": t.size(1), "kernel": tuple(t.size()[2:])}
        if t.dim() == 3:
            # Conv1d: (out, in, kW)
            return {"type": "conv1d", "out_channels": t.size(0), "in_channels": t.size(1), "kernel": (t.size(2),)}
        if t.dim() == 2:
            # Linear: (out_features, in_features)
            return {"type": "linear", "out_features": t.size(0), "in_features": t.size(1)}
        return {"type": f"tensor_{t.dim()}d", "shape": tuple(t.size())}

    first_info = interpret(first_tensor)
    last_info = interpret(last_tensor)

    _print_inferred("First weight tensor", f"{first_key}: {first_info}")
    _print_inferred("Last weight tensor", f"{last_key}: {last_info}")

    # Summarize likely expected input / output channels/features
    if first_info.get("in_channels") is not None:
        _print_inferred("Likely expected input channels", first_info["in_channels"])
    if last_info.get("out_channels") is not None:
        _print_inferred("Likely model output channels", last_info["out_channels"])
    if first_info.get("in_features") is not None:
        _print_inferred("Likely expected input features (linear)", first_info["in_features"])
    if last_info.get("out_features") is not None:
        _print_inferred("Likely model output features (linear)", last_info["out_features"])


def _infer_from_module(mod):
    # quick attribute probes
    attrs = {}
    for name in ("in_channels", "input_channels", "num_input_channels", "out_channels", "num_classes", "num_outputs"):
        if hasattr(mod, name):
            attrs[name] = getattr(mod, name)
    if attrs:
        _print_inferred("Module attributes found", attrs)

    convs = []
    linears = []
    for nm, m in mod.named_modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            convs.append((nm, m))
        elif isinstance(m, torch.nn.Linear):
            linears.append((nm, m))

    if convs:
        fn, fm = convs[0]
        ln, lm = convs[-1]
        _print_inferred("First conv layer", f"{fn}: weight shape={tuple(getattr(fm, 'weight').size())}")
        _print_inferred("Last conv layer", f"{ln}: weight shape={tuple(getattr(lm, 'weight').size())}")
        in_ch = getattr(fm, "in_channels", None) or fm.weight.size(1)
        out_ch = getattr(lm, "out_channels", None) or lm.weight.size(0)
        _print_inferred("Inferred input channels", in_ch)
        _print_inferred("Inferred output channels", out_ch)
        spatial_ndim = 1 if isinstance(fm, torch.nn.Conv1d) else 3 if isinstance(fm, torch.nn.Conv3d) else 2
    elif linears:
        fn, fm = linears[0]
        ln, lm = linears[-1]
        _print_inferred("First linear layer", f"{fn}: weight shape={tuple(fm.weight.size())}")
        _print_inferred("Last linear layer", f"{ln}: weight shape={tuple(lm.weight.size())}")
        _print_inferred("Inferred input features", fm.in_features)
        _print_inferred("Inferred output features", lm.out_features)
        spatial_ndim = 0
    else:
        _print_inferred("Module inference", "no Conv/Linear layers found to infer dimensions")
        spatial_ndim = None

    # attempt a safe forward pass with a dummy input when possible
    if spatial_ndim is None:
        return

    try:
        mod_cpu = mod.to("cpu")
        mod_cpu.eval()
        if spatial_ndim == 0:
            # linear-only model
            in_features = fm.in_features if linears else (getattr(mod_cpu, "in_features", None) or 1)
            dummy = torch.randn(1, in_features, dtype=torch.float32)
        elif spatial_ndim == 1:
            dummy = torch.randn(1, in_ch, 64, dtype=torch.float32)
        elif spatial_ndim == 2:
            dummy = torch.randn(1, in_ch, 64, 64, dtype=torch.float32)
        else:
            dummy = torch.randn(1, in_ch, 16, 64, 64, dtype=torch.float32)  # conv3d guess

            with torch.no_grad():
                out = mod_cpu(dummy)
            if isinstance(out, torch.Tensor):
                _print_inferred("Example forward output shape", tuple(out.size()))
            else:
                _print_inferred("Example forward output", type(out))
    except Exception as e:
        _print_inferred("Dummy forward attempt failed", str(e))


# Main inference entry: handle dict, module, or object with state_dict
def _run_inference(obj):
    # Reuse the previously-loaded `obj` and attempt to infer IO dims.
    if isinstance(obj, torch.nn.Module):
        print("Inferring IO dims from torch.nn.Module...")
        _infer_from_module(obj)
    elif isinstance(obj, dict):
        # try to locate a state_dict inside the dict (same candidates as earlier)
        candidate_keys = ("state_dict", "model_state_dict", "model", "state")
        sd = None
        for k in candidate_keys:
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                print(f"Using '{k}' entry as state-dict for inference")
                break
        if sd is None and all(isinstance(v, torch.Tensor) for v in obj.values()):
            sd = obj
            print("Top-level dict looks like a state-dict; using it for inference")
        if sd is not None:
            _infer_from_state_dict(sd)
        else:
            print("No usable state-dict found in the loaded dict; cannot infer IO dims")
    elif hasattr(obj, "state_dict") and callable(obj.state_dict):
        try:
            sd = obj.state_dict()
            print("Using object's state_dict() for inference")
            _infer_from_state_dict(sd)
        except Exception as e:
            print("Calling state_dict() failed for inference:", e)
    else:
        print("Cannot infer input/output dimensions for this object type.")


def main():

    models_path = "/home/users/bradlesc/projects/ClimSim/models/models/saved_model/"
    model_names = ["v4plus_unet_nonaggressive_cliprh_huber_rop2_r3", "v2rh_unet_nonaggressive_cliprh_huber", "v2rh_unet_nonaggressive_cliprh_huber_rop2", "v2rh_unet_nonaggressive_cliprh_mae", "v4plus_unet_nonaggressive_cliprh_huber", "v4plus_unet_nonaggressive_cliprh_mae", "v5_unet_nonaggressive_cliprh_huber", "v5_unet_nonaggressive_cliprh_huber_rop2_r2", "v5_unet_nonaggressive_cliprh_mae"] 

    for model_name in model_names:
        model_path = os.path.join(models_path, model_name, "model.pt")
        # Create a log filename based on the model file and timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = os.path.splitext(os.path.basename(model_path))[0]
        log_dir = os.path.dirname(model_path) or os.getcwd()
        log_filename = f"{model_name}.log"
        log_path = os.path.join(log_filename)

        # Redirect stdout to the log file for the duration of inspection
        with open(log_path, 'w') as log_f:
            with contextlib.redirect_stdout(log_f):
                # Run the inspection (this will write into the log file)
                _run_inspection(log_f=log_f, model_path=model_path)

        # Re-open the model to run inference/IO-dim probing and append that to log
        # (doing a second open so outputs are grouped; callers can modify behavior)
        try:
            obj = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Failed to reload model for inference probing: {e}")
            obj = None

        # Append inference results to the same log file
        with open(log_path, 'a') as log_f:
            with contextlib.redirect_stdout(log_f):
                if obj is not None:
                    _run_inference(obj)

        # Briefly notify the user on the original stdout where the log was written
        print(f"Inspection complete — saved output to: {log_path}")


if __name__ == '__main__':
    main()