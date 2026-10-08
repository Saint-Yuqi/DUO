"""[duo.adapters — verbatim copy, do not edit here]
source : /tmp/robotwin_native_dual/install_native_dual.py
copied : 2026-10-08 from quic5000 (as it ran; snapshot date 2026-10-06)
ledger : openwam_native_dual_clean2500
note   : patches joint_self_attn.py / ckpt_model_loader.py of the OpenWAM clone and writes configs/model/action_backbone/native_dual_dit.yaml
Edit the canonical copy in the training workspace, then re-copy; the ledger row points at this file + the source path.
"""
from pathlib import Path

root = Path('/tmp/robotwin_native_dual/source')
arch = root / 'openwam/model/architectures/dual_system/joint_self_attn.py'
s = arch.read_text()
s = s.replace('from openwam.model.action_backbone.separate_action_dit import ActionDiT',
              'from openwam.model.action_backbone.separate_action_dit import ActionDiT\n'
              'from openwam.model.action_backbone.native_dual_dit import NativeDualActionDiT, NativeDualMoTDriver')
s = s.replace('self.action_backbone = ActionDiT(',
              'action_class = NativeDualActionDiT if cfg.get("name") == "native_dual_dit" else ActionDiT\n'
              '        self.action_backbone = action_class(')
s = s.replace('self._mot_driver = DualSystemMoTDriver(',
              'driver_class = NativeDualMoTDriver if isinstance(self.action_backbone, NativeDualActionDiT) else DualSystemMoTDriver\n'
              '        self._mot_driver = driver_class(')
s = s.replace('from openwam.model.action_backbone.native_dual_dit import NativeDualActionDiT, NativeDualMoTDriver\nfrom openwam.model.action_backbone.native_dual_dit import NativeDualActionDiT, NativeDualMoTDriver', 'from openwam.model.action_backbone.native_dual_dit import NativeDualActionDiT, NativeDualMoTDriver')
arch.write_text(s)

loader = root / 'openwam/train/utils/ckpt_model_loader.py'
s = loader.read_text()
start = s.find('        elif (OmegaConf.select(model_cfg, "action_backbone.name") == "native_dual_dit"')
if start >= 0:
    end = s.index('        else:\n            architecture.load_checkpoint(weights)\n', start)
    s = s[:start] + s[end:]
needle = '        else:\n            architecture.load_checkpoint(weights)\n'
replacement = """        elif (OmegaConf.select(model_cfg, "action_backbone.name") == "native_dual_dit"
              and OmegaConf.select(ckpt_cfg, "model.action_backbone.name") != "native_dual_dit"):
            from safetensors.torch import load_file
            from openwam.model.action_backbone.native_dual_dit import native_dual_warm_start

            compatible = native_dual_warm_start(load_file(weights), architecture.state_dict())
            has_meta = any(p.device.type == "meta" for p in architecture.parameters())
            result = architecture.load_state_dict(compatible, strict=False, assign=has_meta)
            if result.unexpected_keys or any(not key.startswith(("action_backbone.left_cross.",
                    "action_backbone.right_cross.")) for key in result.missing_keys):
                raise RuntimeError(f"Unexpected native dual warm-start mismatch: {result}")
            n_action = sum(key.startswith("action_backbone.") for key in compatible)
            print(f"[{tag}] transferred video/proprio and {n_action} native action tensors; only new CrossArm tensors initialized", flush=True)
        else:
            architecture.load_checkpoint(weights)
"""
assert s.count(needle) == 1
loader.write_text(s.replace(needle, replacement))

config = root / 'configs/model/action_backbone/native_dual_dit.yaml'
config.write_text('name: native_dual_dit\ndim: 1024\nffn_dim: 4096\nshift_action: 5.0\n')
print('Installed native dual OpenWAM backbone')
