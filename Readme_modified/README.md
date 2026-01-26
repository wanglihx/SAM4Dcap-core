# Uncommitted Changes Summary

Summary of current uncommitted/untracked changes under `/root/TVB`, gathered via `git status` and `git diff`.

## AddBiomechanics
| File/Path | Change Type | Summary | Status |
| --- | --- | --- | --- |
| frontend/src/index.tsx | Modified | Added local preview-only mode (NimbleStandaloneReact); load local `preview.bin` via `REACT_APP_LOCAL_PREVIEW_URL` and skip AWS initialization | Modified, unstaged |
| frontend/public/localdata | Added (untracked) | Symlink to `/root/TVB/SAM4Dcap/pipeline1` for local preview data | Untracked |
| frontend/src/aws-exports.ts | Added (untracked) | Local AWS Amplify config file (not versioned) | Untracked |

## MHR
| File/Path | Change Type | Summary | Status |
| --- | --- | --- | --- |
| tools/mhr_smpl_conversion/conversion.py | Modified | Skip cm→m scaling in mhr2smpl; assume source vertices already in meters | Modified, unstaged |
| tools/mhr_smpl_conversion/file_assets.py | Modified | Asset paths now resolved from module directory to avoid relative-path load issues | Modified, unstaged |

## SMPL2AddBiomechanics
| File/Path | Change Type | Summary | Status |
| --- | --- | --- | --- |
| smpl2ab/smpl2addbio.py | Modified | Lazy-import Smpl2osim only when `use_osso` is enabled to reduce optional dependencies | Modified, unstaged |
| smpl2ab/utils/smpl_utils.py | Modified | Load SMPL via `smplx.SMPL` with fixed model path `/root/TVB/MHRtoSMPL/SMPL_NEUTRAL_chumpy_free.pkl` (no `model_type` switch) | Modified, unstaged |

## opencap-core
| File/Path | Change Type | Summary | Status |
| --- | --- | --- | --- |
| main.py | Modified | `generateVideo` now respects caller argument instead of forcing True | Modified, unstaged |
| utils.py | Modified | `getMMposeDirectory` honors env `MMPOSE_DIRECTORY/MMPOSE_DIR`; default to bundled `mmpose` directory, keep legacy host path fallback | Modified, unstaged |
| utilsChecker.py | Modified | Sync video output name now uses `trialName` prefix to keep paths consistent across varying input filenames | Modified, unstaged |
| utilsDetector.py | Modified | MMpose detection/pose choose CUDA device when available and pass device through; added torch import | Modified, unstaged |
| utilsOpenSim.py | Modified | Scale/IK first call API, fallback to `opensim-cmd` on failure; clear `pathToSubject` to support absolute paths | Modified, unstaged |

## sam-body4d
| File/Path | Change Type | Summary | Status |
| --- | --- | --- | --- |
| configs/body4d.yaml | Modified | Set `ckpt_root` to `/root/TVB`; Depth Anything weight path points to `Depth-Anything-V2-Large` subfolder | Modified, unstaged |
| models/sam_3d_body/sam_3d_body/models/heads/mhr_head.py | Modified | Force float32 and disable autocast in MHR forward to avoid bfloat16 sparse-matrix errors | Modified, unstaged |
| scripts/run_batch.py | Added | New batch CLI: loads SAM-3/SAM-3D-Body + ViTDet, disables occlusion recovery for speed, supports batch 4D reconstruction | Untracked |

